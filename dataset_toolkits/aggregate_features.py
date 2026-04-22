import argparse
import copy
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import utils3d
from easydict import EasyDict as edict
from tqdm import tqdm


torch.set_grad_enabled(False)


def compute_zbuf_weights(
    valid,
    flat_idx,
    depth,
    h,
    w,
    selection,
    depth_tolerance,
    topk,
    soft_tau,
):
    """Return per-view per-voxel z-buffer weights in [0, 1]."""
    bs, n_voxels = valid.shape
    weights = torch.zeros_like(depth)

    for b in range(bs):
        valid_b = valid[b]
        if not valid_b.any():
            continue

        idx_b = flat_idx[b, valid_b]
        depth_b = depth[b, valid_b]

        min_depth = torch.full(
            (h * w,),
            torch.inf,
            device=depth.device,
            dtype=depth.dtype,
        )
        min_depth.scatter_reduce_(0, idx_b, depth_b, reduce='amin', include_self=True)

        nearest_depth = min_depth[idx_b]

        if selection == 'nearest':
            keep_valid = depth_b <= (nearest_depth + depth_tolerance)
            weights_valid = keep_valid.to(depth.dtype)
        elif selection == 'topk':
            keep_valid = torch.zeros_like(depth_b, dtype=torch.bool)
            unique_cells = torch.unique(idx_b)
            for cell in unique_cells:
                cell_mask = idx_b == cell
                cell_depth = depth_b[cell_mask]
                k = min(topk, int(cell_depth.numel()))
                kth_depth = torch.topk(cell_depth, k=k, largest=False).values.max()
                keep_valid[cell_mask] = cell_depth <= (kth_depth + depth_tolerance)
            weights_valid = keep_valid.to(depth.dtype)
        elif selection == 'soft':
            # Keep all visible voxels but downweight those farther than cell front depth.
            delta = torch.clamp(depth_b - nearest_depth, min=0.0)
            weights_valid = torch.exp(-delta / max(soft_tau, 1e-8))
        else:
            raise ValueError(
                f'Unsupported zbuf_selection="{selection}". Supported: ["nearest", "topk", "soft"]'
            )

        row = torch.zeros(n_voxels, device=depth.device, dtype=depth.dtype)
        row[valid_b] = weights_valid
        weights[b] = row

    return weights


def load_metadata(opt):
    metadata_path = os.path.join(opt.output_dir, 'metadata.csv')
    if not os.path.exists(metadata_path):
        raise ValueError('metadata.csv not found')

    metadata = pd.read_csv(metadata_path)
    if opt.instances is not None:
        with open(opt.instances, 'r') as f:
            instances = f.read().splitlines()
        metadata = metadata[metadata['sha256'].isin(instances)]
    else:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        if f'feature_{opt.output_feature_name}' in metadata.columns:
            metadata = metadata[metadata[f'feature_{opt.output_feature_name}'] == False]
        metadata = metadata[metadata['voxelized'] == True]
        metadata = metadata[metadata['rendered'] == True]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    return metadata[start:end]


def load_camera_tensors(transforms_path):
    with open(transforms_path, 'r') as f:
        transforms_metadata = json.load(f)

    extrinsics_list = []
    intrinsics_list = []
    for frame in transforms_metadata['frames']:
        c2w = torch.tensor(frame['transform_matrix'], dtype=torch.float32)
        c2w[:3, 1:3] *= -1
        extrinsics = torch.inverse(c2w)

        fov = frame['camera_angle_x']
        intrinsics = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(fov), torch.tensor(fov))

        extrinsics_list.append(extrinsics)
        intrinsics_list.append(intrinsics)

    return torch.stack(extrinsics_list, dim=0), torch.stack(intrinsics_list, dim=0)


def aggregate_one_object(opt, sha256):
    view_feature_path = os.path.join(opt.output_dir, 'view_features', opt.model, f'{sha256}.npz')
    transforms_path = os.path.join(opt.output_dir, 'renders', sha256, 'transforms.json')
    voxel_path = os.path.join(opt.output_dir, 'voxels', f'{sha256}.ply')

    if not os.path.exists(view_feature_path):
        raise FileNotFoundError(f'Missing cached per-view features: {view_feature_path}')
    if not os.path.exists(transforms_path):
        raise FileNotFoundError(f'Missing transforms metadata: {transforms_path}')
    if not os.path.exists(voxel_path):
        raise FileNotFoundError(f'Missing voxel ply: {voxel_path}')

    patchtokens = torch.from_numpy(np.load(view_feature_path)['patchtokens']).float().cuda()
    extrinsics, intrinsics = load_camera_tensors(transforms_path)
    extrinsics = extrinsics.cuda()
    intrinsics = intrinsics.cuda()

    if patchtokens.shape[0] != extrinsics.shape[0]:
        raise ValueError(
            f'View count mismatch for {sha256}: patchtokens={patchtokens.shape[0]}, cameras={extrinsics.shape[0]}'
        )

    positions = utils3d.io.read_ply(voxel_path)[0]
    positions = torch.from_numpy(positions).float().cuda()
    indices = ((positions + 0.5) * 64).long()
    assert torch.all(indices >= 0) and torch.all(indices < 64), 'Some vertices are out of bounds'

    n_views = patchtokens.shape[0]
    n_voxels = positions.shape[0]
    aggregated_sum = None
    visibility_count = None
    fallback_sum = None
    fallback_count = None
    view_count = 0

    positions_h = torch.cat([
        positions,
        torch.ones(positions.shape[0], 1, device=positions.device, dtype=positions.dtype),
    ], dim=1)

    for i in range(0, n_views, opt.batch_size):
        patchtokens_batch = patchtokens[i:i + opt.batch_size]
        batch_extrinsics = extrinsics[i:i + opt.batch_size]
        batch_intrinsics = intrinsics[i:i + opt.batch_size]
        bs = patchtokens_batch.shape[0]

        uv = utils3d.torch.project_cv(positions, batch_extrinsics, batch_intrinsics)[0] * 2 - 1
        cam_space = torch.matmul(batch_extrinsics, positions_h.t()).transpose(1, 2)
        depth = cam_space[..., 2]

        in_bounds = (
            (uv[..., 0] >= (-1.0 + opt.visibility_margin)) &
            (uv[..., 0] <= (1.0 - opt.visibility_margin)) &
            (uv[..., 1] >= (-1.0 + opt.visibility_margin)) &
            (uv[..., 1] <= (1.0 - opt.visibility_margin))
        )
        base_visibility_mask = (depth > 0) & in_bounds
        visibility_weights = base_visibility_mask.to(patchtokens_batch.dtype)

        if opt.aggregation_mode == 'visible_only_zbuf':
            # Z-buffer style visibility on voxel projections: keep voxels that are
            # closest to the camera per projected patch cell.
            h = patchtokens_batch.shape[-2]
            w = patchtokens_batch.shape[-1]
            x = ((uv[..., 0] + 1) * 0.5 * (w - 1)).round().long()
            y = ((uv[..., 1] + 1) * 0.5 * (h - 1)).round().long()
            in_grid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            valid = base_visibility_mask & in_grid
            flat_idx = y * w + x

            visibility_weights = compute_zbuf_weights(
                valid=valid,
                flat_idx=flat_idx,
                depth=depth,
                h=h,
                w=w,
                selection=opt.zbuf_selection,
                depth_tolerance=opt.zbuf_depth_tolerance,
                topk=opt.zbuf_topk,
                soft_tau=opt.zbuf_soft_tau,
            ).to(patchtokens_batch.dtype)

            if opt.zbuf_fallback_mode == 'visible_only':
                if fallback_sum is None:
                    fallback_sum = torch.zeros(
                        (n_voxels, patchtokens_batch.shape[1]),
                        device=patchtokens_batch.device,
                        dtype=torch.float32,
                    )
                if fallback_count is None:
                    fallback_count = torch.zeros(
                        (n_voxels, 1),
                        device=patchtokens_batch.device,
                        dtype=torch.float32,
                    )

        sampled = F.grid_sample(
            patchtokens_batch,
            uv.unsqueeze(1),
            mode='bilinear',
            align_corners=False,
        ).squeeze(2).permute(0, 2, 1)

        if aggregated_sum is None:
            aggregated_sum = torch.zeros(
                (n_voxels, sampled.shape[-1]),
                device=sampled.device,
                dtype=torch.float32,
            )

        if opt.aggregation_mode == 'mean':
            aggregated_sum += sampled.sum(dim=0, dtype=torch.float32)
            view_count += bs
        elif opt.aggregation_mode in ('visible_only', 'visible_only_zbuf'):
            aggregated_sum += (sampled * visibility_weights[..., None]).sum(dim=0, dtype=torch.float32)
            if visibility_count is None:
                visibility_count = torch.zeros(
                    (n_voxels, 1),
                    device=sampled.device,
                    dtype=torch.float32,
                )
            visibility_count += visibility_weights.sum(dim=0, dtype=torch.float32)[:, None]

            if opt.aggregation_mode == 'visible_only_zbuf' and opt.zbuf_fallback_mode == 'visible_only':
                base_visibility_float = base_visibility_mask.to(sampled.dtype)
                fallback_sum += (sampled * base_visibility_float[..., None]).sum(dim=0, dtype=torch.float32)
                fallback_count += base_visibility_float.sum(dim=0, dtype=torch.float32)[:, None]
        else:
            raise ValueError(
                f'Unsupported aggregation_mode="{opt.aggregation_mode}". Supported: ["mean", "visible_only", "visible_only_zbuf"]'
            )

    if aggregated_sum is None:
        raise ValueError(f'No view data found for {sha256}')

    if opt.aggregation_mode == 'mean':
        denominator = max(view_count, 1)
        aggregated = aggregated_sum / denominator
    else:
        aggregated = aggregated_sum / (visibility_count + opt.visibility_eps)

    if (
        opt.aggregation_mode == 'visible_only_zbuf'
        and opt.zbuf_fallback_mode == 'visible_only'
        and opt.zbuf_min_visibility_count > 0
    ):
        fallback_aggregated = fallback_sum / (fallback_count + opt.visibility_eps)
        low_support = visibility_count[:, 0] < opt.zbuf_min_visibility_count
        if low_support.any():
            aggregated[low_support] = fallback_aggregated[low_support]

    np.savez_compressed(
        os.path.join(opt.output_dir, 'features', opt.output_feature_name, f'{sha256}.npz'),
        indices=indices.cpu().numpy().astype(np.uint8),
        patchtokens=aggregated.cpu().numpy().astype(np.float16),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory containing metadata/renders/voxels/view_features')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--model', type=str, default='dinov2_vitl14_reg', help='Feature extraction model')
    parser.add_argument('--instances', type=str, default=None, help='Path to file with one sha256 per line')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--aggregation_mode', type=str, default='visible_only', choices=['visible_only', 'visible_only_zbuf', 'mean'],
                        help='Multiview feature aggregation mode.')
    parser.add_argument('--visibility_eps', type=float, default=1e-6,
                        help='Small constant used in visible-only denominator to avoid division by zero.')
    parser.add_argument('--visibility_margin', type=float, default=0.0,
                        help='In-bound margin in normalized grid coordinates for visibility mask.')
    parser.add_argument('--zbuf_depth_tolerance', type=float, default=1e-3,
                        help='Depth tolerance for visible_only_zbuf mode in camera-space depth units.')
    parser.add_argument('--zbuf_selection', type=str, default='nearest', choices=['nearest', 'topk', 'soft'],
                        help='Cell-level z-buffer policy for visible_only_zbuf mode.')
    parser.add_argument('--zbuf_topk', type=int, default=1,
                        help='Top-k nearest depths kept per patch cell when zbuf_selection=topk.')
    parser.add_argument('--zbuf_soft_tau', type=float, default=5e-3,
                        help='Depth decay scale for zbuf_selection=soft. Larger values are less aggressive.')
    parser.add_argument('--zbuf_fallback_mode', type=str, default='none', choices=['none', 'visible_only'],
                        help='Optional fallback policy for low-support voxels in visible_only_zbuf mode.')
    parser.add_argument('--zbuf_min_visibility_count', type=float, default=0.0,
                        help='If fallback enabled, voxels below this aggregated zbuf support use visible_only fallback.')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--output_feature_name', type=str, default=None,
                        help='Output feature directory under features/. Default: <model>_<aggregation_mode>.')
    parser.add_argument('--timing_log_interval', type=int, default=100,
                        help='Print timing summary every N processed objects (<=0 disables periodic logs).')
    parser.add_argument('--max_failures', type=int, default=-1,
                        help='Maximum allowed per-object failures before exiting non-zero. Set to -1 to disable.')
    opt = parser.parse_args()
    opt = edict(vars(opt))

    if opt.zbuf_topk < 1:
        raise ValueError('--zbuf_topk must be >= 1')

    if opt.output_feature_name is None:
        opt.output_feature_name = f'{opt.model}_{opt.aggregation_mode}'

    feature_dir = os.path.join(opt.output_dir, 'features', opt.output_feature_name)
    os.makedirs(feature_dir, exist_ok=True)

    metadata = load_metadata(opt)
    records = []

    sha256s = list(metadata['sha256'].values)
    for sha256 in copy.copy(sha256s):
        if os.path.exists(os.path.join(feature_dir, f'{sha256}.npz')):
            records.append({'sha256': sha256, f'feature_{opt.output_feature_name}': True})
            sha256s.remove(sha256)

    script_start = time.perf_counter()
    processed = 0
    failed = 0
    timing_totals = {
        'wait_for_load': 0.0,
        'aggregate': 0.0,
        'object_total': 0.0,
    }

    load_queue = Queue(maxsize=4)
    try:
        with ThreadPoolExecutor(max_workers=8) as loader_executor:
            def loader(sha256):
                try:
                    load_queue.put(sha256)
                except Exception as e:
                    print(f'Error staging data for {sha256}: {e}')

            loader_executor.map(loader, sha256s)

            for _ in tqdm(range(len(sha256s)), desc='Aggregating features'):
                object_start = time.perf_counter()
                t_wait = time.perf_counter()
                sha256 = load_queue.get()
                timing_totals['wait_for_load'] += time.perf_counter() - t_wait

                t_agg = time.perf_counter()
                try:
                    aggregate_one_object(opt, sha256)
                    records.append({'sha256': sha256, f'feature_{opt.output_feature_name}': True})
                except Exception as e:
                    failed += 1
                    print(f'Error processing {sha256}: {e}')
                timing_totals['aggregate'] += time.perf_counter() - t_agg

                processed += 1
                timing_totals['object_total'] += time.perf_counter() - object_start
                if opt.timing_log_interval > 0 and (
                    processed % opt.timing_log_interval == 0 or processed == len(sha256s)
                ):
                    print(
                        '[TIMING][AGG] '
                        f'processed={processed}/{len(sha256s)} '
                        f'avg_object_sec={timing_totals["object_total"] / processed:.3f} '
                        f'avg_wait_sec={timing_totals["wait_for_load"] / processed:.3f} '
                        f'avg_aggregate_sec={timing_totals["aggregate"] / processed:.3f}'
                    )
    except Exception as e:
        print(f'Error happened during processing: {e}')

    elapsed = time.perf_counter() - script_start
    succeeded = len(records)
    print(
        f'[TIMING][AGG][SUMMARY] processed={processed} '
        f'succeeded={succeeded} failed={failed} total_sec={elapsed:.3f}'
    )

    if opt.max_failures >= 0 and failed > opt.max_failures:
        raise RuntimeError(
            f'aggregate_features failed on {failed} objects (max_failures={opt.max_failures}).'
        )

    records = pd.DataFrame.from_records(records)
    records.to_csv(os.path.join(opt.output_dir, f'feature_{opt.output_feature_name}_{opt.rank}.csv'), index=False)


if __name__ == '__main__':
    main()
