import argparse
import copy
import json
import os
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


def load_camera_tensors(transforms_path):
    """Load camera extrinsics/intrinsics from a transforms.json file.

    Returns:
        extrinsics: torch.FloatTensor [V, 4, 4]
        intrinsics: torch.FloatTensor [V, 3, 3]
    """
    with open(transforms_path, 'r') as f:
        transforms_metadata = json.load(f)

    frames = transforms_metadata['frames']
    extrinsics_list = []
    intrinsics_list = []

    for frame in frames:
        c2w = torch.tensor(frame['transform_matrix'], dtype=torch.float32)
        c2w[:3, 1:3] *= -1
        extrinsics = torch.inverse(c2w)

        fov = frame['camera_angle_x']
        intrinsics = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(fov), torch.tensor(fov))

        extrinsics_list.append(extrinsics)
        intrinsics_list.append(intrinsics)

    extrinsics = torch.stack(extrinsics_list, dim=0)
    intrinsics = torch.stack(intrinsics_list, dim=0)
    return extrinsics, intrinsics


def load_voxel_positions_and_indices(voxel_path):
    """Load voxel positions from ply and compute uint8 voxel indices.

    Returns:
        positions_cuda: torch.FloatTensor [N, 3] on CUDA
        indices_u8: np.ndarray [N, 3] uint8
    """
    positions = utils3d.io.read_ply(voxel_path)[0]
    positions_cuda = torch.from_numpy(positions).float().cuda()

    # Reproduce old indexing logic exactly.
    indices = ((positions_cuda + 0.5) * 64).long()
    assert torch.all(indices >= 0) and torch.all(indices < 64), 'Some vertices are out of bounds'

    indices_u8 = indices.cpu().numpy().astype(np.uint8)
    return positions_cuda, indices_u8


def sample_patchtokens_for_all_views(patchtokens, positions, extrinsics, intrinsics, batch_size):
    """Project 3D voxels to each view and sample cached DINO patch maps.

    Args:
        patchtokens: torch.FloatTensor [V, C, H_patch, W_patch]
        positions: torch.FloatTensor [N, 3] (CUDA)
        extrinsics: torch.FloatTensor [V, 4, 4]
        intrinsics: torch.FloatTensor [V, 3, 3]

    Returns:
        sampled: torch.FloatTensor [V, N, C]
    """
    assert patchtokens.ndim == 4, f'Expected patchtokens [V,C,H,W], got {patchtokens.shape}'
    n_views = patchtokens.shape[0]

    sampled_list = []
    for i in range(0, n_views, batch_size):
        patchtokens_batch = patchtokens[i:i + batch_size]  # [B, C, H_patch, W_patch]
        batch_extrinsics = extrinsics[i:i + batch_size]    # [B, 4, 4]
        batch_intrinsics = intrinsics[i:i + batch_size]    # [B, 3, 3]

        # uv shape: [B, N, 2], normalized to [-1, 1] for grid_sample
        uv = utils3d.torch.project_cv(positions, batch_extrinsics, batch_intrinsics)[0] * 2 - 1

        # grid_sample output shape: [B, C, 1, N]
        sampled = F.grid_sample(
            patchtokens_batch,
            uv.unsqueeze(1),
            mode='bilinear',
            align_corners=False,
        )

        # Convert to [B, N, C]
        sampled = sampled.squeeze(2).permute(0, 2, 1).contiguous()
        sampled_list.append(sampled)

    sampled = torch.cat(sampled_list, dim=0)
    assert sampled.shape[0] == n_views
    return sampled


def aggregate_patchtokens(sampled_patchtokens, agg):
    """Aggregate per-view sampled features.

    Args:
        sampled_patchtokens: torch.FloatTensor [V, N, C]
        agg: aggregation mode

    Returns:
        aggregated: np.ndarray [N, C] float16
    """
    if agg != 'mean':
        raise ValueError(f"Unsupported --agg '{agg}'. Only 'mean' is implemented in this refactor step.")

    # Baseline behavior equivalent to np.mean(sampled_patchtokens, axis=0)
    aggregated = sampled_patchtokens.mean(dim=0)
    return aggregated.cpu().numpy().astype(np.float16)


def load_metadata(opt):
    """Load and filter metadata with behavior close to extract_feature.py."""
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
        if f'feature_{opt.model}' in metadata.columns:
            metadata = metadata[metadata[f'feature_{opt.model}'] == False]
        metadata = metadata[metadata['voxelized'] == True]
        metadata = metadata[metadata['rendered'] == True]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    return metadata


def process_one_object(opt, sha256):
    """Aggregate cached per-view patch maps into voxel-level features for one object."""
    view_feature_path = os.path.join(opt.output_dir, 'view_features', opt.model, f'{sha256}.npz')
    transforms_path = os.path.join(opt.output_dir, 'renders', sha256, 'transforms.json')
    voxel_path = os.path.join(opt.output_dir, 'voxels', f'{sha256}.ply')

    if not os.path.exists(view_feature_path):
        raise FileNotFoundError(f'Missing cached per-view features: {view_feature_path}')
    if not os.path.exists(transforms_path):
        raise FileNotFoundError(f'Missing camera metadata: {transforms_path}')
    if not os.path.exists(voxel_path):
        raise FileNotFoundError(f'Missing voxel ply: {voxel_path}')

    cached = np.load(view_feature_path)
    assert 'patchtokens' in cached, f"'patchtokens' not found in {view_feature_path}"

    # patchtokens shape: [V, C, H_patch, W_patch]
    patchtokens = torch.from_numpy(cached['patchtokens']).float().cuda()

    extrinsics, intrinsics = load_camera_tensors(transforms_path)
    extrinsics = extrinsics.cuda()
    intrinsics = intrinsics.cuda()

    n_views_patchtokens = patchtokens.shape[0]
    n_views_cameras = extrinsics.shape[0]
    assert n_views_patchtokens == n_views_cameras, (
        f'View count mismatch for {sha256}: patchtokens has {n_views_patchtokens}, '
        f'cameras has {n_views_cameras}'
    )

    positions, indices_u8 = load_voxel_positions_and_indices(voxel_path)
    sampled_patchtokens = sample_patchtokens_for_all_views(
        patchtokens=patchtokens,
        positions=positions,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        batch_size=opt.batch_size,
    )
    # sampled_patchtokens: [V, N, C]

    aggregated_patchtokens = aggregate_patchtokens(sampled_patchtokens, agg=opt.agg)
    # aggregated_patchtokens: [N, C]

    save_path = os.path.join(opt.output_dir, 'features', opt.model, f'{sha256}.npz')
    np.savez_compressed(
        save_path,
        indices=indices_u8,
        patchtokens=aggregated_patchtokens,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory containing metadata.csv and cached/intermediate outputs')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--model', type=str, default='dinov2_vitl14_reg',
                        help='Feature extraction model')
    parser.add_argument('--instances', type=str, default=None,
                        help='Path to text file containing sha256 IDs to process (one per line)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--agg', type=str, default='mean',
                        help="Aggregation mode. Only 'mean' is currently supported.")
    opt = parser.parse_args()
    opt = edict(vars(opt))

    if opt.agg != 'mean':
        raise ValueError(f"Unsupported --agg '{opt.agg}'. Only 'mean' is implemented in this refactor step.")

    feature_dir = os.path.join(opt.output_dir, 'features', opt.model)
    os.makedirs(feature_dir, exist_ok=True)

    metadata = load_metadata(opt)
    records = []

    # Filter out already processed objects, preserving old bookkeeping behavior.
    sha256s = list(metadata['sha256'].values)
    for sha256 in copy.copy(sha256s):
        if os.path.exists(os.path.join(feature_dir, f'{sha256}.npz')):
            records.append({'sha256': sha256, f'feature_{opt.model}': True})
            sha256s.remove(sha256)

    # Preload object references similarly to the old producer/consumer pattern.
    load_queue = Queue(maxsize=4)
    try:
        with ThreadPoolExecutor(max_workers=8) as loader_executor:
            def loader(sha256):
                try:
                    view_feature_path = os.path.join(opt.output_dir, 'view_features', opt.model, f'{sha256}.npz')
                    transforms_path = os.path.join(opt.output_dir, 'renders', sha256, 'transforms.json')
                    voxel_path = os.path.join(opt.output_dir, 'voxels', f'{sha256}.ply')
                    load_queue.put((sha256, view_feature_path, transforms_path, voxel_path))
                except Exception as e:
                    print(f'Error staging data for {sha256}: {e}')

            loader_executor.map(loader, sha256s)

            for _ in tqdm(range(len(sha256s)), desc='Aggregating features'):
                sha256, _, _, _ = load_queue.get()
                try:
                    process_one_object(opt, sha256)
                    records.append({'sha256': sha256, f'feature_{opt.model}': True})
                except Exception as e:
                    print(f'Error processing {sha256}: {e}')
    except Exception as e:
        print(f'Error happened during processing: {e}')

    records = pd.DataFrame.from_records(records)
    records.to_csv(os.path.join(opt.output_dir, f'feature_{opt.model}_{opt.rank}.csv'), index=False)


if __name__ == '__main__':
    main()
