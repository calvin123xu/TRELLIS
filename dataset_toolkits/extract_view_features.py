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
from easydict import EasyDict as edict
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


torch.set_grad_enabled(False)


def load_rgba_image(output_dir, sha256, frame):
    image_path = os.path.join(output_dir, 'renders', sha256, frame['file_path'])
    image = Image.open(image_path)
    image = image.resize((518, 518), Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[:, :, :3] * image[:, :, 3:]
    return torch.from_numpy(image).permute(2, 0, 1).float()


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
        metadata = metadata[metadata['voxelized'] == True]
        metadata = metadata[metadata['rendered'] == True]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    return metadata[start:end]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True, help='Directory containing metadata/renders')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--model', type=str, default='dinov2_vitl14_reg', help='Feature extraction model')
    parser.add_argument('--instances', type=str, default=None, help='Path to file with one sha256 per line')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--timing_log_interval', type=int, default=100,
                        help='Print timing summary every N processed objects (<=0 disables periodic logs).')
    opt = parser.parse_args()
    opt = edict(vars(opt))

    feature_name = opt.model
    save_dir = os.path.join(opt.output_dir, 'view_features', feature_name)
    os.makedirs(save_dir, exist_ok=True)

    dinov2_model = torch.hub.load('facebookresearch/dinov2', opt.model)
    dinov2_model.eval().cuda()
    normalize = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    n_patch = 518 // 14

    metadata = load_metadata(opt)
    records = []

    sha256s = list(metadata['sha256'].values)
    for sha256 in copy.copy(sha256s):
        if os.path.exists(os.path.join(save_dir, f'{sha256}.npz')):
            records.append({'sha256': sha256, f'view_feature_{feature_name}': True})
            sha256s.remove(sha256)

    script_start = time.perf_counter()
    processed = 0
    timing_totals = {
        'wait_for_load': 0.0,
        'model_forward': 0.0,
        'save': 0.0,
        'object_total': 0.0,
    }

    load_queue = Queue(maxsize=4)
    try:
        with ThreadPoolExecutor(max_workers=8) as loader_executor:
            def loader(sha256):
                try:
                    with open(os.path.join(opt.output_dir, 'renders', sha256, 'transforms.json'), 'r') as f:
                        render_metadata = json.load(f)
                    frames = render_metadata['frames']
                    with ThreadPoolExecutor(max_workers=16) as frame_executor:
                        images = list(frame_executor.map(
                            lambda frame: load_rgba_image(opt.output_dir, sha256, frame),
                            frames,
                        ))
                    images = [normalize(image) for image in images]
                    load_queue.put((sha256, images))
                except Exception as e:
                    print(f'Error loading data for {sha256}: {e}')

            loader_executor.map(loader, sha256s)

            for _ in tqdm(range(len(sha256s)), desc='Extracting per-view features'):
                object_start = time.perf_counter()
                t_wait = time.perf_counter()
                sha256, images = load_queue.get()
                timing_totals['wait_for_load'] += time.perf_counter() - t_wait

                patchtokens_list = []
                for i in range(0, len(images), opt.batch_size):
                    batch_images = torch.stack(images[i:i + opt.batch_size]).cuda()
                    bs = batch_images.shape[0]

                    t_model = time.perf_counter()
                    features = dinov2_model(batch_images, is_training=True)
                    timing_totals['model_forward'] += time.perf_counter() - t_model

                    patch_tokens = features['x_prenorm'][:, dinov2_model.num_register_tokens + 1:]
                    patchtokens = patch_tokens.reshape(bs, n_patch, n_patch, patch_tokens.shape[-1]).permute(0, 3, 1, 2).contiguous()
                    patchtokens_list.append(patchtokens)

                patchtokens = torch.cat(patchtokens_list, dim=0)

                t_save = time.perf_counter()
                np.savez_compressed(
                    os.path.join(save_dir, f'{sha256}.npz'),
                    patchtokens=patchtokens.cpu().numpy().astype(np.float16),
                )
                timing_totals['save'] += time.perf_counter() - t_save

                records.append({'sha256': sha256, f'view_feature_{feature_name}': True})
                processed += 1
                timing_totals['object_total'] += time.perf_counter() - object_start

                if opt.timing_log_interval > 0 and (
                    processed % opt.timing_log_interval == 0 or processed == len(sha256s)
                ):
                    print(
                        '[TIMING][VIEW] '
                        f'processed={processed}/{len(sha256s)} '
                        f'avg_object_sec={timing_totals["object_total"] / processed:.3f} '
                        f'avg_wait_sec={timing_totals["wait_for_load"] / processed:.3f} '
                        f'avg_model_sec={timing_totals["model_forward"] / processed:.3f} '
                        f'avg_save_sec={timing_totals["save"] / processed:.3f}'
                    )
    except Exception as e:
        print(f'Error happened during processing: {e}')

    elapsed = time.perf_counter() - script_start
    print(f'[TIMING][VIEW][SUMMARY] processed={processed} total_sec={elapsed:.3f}')

    records = pd.DataFrame.from_records(records)
    records.to_csv(os.path.join(opt.output_dir, f'view_feature_{feature_name}_{opt.rank}.csv'), index=False)


if __name__ == '__main__':
    main()
