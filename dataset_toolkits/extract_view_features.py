import argparse
import copy
import json
import os
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


def load_and_preprocess_frame(output_dir, sha256, frame):
    """Load one rendered RGBA frame and apply the same preprocessing as extract_feature.py.

    Steps:
      1) resize to 518x518 with LANCZOS
      2) convert to float32 in [0, 1]
      3) alpha composite on black: rgb * alpha
      4) convert to CHW torch tensor
    """
    image_path = os.path.join(output_dir, 'renders', sha256, frame['file_path'])
    image = Image.open(image_path)
    image = image.resize((518, 518), Image.Resampling.LANCZOS)

    image = np.array(image).astype(np.float32) / 255.0
    assert image.ndim == 3 and image.shape[2] >= 4, f'Expected RGBA image at {image_path}, got shape={image.shape}'

    # Match existing behavior in extract_feature.py
    image = image[:, :, :3] * image[:, :, 3:]
    image = torch.from_numpy(image).permute(2, 0, 1).float()
    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory containing metadata.csv/renders and where outputs are saved')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--model', type=str, default='dinov2_vitl14_reg',
                        help='Feature extraction model')
    parser.add_argument('--instances', type=str, default=None,
                        help='Path to text file containing sha256 IDs to process (one per line)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    opt = parser.parse_args()
    opt = edict(vars(opt))

    feature_name = opt.model
    save_dir = os.path.join(opt.output_dir, 'view_features', feature_name)
    os.makedirs(save_dir, exist_ok=True)

    # Load DINOv2 model (same behavior/default as extract_feature.py)
    dinov2_model = torch.hub.load('facebookresearch/dinov2', opt.model)
    dinov2_model.eval().cuda()

    # Match normalization from extract_feature.py
    normalize = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    n_patch = 518 // 14

    # get file list
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

    # Keep requiring objects to be voxelized/rendered like current script behavior
    metadata = metadata[metadata['voxelized'] == True]
    metadata = metadata[metadata['rendered'] == True]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []

    # filter out objects that are already processed
    sha256s = list(metadata['sha256'].values)
    for sha256 in copy.copy(sha256s):
        if os.path.exists(os.path.join(save_dir, f'{sha256}.npz')):
            records.append({'sha256': sha256, f'view_feature_{feature_name}': True})
            sha256s.remove(sha256)

    # extract per-view 2D patch tokens
    load_queue = Queue(maxsize=4)
    try:
        with ThreadPoolExecutor(max_workers=8) as loader_executor:
            def loader(sha256):
                try:
                    with open(os.path.join(opt.output_dir, 'renders', sha256, 'transforms.json'), 'r') as f:
                        render_metadata = json.load(f)
                    frames = render_metadata['frames']

                    # Parallelize frame loading within each object to match current behavior pattern
                    with ThreadPoolExecutor(max_workers=16) as frame_executor:
                        images = list(frame_executor.map(
                            lambda frame: load_and_preprocess_frame(opt.output_dir, sha256, frame),
                            frames,
                        ))
                    images = [normalize(image) for image in images]

                    load_queue.put((sha256, images))
                except Exception as e:
                    print(f"Error loading data for {sha256}: {e}")

            loader_executor.map(loader, sha256s)

            for _ in tqdm(range(len(sha256s)), desc='Extracting per-view features'):
                sha256, images = load_queue.get()
                n_views = len(images)
                patchtokens_lst = []

                for i in range(0, n_views, opt.batch_size):
                    batch_images = torch.stack(images[i:i + opt.batch_size]).cuda()
                    bs = batch_images.shape[0]

                    features = dinov2_model(batch_images, is_training=True)

                    # x_prenorm shape: [B, 1 + num_register_tokens + (H_patch*W_patch), C]
                    # Keep only spatial patch tokens (drop cls/register tokens).
                    patch_tokens = features['x_prenorm'][:, dinov2_model.num_register_tokens + 1:]
                    assert patch_tokens.shape[1] == n_patch * n_patch, (
                        f'Unexpected number of patches: got {patch_tokens.shape[1]}, '
                        f'expected {n_patch * n_patch}'
                    )

                    # Convert [B, N_patch, C] -> [B, C, H_patch, W_patch]
                    channels = patch_tokens.shape[-1]
                    patchtokens = patch_tokens.reshape(bs, n_patch, n_patch, channels).permute(0, 3, 1, 2).contiguous()
                    patchtokens_lst.append(patchtokens)

                # Final tensor shape: [V, C, H_patch, W_patch]
                patchtokens = torch.cat(patchtokens_lst, dim=0)
                assert patchtokens.shape[0] == n_views

                np.savez_compressed(
                    os.path.join(save_dir, f'{sha256}.npz'),
                    patchtokens=patchtokens.cpu().numpy().astype(np.float16),
                )
                records.append({'sha256': sha256, f'view_feature_{feature_name}': True})
    except Exception as e:
        print(f'Error happened during processing: {e}')

    records = pd.DataFrame.from_records(records)
    records.to_csv(os.path.join(opt.output_dir, f'view_feature_{feature_name}_{opt.rank}.csv'), index=False)
