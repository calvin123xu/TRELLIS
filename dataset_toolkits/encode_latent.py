import os
os.environ["ATTN_BACKEND"] = "xformers"
os.environ["SPARSE_ATTN_BACKEND"] = "xformers"
os.environ["SPCONV_ALGO"] = "native"
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import copy
import json
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from easydict import EasyDict as edict
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import trellis.models as models
import trellis.modules.sparse as sp

torch.set_grad_enabled(False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--feat_model', type=str, default='dinov2_vitl14_reg',
                        help='Feature model')
    parser.add_argument('--feature_subdir', type=str, default=None,
                        help='Optional feature subdirectory under output_dir/features. Defaults to feat_model when not set')
    parser.add_argument('--enc_pretrained', type=str, default='microsoft/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16',
                        help='Pretrained encoder model')
    parser.add_argument('--model_root', type=str, default='results',
                        help='Root directory of models')
    parser.add_argument('--enc_model', type=str, default=None,
                        help='Encoder model. if specified, use this model instead of pretrained model')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Checkpoint to load')
    parser.add_argument('--latent_subdir', type=str, default=None,
                        help='Optional latent subdirectory under output_dir/latents. Defaults to computed latent name when not set')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    opt = parser.parse_args()
    opt = edict(vars(opt))

    if opt.enc_model is None:
        latent_name = f'{opt.feat_model}_{opt.enc_pretrained.split("/")[-1]}'
        encoder = models.from_pretrained(opt.enc_pretrained).eval().cuda()
    else:
        latent_name = f'{opt.feat_model}_{opt.enc_model}_{opt.ckpt}'
        cfg = edict(json.load(open(os.path.join(opt.model_root, opt.enc_model, 'config.json'), 'r')))
        encoder = getattr(models, cfg.models.encoder.name)(**cfg.models.encoder.args).cuda()
        ckpt_path = os.path.join(opt.model_root, opt.enc_model, 'ckpts', f'encoder_{opt.ckpt}.pt')
        encoder.load_state_dict(torch.load(ckpt_path), strict=False)
        encoder.eval()
        print(f'Loaded model from {ckpt_path}')

    # Keep default behavior unless user overrides directory tags.
    feature_tag = opt.feature_subdir if opt.feature_subdir is not None else opt.feat_model
    latent_tag = opt.latent_subdir if opt.latent_subdir is not None else latent_name

    feature_col = f'feature_{feature_tag}'
    latent_col = f'latent_{latent_tag}'
    feature_dir = os.path.join(opt.output_dir, 'features', feature_tag)
    latent_dir = os.path.join(opt.output_dir, 'latents', latent_tag)

    os.makedirs(latent_dir, exist_ok=True)

    # get file list
    if os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    else:
        raise ValueError('metadata.csv not found')
    if opt.instances is not None:
        with open(opt.instances, 'r') as f:
            sha256s = [line.strip() for line in f]
        metadata = metadata[metadata['sha256'].isin(sha256s)]
    else:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        if feature_col in metadata.columns:
            metadata = metadata[metadata[feature_col] == True]
        else:
            # Graceful fallback for custom feature tags without metadata columns.
            metadata = metadata[metadata['sha256'].map(lambda x: os.path.exists(os.path.join(feature_dir, f'{x}.npz')))]
        if latent_col in metadata.columns:
            metadata = metadata[metadata[latent_col] == False]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []
    
    # filter out objects that are already processed
    sha256s = list(metadata['sha256'].values)
    for sha256 in copy.copy(sha256s):
        if os.path.exists(os.path.join(latent_dir, f'{sha256}.npz')):
            records.append({'sha256': sha256, latent_col: True})
            sha256s.remove(sha256)

    # encode latents
    load_queue = Queue(maxsize=4)
    try:
        with ThreadPoolExecutor(max_workers=32) as loader_executor, \
            ThreadPoolExecutor(max_workers=32) as saver_executor:
            def loader(sha256):
                try:
                    feats = np.load(os.path.join(feature_dir, f'{sha256}.npz'))
                    load_queue.put((sha256, feats))
                except Exception as e:
                    print(f"Error loading features for {sha256}: {e}")
            loader_executor.map(loader, sha256s)
            
            def saver(sha256, pack):
                save_path = os.path.join(latent_dir, f'{sha256}.npz')
                np.savez_compressed(save_path, **pack)
                records.append({'sha256': sha256, latent_col: True})
                
            for _ in tqdm(range(len(sha256s)), desc="Extracting latents"):
                sha256, feats = load_queue.get()
                feats = sp.SparseTensor(
                    feats = torch.from_numpy(feats['patchtokens']).float(),
                    coords = torch.cat([
                        torch.zeros(feats['patchtokens'].shape[0], 1).int(),
                        torch.from_numpy(feats['indices']).int(),
                    ], dim=1),
                ).cuda()
                latent = encoder(feats, sample_posterior=False)
                assert torch.isfinite(latent.feats).all(), "Non-finite latent"
                pack = {
                    'feats': latent.feats.cpu().numpy().astype(np.float32),
                    'coords': latent.coords[:, 1:].cpu().numpy().astype(np.uint8),
                }
                saver_executor.submit(saver, sha256, pack)
                
            saver_executor.shutdown(wait=True)
    except:
        print("Error happened during processing.")
        
    records = pd.DataFrame.from_records(records)
    records.to_csv(os.path.join(opt.output_dir, f'latent_{latent_tag}_{opt.rank}.csv'), index=False)
