import os
import sys

os.environ["ATTN_BACKEND"] = "xformers"
os.environ["SPARSE_ATTN_BACKEND"] = "xformers"
os.environ["SPCONV_ALGO"] = "native"
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
import torch
from tqdm import tqdm

import trellis.models as models
import trellis.modules.sparse as sp


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Dataset root, for example datasets/ABO_full")
    parser.add_argument("--latent_model", type=str, required=True,
                        help="Latent model folder name under <output_dir>/latents")
    parser.add_argument("--dec_pretrained", type=str,
                        default="microsoft/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16",
                        help="Pretrained gaussian decoder")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Where to save ply files. Default: <output_dir>/gaussians_decoded")
    parser.add_argument("--max_items", type=int, default=-1,
                        help="Decode at most N items; -1 means all")
    args = parser.parse_args()

    latent_dir = os.path.join(args.output_dir, "latents", args.latent_model)
    if not os.path.isdir(latent_dir):
        raise FileNotFoundError(f"latent dir not found: {latent_dir}")

    sha_list = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(latent_dir)
        if f.endswith('.npz')
    )
    if args.max_items > 0:
        sha_list = sha_list[:args.max_items]
    if len(sha_list) == 0:
        raise ValueError(f"No latent npz found in {latent_dir}")

    save_dir = args.save_dir or os.path.join(args.output_dir, "gaussians_decoded")
    os.makedirs(save_dir, exist_ok=True)

    decoder = models.from_pretrained(args.dec_pretrained).eval().cuda()

    for sha in tqdm(sha_list, desc="Decoding latents to gaussian"):
        npz_path = os.path.join(latent_dir, f"{sha}.npz")
        data = np.load(npz_path)

        feats = torch.from_numpy(data["feats"]).float().cuda()
        coords_xyz = torch.from_numpy(data["coords"]).int().cuda()
        batch_idx = torch.zeros((coords_xyz.shape[0], 1), dtype=torch.int32, device=coords_xyz.device)
        coords = torch.cat([batch_idx, coords_xyz], dim=1)

        slat = sp.SparseTensor(feats=feats, coords=coords)
        gaussian = decoder(slat)[0]
        gaussian.save_ply(os.path.join(save_dir, f"{sha}.ply"))

    print(f"Done. Saved {len(sha_list)} ply files to: {save_dir}")


if __name__ == "__main__":
    main()
