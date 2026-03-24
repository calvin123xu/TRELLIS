import os
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import trellis.models as models
import trellis.modules.sparse as sp


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True,
                        help="dataset root, e.g. datasets/Toys4k_small")
    parser.add_argument("--latent_model", type=str, default=None,
                        help="latent model folder name under output_dir/latents. "
                             "If not set, auto-detect from metadata.")
    parser.add_argument("--dec_pretrained", type=str,
                        default="microsoft/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16",
                        help="pretrained gaussian decoder")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="where to save ply files. default: <output_dir>/gaussians_decoded")
    parser.add_argument("--max_items", type=int, default=-1,
                        help="decode at most N items; -1 means all")
    args = parser.parse_args()

    metadata_path = os.path.join(args.output_dir, "metadata.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.csv not found: {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    latent_root = os.path.join(args.output_dir, "latents")
    if not os.path.isdir(latent_root):
        raise FileNotFoundError(f"latents dir not found: {latent_root}")

    # 1) decide latent model name
    latent_model = args.latent_model
    if latent_model is None:
        latent_cols = [c for c in metadata.columns if c.startswith("latent_")]
        latent_cols = [c for c in latent_cols if metadata[c].fillna(False).any()]
        if len(latent_cols) == 0:
            raise ValueError("No latent_* column with True values in metadata.")
        # pick first valid column that also has a matching folder
        picked = None
        for c in latent_cols:
            name = c[len("latent_"):]
            if os.path.isdir(os.path.join(latent_root, name)):
                picked = name
                break
        if picked is None:
            raise ValueError(f"Found latent columns {latent_cols}, but no matching folder under {latent_root}")
        latent_model = picked

    latent_col = f"latent_{latent_model}"
    if latent_col not in metadata.columns:
        raise ValueError(f"{latent_col} not found in metadata columns.")

    # 2) collect instances from metadata
    valid = metadata[metadata[latent_col] == True]  # noqa: E712
    if "sha256" not in valid.columns:
        raise ValueError("metadata.csv must contain 'sha256' column")

    sha_list = valid["sha256"].astype(str).tolist()

    # keep only existing files
    latent_dir = os.path.join(latent_root, latent_model)
    sha_list = [s for s in sha_list if os.path.exists(os.path.join(latent_dir, f"{s}.npz"))]

    if args.max_items > 0:
        sha_list = sha_list[:args.max_items]

    if len(sha_list) == 0:
        raise ValueError("No valid latent npz found to decode.")

    save_dir = args.save_dir or os.path.join(args.output_dir, "gaussians_decoded")
    os.makedirs(save_dir, exist_ok=True)

    # 3) load Gaussian decoder
    decoder = models.from_pretrained(args.dec_pretrained).eval().cuda()

    # 4) decode each latent -> gaussian -> ply
    for sha in tqdm(sha_list, desc="Decoding latents to gaussian"):
        npz_path = os.path.join(latent_dir, f"{sha}.npz")
        data = np.load(npz_path)

        feats = torch.from_numpy(data["feats"]).float().cuda()          # [N, C]
        coords_xyz = torch.from_numpy(data["coords"]).int().cuda()      # [N, 3]
        batch_idx = torch.zeros((coords_xyz.shape[0], 1), dtype=torch.int32, device=coords_xyz.device)
        coords = torch.cat([batch_idx, coords_xyz], dim=1)              # [N, 4], batch-first

        slat = sp.SparseTensor(feats=feats, coords=coords)
        gaussian = decoder(slat)[0]  # decoder returns List[Gaussian]

        out_ply = os.path.join(save_dir, f"{sha}.ply")
        gaussian.save_ply(out_ply)

    print(f"Done. Saved {len(sha_list)} ply files to: {save_dir}")


if __name__ == "__main__":
    main()
