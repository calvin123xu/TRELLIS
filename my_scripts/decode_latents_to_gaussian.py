import os, sys
os.environ["ATTN_BACKEND"] = "xformers"
os.environ["SPARSE_ATTN_BACKEND"] = "xformers"
os.environ["SPCONV_ALGO"] = "native"
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from easydict import EasyDict as edict

import trellis.models as models
import trellis.modules.sparse as sp


def _load_config(config_path: str):
    with open(config_path, "r") as f:
        return edict(json.load(f))


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        candidate_keys = ["state_dict", "model", "model_state_dict", "module", "ema"]
        for key in candidate_keys:
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
        if all(isinstance(k, str) for k in ckpt_obj.keys()):
            return ckpt_obj
    raise ValueError("Unsupported checkpoint format: expected a state_dict or a dict containing state_dict/model/model_state_dict/module/ema.")


def _strip_module_prefix(state_dict):
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}



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
    parser.add_argument("--dec_ckpt_path", type=str, default=None,
                        help="explicit local decoder checkpoint .pt path")
    parser.add_argument("--dec_config_path", type=str, default=None,
                        help="explicit config.json path for local decoder checkpoint loading")
    parser.add_argument("--dec_model_dir", type=str, default=None,
                        help="optional trained model dir containing config.json (and usually ckpts/)")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="decoded output directory. If relative, it is resolved under "
                             "<output_dir>. default: <output_dir>/gaussians_decoded")
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

    latent_dir = os.path.join(latent_root, latent_model)
    if not os.path.isdir(latent_dir):
        raise FileNotFoundError(f"latent subdir not found: {latent_dir}")

    latent_col = f"latent_{latent_model}"
    # 2) collect instances; prefer metadata flag if present, with graceful fallbacks
    if latent_col in metadata.columns:
        valid = metadata[metadata[latent_col] == True]  # noqa: E712
        if "sha256" not in valid.columns:
            raise ValueError("metadata.csv must contain 'sha256' column")
        sha_list = valid["sha256"].astype(str).tolist()
    elif "sha256" in metadata.columns:
        # Custom latent folders may not have a dedicated latent_<name> bookkeeping column.
        # In that case, decode any existing latent files for metadata rows.
        sha_list = metadata["sha256"].astype(str).tolist()
    else:
        # Final fallback: decode directly from files when metadata bookkeeping is unavailable.
        sha_list = [os.path.splitext(f)[0] for f in os.listdir(latent_dir) if f.endswith(".npz")]

    # keep only existing files in the chosen latent subdirectory
    sha_list = [s for s in sha_list if os.path.exists(os.path.join(latent_dir, f"{s}.npz"))]

    if args.max_items > 0:
        sha_list = sha_list[:args.max_items]

    if len(sha_list) == 0:
        raise ValueError("No valid latent npz found to decode.")

    if args.save_dir is None:
        save_dir = os.path.join(args.output_dir, "gaussians_decoded")
    elif os.path.isabs(args.save_dir):
        save_dir = args.save_dir
    else:
        save_dir = os.path.join(args.output_dir, args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # 3) load Gaussian decoder
    if args.dec_ckpt_path is not None:
        if args.dec_config_path is not None:
            config_path = args.dec_config_path
        elif args.dec_model_dir is not None:
            config_path = os.path.join(args.dec_model_dir, "config.json")
        else:
            raise ValueError("When --dec_ckpt_path is provided, you must also provide --dec_config_path or --dec_model_dir.")

        cfg = _load_config(config_path)
        decoder = getattr(models, cfg.models.decoder.name)(**cfg.models.decoder.args).cuda()
        ckpt_obj = torch.load(args.dec_ckpt_path, map_location="cpu")
        state_dict = _strip_module_prefix(_extract_state_dict(ckpt_obj))
        incompatible = decoder.load_state_dict(state_dict, strict=False)
        if len(incompatible.missing_keys) > 0:
            print(f"[decoder] missing_keys: {incompatible.missing_keys}")
        if len(incompatible.unexpected_keys) > 0:
            print(f"[decoder] unexpected_keys: {incompatible.unexpected_keys}")
        decoder.eval()
        print(f"Loaded decoder checkpoint from {args.dec_ckpt_path}")
    else:
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
