import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

os.environ["ATTN_BACKEND"] = "xformers"
os.environ["SPARSE_ATTN_BACKEND"] = "xformers"
os.environ["SPCONV_ALGO"] = "native"
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import trellis.models as models
import trellis.modules.sparse as sp


def _strip_module_prefix(state_dict):
    if not state_dict:
        return state_dict
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def _load_decoder(model_root: str, model_name: str, ckpt: str) -> torch.nn.Module:
    model_dir = os.path.join(model_root, model_name)
    cfg_path = os.path.join(model_dir, "config.json")
    ckpt_path = os.path.join(model_dir, "ckpts", f"decoder_{ckpt}.pt")

    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Missing config: {cfg_path}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Missing decoder checkpoint: {ckpt_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    dec_cfg = cfg["models"]["decoder"]
    dec_name = dec_cfg["name"]
    dec_args = dec_cfg["args"]

    decoder = getattr(models, dec_name)(**dec_args).cuda()
    raw = torch.load(ckpt_path, map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        raw = raw["state_dict"]
    if not isinstance(raw, dict):
        raise TypeError(f"Unexpected checkpoint payload type at {ckpt_path}: {type(raw)}")

    state_dict = _strip_module_prefix(raw)
    missing, unexpected = decoder.load_state_dict(state_dict, strict=False)
    print(f"[LOAD] decoder={ckpt_path}")
    print(f"[LOAD] missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
    return decoder.eval()


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Dataset root, e.g. datasets/Toys4k")
    parser.add_argument("--latent_model", type=str, required=True,
                        help="Latent model folder name under <output_dir>/latents")
    parser.add_argument("--model_root", type=str, required=True,
                        help="Model root that contains the co-finetune run folder")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Co-finetune run folder name")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Checkpoint suffix after decoder_, e.g. ema0.9999_step0120000")
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
    if not sha_list:
        raise ValueError(f"No latent npz found in {latent_dir}")

    save_dir = args.save_dir or os.path.join(args.output_dir, "gaussians_decoded")
    os.makedirs(save_dir, exist_ok=True)

    decoder = _load_decoder(args.model_root, args.model_name, args.ckpt)

    for sha in tqdm(sha_list, desc="Decoding latents with co-finetuned decoder"):
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
