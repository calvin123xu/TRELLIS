import os
os.environ["ATTN_BACKEND"] = "xformers"
os.environ["SPARSE_ATTN_BACKEND"] = "xformers"
os.environ["SPCONV_ALGO"] = "native"
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import json
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# IMPORTANT: set before trellis import
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

import trellis.utils.render_utils as render_utils
from trellis.representations import Gaussian


def load_gt_rgba_to_rgb(path, bg=(0, 0, 0), size=512):
    """Load GT png and composite RGBA over background, return float32 [H,W,3] in [0,1]."""
    img = Image.open(path).convert("RGBA").resize((size, size), Image.Resampling.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 255.0
    rgb = arr[..., :3]
    a = arr[..., 3:4]
    bg = np.array(bg, dtype=np.float32).reshape(1, 1, 3) / 255.0
    out = rgb * a + bg * (1.0 - a)
    return out


def camera_from_frame(frame, device="cuda"):
    """
    match dataset_toolkits/extract_feature.py:
      c2w[:3,1:3]*=-1, extrinsics=inverse(c2w), intrinsics from camera_angle_x
    """
    c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32, device=device)
    c2w[:3, 1:3] *= -1
    extr = torch.inverse(c2w)
    fov = torch.tensor(frame["camera_angle_x"], dtype=torch.float32, device=device)
    intr = __import__("utils3d").torch.intrinsics_from_fov_xy(fov, fov)
    return extr, intr


@torch.no_grad()
def render_pred_views(gaussian_ply, transforms_json, resolution=512, bg=(0,0,0)):
    g = Gaussian(aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0], sh_degree=0, device="cuda")
    g.load_ply(gaussian_ply)

    with open(transforms_json, "r") as f:
        t = json.load(f)
    frames = t["frames"]

    renderer = render_utils.get_renderer(
        g,
        resolution=resolution,
        bg_color=bg,
        near=0.8,
        far=1.6,
        ssaa=1
    )

    pred = []
    for fr in frames:
        extr, intr = camera_from_frame(fr, device="cuda")
        res = renderer.render(g, extr, intr)
        rgb = res["color"].detach().cpu().numpy().transpose(1, 2, 0)  # [H,W,3], [0,1]
        pred.append(np.clip(rgb, 0.0, 1.0))
    return pred, frames


def to_lpips_tensor(x):  # x: [H,W,3], [0,1]
    t = torch.from_numpy(x).permute(2,0,1).unsqueeze(0).float().cuda()
    return t * 2.0 - 1.0


def eval_one_object(sha, output_dir, gauss_dir, resolution=512, bg=(0,0,0)):
    gt_dir = os.path.join(output_dir, "renders", sha)
    transforms_json = os.path.join(gt_dir, "transforms.json")
    ply_path = os.path.join(gauss_dir, f"{sha}.ply")

    if not os.path.exists(transforms_json) or not os.path.exists(ply_path):
        return None

    pred_views, frames = render_pred_views(
        ply_path, transforms_json, resolution=resolution, bg=bg
    )

    lpips_fn = lpips.LPIPS(net="alex").cuda().eval()

    psnr_list, ssim_list, lpips_list = [], [], []

    for i, fr in enumerate(frames):
        # file_path usually like "000.png"
        gt_path = os.path.join(gt_dir, fr["file_path"])
        if not os.path.exists(gt_path):
            continue

        gt = load_gt_rgba_to_rgb(gt_path, bg=bg, size=resolution)  # [H,W,3], [0,1]
        pr = pred_views[i]

        # metrics
        psnr = peak_signal_noise_ratio(gt, pr, data_range=1.0)
        ssim = structural_similarity(gt, pr, data_range=1.0, channel_axis=2)
        lp = lpips_fn(to_lpips_tensor(gt), to_lpips_tensor(pr)).item()

        psnr_list.append(psnr)
        ssim_list.append(ssim)
        lpips_list.append(lp)

    if len(psnr_list) == 0:
        return None

    return {
        "sha256": sha,
        "num_views": len(psnr_list),
        "psnr": float(np.mean(psnr_list)),
        "ssim": float(np.mean(ssim_list)),
        "lpips": float(np.mean(lpips_list)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True,
                        help="e.g. datasets/Toys4k_small")
    parser.add_argument("--gauss_dir", type=str, default=None,
                        help="decoded ply dir, default: <output_dir>/gaussians_decoded")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--bg", type=str, default="0,0,0",
                        help="background color in 0-255, e.g. 0,0,0")
    parser.add_argument("--max_items", type=int, default=-1)
    parser.add_argument("--out_csv", type=str, default=None)
    args = parser.parse_args()

    bg = tuple(int(x) for x in args.bg.split(","))
    gauss_dir = args.gauss_dir or os.path.join(args.output_dir, "gaussians_decoded")
    out_csv = args.out_csv or os.path.join(args.output_dir, "render_metrics.csv")

    meta = pd.read_csv(os.path.join(args.output_dir, "metadata.csv"))
    shas = meta["sha256"].astype(str).tolist()

    if args.max_items > 0:
        shas = shas[:args.max_items]

    rows = []
    for sha in tqdm(shas, desc="Evaluating"):
        r = eval_one_object(sha, args.output_dir, gauss_dir,
                            resolution=args.resolution, bg=bg)
        if r is not None:
            rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    if len(df) > 0:
        print("=== Aggregate ===")
        print(f"N objects: {len(df)}")
        print(f"PSNR : {df['psnr'].mean():.4f}")
        print(f"SSIM : {df['ssim'].mean():.4f}")
        print(f"LPIPS: {df['lpips'].mean():.4f}")
    else:
        print("No valid objects evaluated.")


if __name__ == "__main__":
    main()
