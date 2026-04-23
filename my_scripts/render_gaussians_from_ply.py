import argparse
import sys
from pathlib import Path

# Ensure local repo root is importable when running:
#   python my_scripts/render_gaussians_from_ply.py ...
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trellis.representations import Gaussian
from trellis.utils import render_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render TRELLIS Gaussian PLY files to videos and/or snapshot images."
    )
    parser.add_argument(
        "--gaussian_dir",
        type=str,
        default="/nfs/speed-scratch/qiaoyu/speed-hpc/project/TRELLIS/datasets/Toys4k_small/gaussians_decoded_vis_weighted_abo_ft_step330k/",
        help="Directory containing Gaussian .ply files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. Default: <gaussian_dir>/renders",
    )
    parser.add_argument("--num_frames", type=int, default=120, help="Frames per output video.")
    parser.add_argument("--fps", type=int, default=15, help="FPS for output videos.")
    parser.add_argument("--resolution", type=int, default=512, help="Render resolution.")
    parser.add_argument(
        "--max_files",
        type=int,
        default=0,
        help="Maximum number of .ply files to process (0 means all).",
    )
    parser.add_argument(
        "--render_video",
        action="store_true",
        help="If set, render MP4 videos.",
    )
    parser.add_argument(
        "--render_snapshot",
        action="store_true",
        help="If set, render 4-view snapshot PNG image.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="If set, skip rendering when requested output files already exist.",
    )

    parser.add_argument(
        "--aabb",
        type=float,
        nargs=6,
        default=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0],
        help="Gaussian init aabb, format: minx miny minz sizex sizey sizez.",
    )
    parser.add_argument("--sh_degree", type=int, default=0)
    parser.add_argument("--mininum_kernel_size", type=float, default=0.0)
    parser.add_argument("--scaling_bias", type=float, default=0.01)
    parser.add_argument("--opacity_bias", type=float, default=0.1)
    parser.add_argument("--scaling_activation", type=str, default="exp", choices=["exp", "softplus"])

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    gaussian_dir = Path(args.gaussian_dir)
    if not gaussian_dir.exists():
        raise FileNotFoundError(f"gaussian_dir does not exist: {gaussian_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else gaussian_dir / "renders"
    output_dir.mkdir(parents=True, exist_ok=True)

    render_video_flag = args.render_video or (not args.render_video and not args.render_snapshot)
    render_snapshot_flag = args.render_snapshot

    ply_files = sorted(gaussian_dir.glob("*.ply"))
    if args.max_files > 0:
        ply_files = ply_files[: args.max_files]

    if len(ply_files) == 0:
        raise RuntimeError(f"No .ply files found in: {gaussian_dir}")

    print(f"Found {len(ply_files)} ply files.")
    print(f"Output dir: {output_dir}")

    for ply_path in ply_files:
        stem = ply_path.stem
        video_path = output_dir / f"{stem}.mp4"
        snapshot_path = output_dir / f"{stem}_snapshot.png"

        if args.skip_existing:
            need_video = render_video_flag and (not video_path.exists())
            need_snapshot = render_snapshot_flag and (not snapshot_path.exists())
            if not need_video and not need_snapshot:
                print(f"[Skip existing] {ply_path}")
                continue
        else:
            need_video = render_video_flag
            need_snapshot = render_snapshot_flag

        print(f"[Render] {ply_path}")

        gs = Gaussian(
            aabb=args.aabb,
            sh_degree=args.sh_degree,
            mininum_kernel_size=args.mininum_kernel_size,
            scaling_bias=args.scaling_bias,
            opacity_bias=args.opacity_bias,
            scaling_activation=args.scaling_activation,
            device="cuda",
        )
        gs.load_ply(str(ply_path))

        if need_video:
            import imageio

            video = render_utils.render_video(
                gs,
                resolution=args.resolution,
                num_frames=args.num_frames,
            )["color"]
            imageio.mimsave(video_path, video, fps=args.fps)

        if need_snapshot:
            import numpy as np
            from PIL import Image

            snaps = render_utils.render_snapshot(gs, resolution=args.resolution)["color"]
            strip = Image.fromarray(np.concatenate(snaps, axis=1))
            strip.save(snapshot_path)

    print("Done.")


if __name__ == "__main__":
    main()
