import os
import argparse
import math
import pandas as pd


def chunk_list(items, chunk_size):
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def main():
    parser = argparse.ArgumentParser(
        description="Generate txt chunks (e.g. 64 IDs each) for assets that are not rendered yet."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Dataset root, e.g. datasets/Toys4k_small",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=64,
        help="Number of assets per txt file",
    )
    parser.add_argument(
        "--txt_dir",
        type=str,
        default=None,
        help="Where to save generated txt files (default: <output_dir>/render_chunks)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="unrendered_",
        help="Prefix for txt names, e.g. unrendered_0000.txt",
    )
    parser.add_argument(
        "--require_local_path",
        action="store_true",
        help="Only keep rows with non-empty local_path (recommended for render.py).",
    )
    args = parser.parse_args()

    metadata_path = os.path.join(args.output_dir, "metadata.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.csv not found: {metadata_path}")

    df = pd.read_csv(metadata_path)
    if "sha256" not in df.columns:
        raise ValueError('metadata.csv must contain "sha256" column')

    # render.py default path requires local_path when --instances is not used.
    # For your use case with --instances txt, filtering local_path still helps avoid invalid jobs.
    if args.require_local_path:
        if "local_path" not in df.columns:
            raise ValueError('metadata.csv does not contain "local_path" column')
        df = df[df["local_path"].notna()]

    # If rendered column is absent, treat all as unrendered.
    if "rendered" in df.columns:
        df = df[df["rendered"] == False]  # noqa: E712

    shas = df["sha256"].astype(str).tolist()
    shas = sorted(set(shas))

    txt_dir = args.txt_dir or os.path.join(args.output_dir, "render_chunks")
    os.makedirs(txt_dir, exist_ok=True)

    # remove old chunk files with same prefix to avoid mixing runs
    for name in os.listdir(txt_dir):
        if name.startswith(args.prefix) and name.endswith(".txt"):
            os.remove(os.path.join(txt_dir, name))

    if len(shas) == 0:
        print("No unrendered assets found. No txt files generated.")
        return

    n_chunks = math.ceil(len(shas) / args.chunk_size)
    for idx, sub in enumerate(chunk_list(shas, args.chunk_size)):
        fname = f"{args.prefix}{idx:04d}.txt"
        fpath = os.path.join(txt_dir, fname)
        with open(fpath, "w") as f:
            for sha in sub:
                f.write(f"{sha}\n")

    print(f"metadata: {metadata_path}")
    print(f"unrendered assets: {len(shas)}")
    print(f"chunk size: {args.chunk_size}")
    print(f"chunks generated: {n_chunks}")
    print(f"txt dir: {txt_dir}")
    print(f"example: {os.path.join(txt_dir, f'{args.prefix}0000.txt')}")


if __name__ == "__main__":
    main()
