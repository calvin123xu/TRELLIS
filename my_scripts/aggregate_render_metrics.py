#!/usr/bin/env python3
"""Aggregate render metric CSVs produced by multiple ranks.

Example:
    python my_scripts/aggregate_render_metrics.py \
        --metrics_dir /nfs/speed-scratch/qiaoyu/speed-hpc/project/TRELLIS/datasets/Toys4k_small/render_metrics \
        --csv_prefix render_metrics_vis_weighted_abo_ft

By default this reads these four files:
    render_metrics_vis_weighted_abo_ft_rank0000_of_0004.csv
    render_metrics_vis_weighted_abo_ft_rank0001_of_0004.csv
    render_metrics_vis_weighted_abo_ft_rank0002_of_0004.csv
    render_metrics_vis_weighted_abo_ft_rank0003_of_0004.csv
and reports mean/std for psnr, ssim, and lpips if present.
"""

import argparse
import csv
import math
from pathlib import Path


DEFAULT_METRICS_DIR = Path(
    "/nfs/speed-scratch/qiaoyu/speed-hpc/project/TRELLIS/"
    "datasets/Toys4k_small/render_metrics"
)
DEFAULT_CSV_PREFIX = "render_metrics_vis_weighted_abo_ft"
DEFAULT_METRICS = ("psnr", "ssim", "lpips")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read rank CSVs and compute mean/std render metrics."
    )
    parser.add_argument(
        "--metrics_dir",
        type=Path,
        default=DEFAULT_METRICS_DIR,
        help=f"Directory containing rank CSVs. Default: {DEFAULT_METRICS_DIR}",
    )
    parser.add_argument(
        "--csv_prefix",
        type=str,
        default=DEFAULT_CSV_PREFIX,
        help=(
            "CSV filename prefix before _rankXXXX_of_YYYY. "
            f"Default: {DEFAULT_CSV_PREFIX}"
        ),
    )
    parser.add_argument(
        "--num_ranks",
        type=int,
        default=4,
        help="Number of rank CSVs to read. Default: 4",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="Metric columns to summarize. Default: psnr ssim lpips",
    )
    parser.add_argument(
        "--ddof",
        type=int,
        default=1,
        choices=(0, 1),
        help="Delta degrees of freedom for std: 1=sample std, 0=population std. Default: 1",
    )
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=None,
        help="Optional path to save the aggregate summary CSV.",
    )
    parser.add_argument(
        "--combined_csv",
        type=Path,
        default=None,
        help="Optional path to save all rows concatenated from every rank CSV.",
    )
    return parser.parse_args()


def rank_csv_path(metrics_dir, csv_prefix, rank, num_ranks):
    return metrics_dir / f"{csv_prefix}_rank{rank:04d}_of_{num_ranks:04d}.csv"


def read_rank_csvs(metrics_dir, csv_prefix, num_ranks):
    rows = []
    fieldnames = []
    missing_paths = []

    for rank in range(num_ranks):
        path = rank_csv_path(metrics_dir, csv_prefix, rank, num_ranks)
        if not path.exists():
            missing_paths.append(path)
            continue

        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {path}")

            for name in reader.fieldnames:
                if name not in fieldnames:
                    fieldnames.append(name)
            if "source_csv" not in fieldnames:
                fieldnames.append("source_csv")
            if "rank" not in fieldnames:
                fieldnames.insert(0, "rank")

            for row in reader:
                if not row.get("rank"):
                    row["rank"] = str(rank)
                row["source_csv"] = str(path)
                rows.append(row)

    if missing_paths:
        missing = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(f"Missing expected rank CSV(s):\n{missing}")

    if not rows:
        raise ValueError("No rows were read from the rank CSVs.")

    return rows, fieldnames


def finite_metric_values(rows, metric):
    values = []
    for row in rows:
        raw_value = row.get(metric, "")
        if raw_value == "":
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def mean(values):
    return sum(values) / len(values)


def std(values, ddof):
    if len(values) <= ddof:
        return float("nan")
    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - ddof)
    return math.sqrt(variance)


def summarize_metrics(rows, metric_names, ddof):
    available_metrics = [name for name in metric_names if name in rows[0]]
    missing_metrics = [name for name in metric_names if name not in rows[0]]

    if not available_metrics:
        raise ValueError(
            "None of the requested metric columns were found. "
            f"Requested: {metric_names}; CSV columns: {list(rows[0])}"
        )

    if missing_metrics:
        print(
            "[Warning] Skipping metric column(s) not found in CSVs: "
            + ", ".join(missing_metrics)
        )

    summary = []
    for metric in available_metrics:
        values = finite_metric_values(rows, metric)
        if not values:
            print(f"[Warning] Skipping metric with no finite values: {metric}")
            continue
        summary.append(
            {
                "metric": metric,
                "count": len(values),
                "mean": mean(values),
                "std": std(values, ddof=ddof),
                "min": min(values),
                "max": max(values),
            }
        )

    if not summary:
        raise ValueError("No finite metric values were found.")

    return summary


def print_summary(summary, total_rows, num_ranks):
    print("=== Aggregate render metrics ===")
    print(f"Rank CSVs read : {num_ranks}")
    print(f"Total rows     : {total_rows}")
    print("")
    for row in summary:
        print(
            f"{row['metric'].upper():<8} mean={row['mean']:.6f}  "
            f"std={row['std']:.6f}  count={row['count']}  "
            f"min={row['min']:.6f}  max={row['max']:.6f}"
        )


def write_combined_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path, summary):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ("metric", "count", "mean", "std", "min", "max")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def main():
    args = parse_args()

    rows, fieldnames = read_rank_csvs(args.metrics_dir, args.csv_prefix, args.num_ranks)
    summary = summarize_metrics(rows, args.metrics, args.ddof)
    print_summary(summary, len(rows), args.num_ranks)

    if args.combined_csv is not None:
        write_combined_csv(args.combined_csv, rows, fieldnames)
        print(f"Saved combined rows to: {args.combined_csv}")

    if args.out_csv is not None:
        write_summary_csv(args.out_csv, summary)
        print(f"Saved aggregate summary to: {args.out_csv}")


if __name__ == "__main__":
    main()
