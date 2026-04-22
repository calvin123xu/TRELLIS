#!/usr/bin/env python3
import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["1", "true", "t", "yes", "y"])


def summarize(values):
    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "p05": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def load_patchtokens(path):
    data = np.load(path)
    if "patchtokens" not in data:
        raise KeyError(f"patchtokens missing in {path}")
    arr = data["patchtokens"].astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"unexpected patchtokens shape {arr.shape} in {path}")
    return arr


def analyze_dataset(base_dir, dataset_name, max_samples, seed, zero_thresh, max_rows_for_similarity):
    dataset_dir = os.path.join(base_dir, dataset_name)
    feature_dirs = {
        "mean": os.path.join(dataset_dir, "features", "dinov2_vitl14_reg"),
        "visible_only": os.path.join(dataset_dir, "features", "dinov2_vitl14_reg_visible_only"),
        "visible_zbuf": os.path.join(dataset_dir, "features", "dinov2_vitl14_reg_visible_zbuf"),
    }

    for k, d in feature_dirs.items():
        if not os.path.isdir(d):
            raise FileNotFoundError(f"missing feature dir for {k}: {d}")

    metadata_path = os.path.join(dataset_dir, "metadata.csv")
    metadata = pd.read_csv(metadata_path)
    expected_mask = pd.Series([True] * len(metadata))
    if "rendered" in metadata.columns:
        expected_mask &= as_bool(metadata["rendered"])
    if "voxelized" in metadata.columns:
        expected_mask &= as_bool(metadata["voxelized"])
    expected_count = int(expected_mask.sum())

    ids = {}
    for k, d in feature_dirs.items():
        ids[k] = {f[:-4] for f in os.listdir(d) if f.endswith(".npz")}

    common_ids = sorted(ids["mean"] & ids["visible_only"] & ids["visible_zbuf"])
    sample_n = min(max_samples, len(common_ids))
    rnd = random.Random(seed)
    sample_ids = rnd.sample(common_ids, sample_n) if sample_n < len(common_ids) else common_ids

    stats = {
        "mean_zero_frac": [],
        "vis_zero_frac": [],
        "zbuf_zero_frac": [],
        "mean_row_norm": [],
        "vis_row_norm": [],
        "zbuf_row_norm": [],
        "cos_vis_zbuf": [],
        "rel_l2_vis_zbuf": [],
        "cos_mean_zbuf": [],
        "rel_l2_mean_zbuf": [],
        "cos_mean_vis": [],
        "rel_l2_mean_vis": [],
    }

    shape_mismatch = 0
    read_failures = 0

    for sid in sample_ids:
        try:
            arr_mean = load_patchtokens(os.path.join(feature_dirs["mean"], sid + ".npz"))
            arr_vis = load_patchtokens(os.path.join(feature_dirs["visible_only"], sid + ".npz"))
            arr_zbuf = load_patchtokens(os.path.join(feature_dirs["visible_zbuf"], sid + ".npz"))
        except Exception:
            read_failures += 1
            continue

        if not (arr_mean.shape == arr_vis.shape == arr_zbuf.shape):
            shape_mismatch += 1
            continue

        norm_mean = np.linalg.norm(arr_mean, axis=1)
        norm_vis = np.linalg.norm(arr_vis, axis=1)
        norm_zbuf = np.linalg.norm(arr_zbuf, axis=1)

        stats["mean_zero_frac"].append(float(np.mean(norm_mean <= zero_thresh)))
        stats["vis_zero_frac"].append(float(np.mean(norm_vis <= zero_thresh)))
        stats["zbuf_zero_frac"].append(float(np.mean(norm_zbuf <= zero_thresh)))
        stats["mean_row_norm"].append(float(np.mean(norm_mean)))
        stats["vis_row_norm"].append(float(np.mean(norm_vis)))
        stats["zbuf_row_norm"].append(float(np.mean(norm_zbuf)))

        # Bound compute cost by sampling rows before flattening for pairwise similarity.
        if arr_mean.shape[0] > max_rows_for_similarity:
            sel = np.random.default_rng(seed + hash(sid) % 1000003).choice(
                arr_mean.shape[0],
                size=max_rows_for_similarity,
                replace=False,
            )
            arr_mean_sim = arr_mean[sel]
            arr_vis_sim = arr_vis[sel]
            arr_zbuf_sim = arr_zbuf[sel]
        else:
            arr_mean_sim = arr_mean
            arr_vis_sim = arr_vis
            arr_zbuf_sim = arr_zbuf

        flat_mean = arr_mean_sim.reshape(-1)
        flat_vis = arr_vis_sim.reshape(-1)
        flat_zbuf = arr_zbuf_sim.reshape(-1)

        def cosine(a, b):
            den = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
            return float(np.dot(a, b) / den)

        def rel_l2(a, b):
            den = np.linalg.norm(a) + 1e-12
            return float(np.linalg.norm(a - b) / den)

        stats["cos_vis_zbuf"].append(cosine(flat_vis, flat_zbuf))
        stats["rel_l2_vis_zbuf"].append(rel_l2(flat_vis, flat_zbuf))
        stats["cos_mean_zbuf"].append(cosine(flat_mean, flat_zbuf))
        stats["rel_l2_mean_zbuf"].append(rel_l2(flat_mean, flat_zbuf))
        stats["cos_mean_vis"].append(cosine(flat_mean, flat_vis))
        stats["rel_l2_mean_vis"].append(rel_l2(flat_mean, flat_vis))

    out = {
        "dataset": dataset_name,
        "expected_rendered_voxelized": expected_count,
        "feature_counts": {
            "mean": len(ids["mean"]),
            "visible_only": len(ids["visible_only"]),
            "visible_zbuf": len(ids["visible_zbuf"]),
            "common_all_three": len(common_ids),
        },
        "sampling": {
            "requested_max_samples": int(max_samples),
            "actual_sampled": int(sample_n),
            "read_failures": int(read_failures),
            "shape_mismatch": int(shape_mismatch),
            "effective_n": int(len(stats["cos_vis_zbuf"])),
            "seed": int(seed),
            "zero_threshold": float(zero_thresh),
            "max_rows_for_similarity": int(max_rows_for_similarity),
        },
        "feature_stats": {
            "mean_zero_frac": summarize(stats["mean_zero_frac"]),
            "visible_only_zero_frac": summarize(stats["vis_zero_frac"]),
            "visible_zbuf_zero_frac": summarize(stats["zbuf_zero_frac"]),
            "mean_row_norm": summarize(stats["mean_row_norm"]),
            "visible_only_row_norm": summarize(stats["vis_row_norm"]),
            "visible_zbuf_row_norm": summarize(stats["zbuf_row_norm"]),
        },
        "pairwise_similarity": {
            "cos_mean_visible_only": summarize(stats["cos_mean_vis"]),
            "rel_l2_mean_visible_only": summarize(stats["rel_l2_mean_vis"]),
            "cos_visible_only_visible_zbuf": summarize(stats["cos_vis_zbuf"]),
            "rel_l2_visible_only_visible_zbuf": summarize(stats["rel_l2_vis_zbuf"]),
            "cos_mean_visible_zbuf": summarize(stats["cos_mean_zbuf"]),
            "rel_l2_mean_visible_zbuf": summarize(stats["rel_l2_mean_zbuf"]),
        },
    }
    return out


def format_brief(report):
    fs = report["feature_stats"]
    sim = report["pairwise_similarity"]
    return [
        f"[{report['dataset']}] expected={report['expected_rendered_voxelized']} common={report['feature_counts']['common_all_three']} n={report['sampling']['effective_n']}",
        (
            f"[{report['dataset']}] zero_frac mean={fs['mean_zero_frac']['mean']:.6f} "
            f"vis={fs['visible_only_zero_frac']['mean']:.6f} "
            f"zbuf={fs['visible_zbuf_zero_frac']['mean']:.6f}"
        ),
        (
            f"[{report['dataset']}] row_norm mean={fs['mean_row_norm']['mean']:.6f} "
            f"vis={fs['visible_only_row_norm']['mean']:.6f} "
            f"zbuf={fs['visible_zbuf_row_norm']['mean']:.6f}"
        ),
        (
            f"[{report['dataset']}] cos(vis,zbuf)={sim['cos_visible_only_visible_zbuf']['mean']:.6f} "
            f"rel_l2(vis,zbuf)={sim['rel_l2_visible_only_visible_zbuf']['mean']:.6f}"
        ),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="/speed-scratch/an_zhan/project/TRELLIS-experiments/datasets")
    parser.add_argument("--datasets", type=str, default="ABO_full,Toys4k")
    parser.add_argument("--max_samples", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zero_thresh", type=float, default=1e-8)
    parser.add_argument("--max_rows_for_similarity", type=int, default=4096)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/speed-scratch/an_zhan/project/TRELLIS-experiments/results/analysis/zbuf_regression",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    reports = []
    for dataset in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        reports.append(
            analyze_dataset(
                base_dir=args.base_dir,
                dataset_name=dataset,
                max_samples=args.max_samples,
                seed=args.seed,
                zero_thresh=args.zero_thresh,
                max_rows_for_similarity=args.max_rows_for_similarity,
            )
        )

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_json = os.path.join(args.output_dir, f"zbuf_feature_diagnosis_{ts}.json")
    out_txt = os.path.join(args.output_dir, f"zbuf_feature_diagnosis_{ts}.txt")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"generated_utc": ts, "reports": reports}, f, indent=2)

    lines = ["ZBUF Regression Feature Diagnosis", f"generated_utc={ts}", ""]
    for rep in reports:
        lines.extend(format_brief(rep))
        lines.append("")
    lines.append(f"json_report={out_json}")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
