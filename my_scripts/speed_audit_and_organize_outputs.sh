#!/encs/bin/bash
set -euo pipefail

BASE_DIR="/speed-scratch/an_zhan/project/TRELLIS-experiments"
MODEL="dinov2_vitl14_reg"
DATASETS=(ABO_full Toys4k)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-dir)
      BASE_DIR="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --datasets)
      shift
      DATASETS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        DATASETS+=("$1")
        shift
      done
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

count_npz() {
  local dir="$1"
  if [[ -d "$dir" ]]; then
    find "$dir" -type f -name '*.npz' | wc -l
  else
    echo 0
  fi
}

safe_snapshot_legacy_mean() {
  local data_dir="$1"
  local mean_dir="$2"
  local safe_root="$3"

  mkdir -p "$safe_root"
  if [[ ! -d "$mean_dir" ]]; then
    echo "[SNAPSHOT] skip: mean dir missing: $mean_dir"
    return
  fi

  local mean_count
  mean_count=$(count_npz "$mean_dir")
  if [[ "$mean_count" -eq 0 ]]; then
    echo "[SNAPSHOT] skip: mean dir has no npz files: $mean_dir"
    return
  fi

  local stamp
  stamp=$(date +%Y%m%d_%H%M%S)
  local target="$safe_root/${MODEL}_legacy_mean_${stamp}"

  if cp -al "$mean_dir" "$target" 2>/dev/null; then
    chmod -R a-w "$target" || true
    echo "[SNAPSHOT] hardlink snapshot created: $target"
  else
    echo "[SNAPSHOT] hardlink failed, creating full copy: $target"
    cp -a "$mean_dir" "$target"
    chmod -R a-w "$target" || true
  fi

  ln -sfn "$target" "$safe_root/${MODEL}_legacy_mean_latest"
}

collect_run_tags() {
  local data_dir="$1"
  local refs_dir="$2"
  local timing_dir="$3"

  declare -A tags=()

  if [[ -d "$refs_dir" ]]; then
    while IFS= read -r d; do
      tags["$(basename "$d")"]=1
    done < <(find "$refs_dir" -mindepth 1 -maxdepth 1 -type d)
  fi

  if [[ -d "$timing_dir" ]]; then
    while IFS= read -r d; do
      tags["$(basename "$d")"]=1
    done < <(find "$timing_dir" -mindepth 1 -maxdepth 1 -type d)
  fi

  shopt -s nullglob
  local f
  for f in "$data_dir"/render_metrics_*.csv; do
    local b tag
    b=$(basename "$f")
    tag="${b#render_metrics_}"
    tag="${tag%.csv}"
    tags["$tag"]=1
  done

  for f in "$data_dir"/gaussians_decoded_*; do
    local b tag
    b=$(basename "$f")
    tag="${b#gaussians_decoded_}"
    tags["$tag"]=1
  done
  shopt -u nullglob

  local out=()
  local k
  for k in "${!tags[@]}"; do
    out+=("$k")
  done

  if [[ ${#out[@]} -gt 0 ]]; then
    printf '%s\n' "${out[@]}" | sort
  fi
}

link_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -e "$src" ]]; then
    ln -sfn "$src" "$dst"
  fi
}

build_runs_index() {
  local data_dir="$1"
  local refs_dir="$2"
  local timing_dir="$3"
  local runs_dir="$4"

  mkdir -p "$runs_dir"
  local idx="$runs_dir/RUNS_INDEX.csv"
  echo "run_tag,refs_dir,timing_dir,metrics_csv,gaussian_dir,feature_mean_dir,feature_visible_only_dir" > "$idx"

  while IFS= read -r tag; do
    [[ -n "$tag" ]] || continue
    local run_dir="$runs_dir/$tag"
    mkdir -p "$run_dir"

    local ref_path="$refs_dir/$tag"
    local timing_path="$timing_dir/$tag"
    local metrics_path="$data_dir/render_metrics_${tag}.csv"
    local gauss_path="$data_dir/gaussians_decoded_${tag}"
    local mean_feature_path="$data_dir/features/${MODEL}"
    local vis_feature_path="$data_dir/features/${MODEL}_visible_only"

    link_if_exists "$ref_path" "$run_dir/refs"
    link_if_exists "$timing_path" "$run_dir/timing"
    link_if_exists "$metrics_path" "$run_dir/render_metrics.csv"
    link_if_exists "$gauss_path" "$run_dir/gaussians"
    link_if_exists "$mean_feature_path" "$run_dir/feature_mean"
    link_if_exists "$vis_feature_path" "$run_dir/feature_visible_only"

    {
      echo "dataset_dir=$data_dir"
      echo "run_tag=$tag"
      echo "refs_dir=$ref_path"
      echo "timing_dir=$timing_path"
      echo "metrics_csv=$metrics_path"
      echo "gaussian_dir=$gauss_path"
      echo "feature_mean_dir=$mean_feature_path"
      echo "feature_visible_only_dir=$vis_feature_path"
    } > "$run_dir/manifest.txt"

    echo "$tag,$ref_path,$timing_path,$metrics_path,$gauss_path,$mean_feature_path,$vis_feature_path" >> "$idx"
  done < <(collect_run_tags "$data_dir" "$refs_dir" "$timing_dir")

  echo "[INDEX] wrote $idx"
}

for dataset in "${DATASETS[@]}"; do
  data_dir="$BASE_DIR/datasets/$dataset"
  refs_dir="$BASE_DIR/datasets/${dataset}_refs"
  timing_dir="$data_dir/timing"
  runs_dir="$data_dir/runs"
  safe_root="$data_dir/_safe_snapshots"

  echo "============================================================"
  echo "[DATASET] $dataset"
  echo "[PATH] data_dir=$data_dir"

  if [[ ! -d "$data_dir" ]]; then
    echo "[WARN] missing dataset directory: $data_dir"
    continue
  fi

  mkdir -p "$data_dir/features" "$data_dir/view_features" "$timing_dir" "$runs_dir" "$safe_root"

  mean_dir="$data_dir/features/$MODEL"
  vis_dir="$data_dir/features/${MODEL}_visible_only"
  view_dir="$data_dir/view_features/$MODEL"

  mean_count=$(count_npz "$mean_dir")
  vis_count=$(count_npz "$vis_dir")
  view_count=$(count_npz "$view_dir")

  echo "[CHECK] mean_feature_dir=$mean_dir count_npz=$mean_count"
  echo "[CHECK] visible_only_feature_dir=$vis_dir count_npz=$vis_count"
  echo "[CHECK] view_feature_dir=$view_dir count_npz=$view_count"

  if [[ "$mean_count" -gt 0 ]]; then
    echo "[CHECK] legacy mean features exist and can be reused as baseline"
    safe_snapshot_legacy_mean "$data_dir" "$mean_dir" "$safe_root"
  else
    echo "[CHECK] no legacy mean feature files found under $mean_dir"
  fi

  build_runs_index "$data_dir" "$refs_dir" "$timing_dir" "$runs_dir"
done

echo "============================================================"
echo "Done. Non-destructive organization complete."
echo "- Existing outputs were not moved/deleted."
echo "- Run-level symlink folders were created under datasets/<name>/runs/."
echo "- Legacy mean snapshot (read-only) created when detected."
