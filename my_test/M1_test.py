import os
import csv
import traceback
import imageio

os.environ["ATTN_BACKEND"] = "xformers"
os.environ["SPCONV_ALGO"] = "native"

from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# -----------------------------
# Config
# -----------------------------
CSV_PATH = "test_cases.csv"
OUTPUT_ROOT = "outputs_text_m1"
RUN_LOG_PATH = "run_log.csv"
SEEDS = [0, 1, 2]

# Optional: put HF cache on scratch if needed
# os.environ["HF_HOME"] = "/speed-scratch/qiaoyu/hf_home"
# os.environ["HUGGINGFACE_HUB_CACHE"] = "/speed-scratch/qiaoyu/hf_home/hub"
# os.environ["TRANSFORMERS_CACHE"] = "/speed-scratch/qiaoyu/hf_home/transformers"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# -----------------------------
# Load pipeline once
# -----------------------------
pipeline = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
pipeline.cuda()

# -----------------------------
# Read CSV
# -----------------------------
test_cases = []
with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    required_cols = {"CaseID", "Level", "StructureType", "Modality", "Prompt", "ImagePath"}
    missing = required_cols - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    for row in reader:
        # only run M1 text-only cases in this script
        if row["Modality"].strip() == "M1":
            test_cases.append(row)

print(f"Loaded {len(test_cases)} M1 test cases from {CSV_PATH}")

# -----------------------------
# Prepare run log
# -----------------------------
write_header = not os.path.exists(RUN_LOG_PATH)
with open(RUN_LOG_PATH, "a", newline="", encoding="utf-8") as log_f:
    log_writer = csv.writer(log_f)
    if write_header:
        log_writer.writerow([
            "CaseID",
            "Level",
            "StructureType",
            "Modality",
            "Seed",
            "Prompt",
            "CaseDir",
            "Status",
            "Error"
        ])

    # -----------------------------
    # Run all cases
    # -----------------------------
    for case in test_cases:
        case_id = case["CaseID"].strip()
        level = case["Level"].strip()
        structure_type = case["StructureType"].strip()
        modality = case["Modality"].strip()
        prompt = case["Prompt"].strip()

        for seed in SEEDS:
            case_dir = os.path.join(OUTPUT_ROOT, case_id, f"seed{seed}")
            os.makedirs(case_dir, exist_ok=True)

            print("=" * 80)
            print(f"Running {case_id} | seed={seed}")
            print(f"Prompt: {prompt}")

            try:
                outputs = pipeline.run(
                    prompt,
                    seed=seed,
                    # Optional parameters:
                    # sparse_structure_sampler_params={
                    #     "steps": 12,
                    #     "cfg_strength": 7.5,
                    # },
                    # slat_sampler_params={
                    #     "steps": 12,
                    #     "cfg_strength": 7.5,
                    # },
                )

                # Save prompt text for traceability
                with open(os.path.join(case_dir, "prompt.txt"), "w", encoding="utf-8") as pf:
                    pf.write(prompt + "\n")

                # Render Gaussian video
                video = render_utils.render_video(outputs["gaussian"][0])["color"]
                imageio.mimsave(os.path.join(case_dir, "gs.mp4"), video, fps=30)

                # Render Radiance Field video
                video = render_utils.render_video(outputs["radiance_field"][0])["color"]
                imageio.mimsave(os.path.join(case_dir, "rf.mp4"), video, fps=30)

                # Render Mesh video
                video = render_utils.render_video(outputs["mesh"][0])["normal"]
                imageio.mimsave(os.path.join(case_dir, "mesh.mp4"), video, fps=30)

                # Export GLB
                glb = postprocessing_utils.to_glb(
                    outputs["gaussian"][0],
                    outputs["mesh"][0],
                    simplify=0.95,
                    texture_size=1024,
                )
                glb.export(os.path.join(case_dir, "mesh.glb"))

                # Save Gaussian as PLY
                outputs["gaussian"][0].save_ply(os.path.join(case_dir, "gaussian.ply"))

                log_writer.writerow([
                    case_id,
                    level,
                    structure_type,
                    modality,
                    seed,
                    prompt,
                    case_dir,
                    "success",
                    ""
                ])
                log_f.flush()

            except Exception as e:
                err_msg = f"{type(e).__name__}: {e}"
                print(f"[ERROR] {case_id} seed={seed} failed: {err_msg}")
                traceback.print_exc()

                log_writer.writerow([
                    case_id,
                    level,
                    structure_type,
                    modality,
                    seed,
                    prompt,
                    case_dir,
                    "failed",
                    err_msg
                ])
                log_f.flush()

print("All done.")
