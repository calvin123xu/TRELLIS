import argparse
import os
import subprocess
import sys


def _append_optional_arg(cmd, name, value):
    if value is None:
        return
    cmd.extend([name, str(value)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compatibility wrapper: stage1 extract view features, stage2 aggregate voxel features.'
    )
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory containing metadata/renders/voxels and where outputs are saved.')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value.')
    parser.add_argument('--model', type=str, default='dinov2_vitl14_reg',
                        help='Feature extraction model.')
    parser.add_argument('--instances', type=str, default=None,
                        help='Path to file with one sha256 per line.')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--aggregation_mode', type=str, default='visible_only', choices=['visible_only', 'visible_only_zbuf', 'mean'],
                        help='Aggregation mode used by aggregate_features.py.')
    parser.add_argument('--visibility_eps', type=float, default=1e-6,
                        help='Numerical epsilon for visible_only aggregation denominator.')
    parser.add_argument('--visibility_margin', type=float, default=0.0,
                        help='In-bound margin in normalized grid coordinates for visibility mask.')
    parser.add_argument('--zbuf_depth_tolerance', type=float, default=1e-3,
                        help='Depth tolerance for visible_only_zbuf mode in camera-space depth units.')
    parser.add_argument('--zbuf_selection', type=str, default='nearest', choices=['nearest', 'topk', 'soft'],
                        help='Cell-level z-buffer policy for visible_only_zbuf mode.')
    parser.add_argument('--zbuf_topk', type=int, default=1,
                        help='Top-k nearest depths kept per patch cell when zbuf_selection=topk.')
    parser.add_argument('--zbuf_soft_tau', type=float, default=5e-3,
                        help='Depth decay scale for zbuf_selection=soft. Larger values are less aggressive.')
    parser.add_argument('--zbuf_fallback_mode', type=str, default='none', choices=['none', 'visible_only'],
                        help='Optional fallback policy for low-support voxels in visible_only_zbuf mode.')
    parser.add_argument('--zbuf_min_visibility_count', type=float, default=0.0,
                        help='If fallback enabled, voxels below this zbuf support use visible_only fallback.')
    parser.add_argument('--timing_log_interval', type=int, default=100,
                        help='Print timing summary every N processed objects in each stage.')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--skip_view_extraction', action='store_true',
                        help='Skip stage1 and aggregate from precomputed view_features only.')
    parser.add_argument('--output_feature_name', type=str, default=None,
                        help='Output feature directory under features/. Default: model name.')
    opt = parser.parse_args()

    if opt.output_feature_name is None:
        if opt.aggregation_mode == 'visible_only':
            opt.output_feature_name = f'{opt.model}_visible_only'
        elif opt.aggregation_mode == 'visible_only_zbuf':
            opt.output_feature_name = f'{opt.model}_visible_zbuf'
        else:
            opt.output_feature_name = opt.model

    script_dir = os.path.dirname(os.path.abspath(__file__))
    extract_view_script = os.path.join(script_dir, 'extract_view_features.py')
    aggregate_script = os.path.join(script_dir, 'aggregate_features.py')

    common_args = [
        '--output_dir', opt.output_dir,
        '--model', opt.model,
        '--batch_size', str(opt.batch_size),
        '--rank', str(opt.rank),
        '--world_size', str(opt.world_size),
        '--timing_log_interval', str(opt.timing_log_interval),
    ]

    _append_optional_arg(common_args, '--filter_low_aesthetic_score', opt.filter_low_aesthetic_score)
    _append_optional_arg(common_args, '--instances', opt.instances)

    if not opt.skip_view_extraction:
        extract_cmd = [sys.executable, extract_view_script] + common_args
        print('[PIPELINE] Running stage1 extract_view_features.py')
        subprocess.run(extract_cmd, check=True)
    else:
        print('[PIPELINE] Skipping stage1 (extract_view_features.py).')

    aggregate_cmd = [
        sys.executable,
        aggregate_script,
        '--aggregation_mode', opt.aggregation_mode,
        '--visibility_eps', str(opt.visibility_eps),
        '--visibility_margin', str(opt.visibility_margin),
        '--zbuf_depth_tolerance', str(opt.zbuf_depth_tolerance),
        '--zbuf_selection', opt.zbuf_selection,
        '--zbuf_topk', str(opt.zbuf_topk),
        '--zbuf_soft_tau', str(opt.zbuf_soft_tau),
        '--zbuf_fallback_mode', opt.zbuf_fallback_mode,
        '--zbuf_min_visibility_count', str(opt.zbuf_min_visibility_count),
        '--output_feature_name', opt.output_feature_name,
    ] + common_args

    print('[PIPELINE] Running stage2 aggregate_features.py')
    subprocess.run(aggregate_cmd, check=True)
        