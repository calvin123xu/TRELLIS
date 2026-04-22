import argparse
import os
import sys

import torch

os.environ["ATTN_BACKEND"] = "xformers"
os.environ["SPARSE_ATTN_BACKEND"] = "xformers"
os.environ["SPCONV_ALGO"] = "native"

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from trellis import models


DEFAULT_ENCODER = "microsoft/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16"
DEFAULT_DECODER = "microsoft/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16"


def maybe_export(pretrained_path: str, out_path: str, force: bool) -> None:
    if os.path.exists(out_path) and not force:
        print(f"[SKIP] {out_path} already exists")
        return

    print(f"[LOAD] {pretrained_path}")
    model = models.from_pretrained(pretrained_path)
    state_dict = model.state_dict()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(state_dict, out_path)
    print(f"[SAVE] {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--encoder_pretrained", type=str, default=DEFAULT_ENCODER)
    parser.add_argument("--decoder_pretrained", type=str, default=DEFAULT_DECODER)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    encoder_out = os.path.join(args.output_dir, "slat_enc_swin8_B_64l8_fp16.pt")
    decoder_out = os.path.join(args.output_dir, "slat_dec_gs_swin8_B_64l8gs32_fp16.pt")

    maybe_export(args.encoder_pretrained, encoder_out, args.force)
    maybe_export(args.decoder_pretrained, decoder_out, args.force)


if __name__ == "__main__":
    main()
