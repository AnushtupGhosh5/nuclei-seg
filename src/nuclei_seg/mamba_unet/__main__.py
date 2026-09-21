from __future__ import annotations

import argparse
from pathlib import Path

from .config import MambaUNetConfig
from .engine import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Official Mamba-UNet GLySAC reproduction")
    parser.add_argument("--config", type=Path, default=Path("configs/mamba_unet_glysac.json"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--pretrained-checkpoint", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = MambaUNetConfig.from_json(args.config)
    for name in ("data_root", "output_dir", "pretrained_checkpoint", "resume_checkpoint"):
        value = getattr(args, name)
        if value is not None:
            setattr(config, name, value)
    if args.smoke_test:
        config.smoke_test = True
    output = run_experiment(config)
    print("Experiment complete:", output)


if __name__ == "__main__":
    main()

