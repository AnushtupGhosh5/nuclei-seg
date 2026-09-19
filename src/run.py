from __future__ import annotations

import argparse
from pathlib import Path

from nuclei_seg.datasets import DATASET_INFO, prepare_dataset
from nuclei_seg.engine import evaluate, regenerate_visualizations, rescore, train


def common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Three-branch nuclei segmentation and classification")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="Convert raw annotations to canonical NP/HV/type targets")
    prepare.add_argument("--dataset", choices=DATASET_INFO, required=True)
    prepare.add_argument("--data-dir", default="data")
    prepare.add_argument("--overwrite", action="store_true")

    training = commands.add_parser("train", help="Train the multi-branch U-Net")
    common_parser(training)
    training.add_argument("--dataset", choices=DATASET_INFO, required=True)
    training.add_argument("--architecture", choices=("unet", "hovernet_fast"), default="hovernet_fast")
    training.add_argument("--pretrained-checkpoint")
    training.add_argument("--freeze-encoder-epochs", type=int, default=50)
    training.add_argument("--resume", action="store_true")
    training.add_argument("--run-name", default="unet_hover_baseline")
    training.add_argument("--epochs", type=int, default=100)
    training.add_argument("--batch-size", type=int, default=1, help="Unfrozen physical batch size")
    training.add_argument("--frozen-batch-size", type=int, default=2)
    training.add_argument("--val-batch-size", type=int, default=2)
    training.add_argument("--gradient-accumulation-steps", type=int, default=4)
    training.add_argument("--frozen-gradient-accumulation-steps", type=int, default=8)
    training.add_argument("--patch-size", type=int, default=256)
    training.add_argument("--workers", type=int, default=4)
    training.add_argument("--learning-rate", type=float, default=1e-4)
    training.add_argument("--weight-decay", type=float, default=1e-4)
    training.add_argument("--val-fraction", type=float, default=0.2)
    training.add_argument("--train-patches-per-image", type=int, default=8)
    training.add_argument("--val-patches-per-image", type=int, default=2)
    training.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after this many epochs without lower validation loss; 0 disables it",
    )
    training.add_argument(
        "--foreground-probability",
        type=float,
        default=0.75,
        help="Fraction of training crops centered on a uniformly selected foreground class",
    )
    training.add_argument(
        "--encoder-channels",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512, 1024],
        help="U-Net channels; the default is the original 2015 architecture",
    )

    evaluation = commands.add_parser("evaluate", help="Run tiled test inference, watershed, and metrics")
    common_parser(evaluation)
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--dataset", choices=DATASET_INFO)
    evaluation.add_argument("--architecture", choices=("unet", "hovernet_fast"))
    evaluation.add_argument("--run-name", default="unet_hover_baseline")
    evaluation.add_argument("--patch-size", type=int, default=256)
    evaluation.add_argument("--overlap", type=int, default=64)
    evaluation.add_argument(
        "--tile-output-size",
        type=int,
        default=164,
        help="Valid center kept from each input tile (164 matches HoVer-Net fast mode)",
    )
    evaluation.add_argument("--batch-size", type=int, default=4)
    evaluation.add_argument("--nucleus-threshold", type=float, default=0.5)
    evaluation.add_argument("--boundary-threshold", type=float, default=0.4)
    evaluation.add_argument("--min-size", type=int, default=10)
    evaluation.add_argument("--type-boundary-weight", type=float, default=0.0)
    evaluation.add_argument(
        "--visualizations",
        type=int,
        default=12,
        help="Number of detailed input/GT/prediction/NP/HV figures to save",
    )

    scoring = commands.add_parser("rescore", help="Recompute metrics from saved predictions")
    common_parser(scoring)
    scoring.add_argument("--dataset", choices=DATASET_INFO, required=True)
    scoring.add_argument("--run-name", required=True)

    visualizing = commands.add_parser(
        "visualize", help="Regenerate detailed figures from saved predictions"
    )
    common_parser(visualizing)
    visualizing.add_argument("--dataset", choices=DATASET_INFO, required=True)
    visualizing.add_argument("--run-name", required=True)
    visualizing.add_argument("--visualizations", type=int, default=12)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        path = prepare_dataset(args.dataset, Path(args.data_dir), args.overwrite)
        print(f"Prepared dataset manifest: {path}")
    elif args.command == "train":
        path = train(args)
        print(f"Best checkpoint: {path}")
    elif args.command == "evaluate":
        path = evaluate(args)
        print(f"Evaluation results: {path}")
    elif args.command == "rescore":
        path = rescore(args)
        print(f"Rescored results: {path}")
    else:
        path = regenerate_visualizations(args)
        print(f"Visualizations: {path}")


if __name__ == "__main__":
    main()
