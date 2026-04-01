from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_project_config
from .data.prepare import prepare_dataset
from .tuning import run_optuna_search
from .trainer import AgeDecoupledTrainer
from .utils import save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Age-Decoupled SurrealGAN pipeline")
    parser.add_argument("--config", type=str, default=None, help="Optional TOML config override")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare-data", help="Prepare processed data artifacts and split manifests")
    train_parser = subparsers.add_parser("train", help="Run model training")
    train_parser.add_argument(
        "--resume-run-dir",
        type=str,
        default=None,
        help="Resume an interrupted training run from an existing run directory.",
    )
    subparsers.add_parser("tune", help="Run Optuna tuning")
    subparsers.add_parser("serve", help="Start the FastAPI backend")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_project_config(args.config)

    if args.command == "prepare-data":
        manifest = prepare_dataset(config)
        print(f"Prepared dataset with {manifest['n_features']} ROI features at {config.paths.processed_dir}")
        return

    if args.command == "train":
        summary = AgeDecoupledTrainer(config, config_path=args.config).train(resume_run_dir=args.resume_run_dir)
        print(f"Training completed. Selected checkpoint: {summary['selected_checkpoint']}")
        return

    if args.command == "tune":
        results = run_optuna_search(config)
        output_path = Path(config.paths.runs_dir) / "optuna_results.json"
        save_json(output_path, results)
        print(f"Optuna search completed. Results written to {output_path}")
        return

    if args.command == "serve":
        from .api.app import run_api

        run_api(config)
        return


if __name__ == "__main__":
    main()
