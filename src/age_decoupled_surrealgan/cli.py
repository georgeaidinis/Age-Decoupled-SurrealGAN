from __future__ import annotations

import argparse
from pathlib import Path

from .analysis_artifacts import backfill_analysis_artifacts
from .config import load_project_config
from .data.prepare import prepare_dataset
from .selection import select_best_runs
from .tuning import best_trial_config, run_optuna_search
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
    train_from_study_parser = subparsers.add_parser(
        "train-best-from-study",
        help="Load the best Optuna trial from a tuning study and run repeated training with it.",
    )
    train_from_study_parser.add_argument("--repetitions", type=int, default=None, help="Optional repetition override")
    train_from_study_parser.add_argument("--epochs", type=int, default=None, help="Optional epoch override")
    train_from_study_parser.add_argument(
        "--record-path",
        type=str,
        default=None,
        help="Optional JSON path where the selected run summary pointer should be written.",
    )
    train_from_study_parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Optional short experiment name override for the repeated training run.",
    )
    subparsers.add_parser("tune", help="Run Optuna tuning")
    subparsers.add_parser("serve", help="Start the FastAPI backend")
    backfill_parser = subparsers.add_parser("backfill-run-artifacts", help="Generate post-run analysis artifacts for runs")
    backfill_parser.add_argument("--run-dir", type=str, default=None, help="Optional single run directory to backfill")
    backfill_parser.add_argument("--force", action="store_true", help="Regenerate analysis artifacts even if they exist")
    select_parser = subparsers.add_parser(
        "select-best-model",
        help="Rank completed runs by admissibility and agreement to choose the best overall model.",
    )
    select_parser.add_argument(
        "--record-dir",
        type=str,
        default=None,
        help="Optional directory of JSON record files emitted by train-best-from-study.",
    )
    select_parser.add_argument(
        "--output-path",
        type=str,
        default="runs/model_selection/best_model_summary.json",
        help="Where to write the selection summary JSON.",
    )
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

    if args.command == "train-best-from-study":
        trial_config, trial_info = best_trial_config(config)
        if args.repetitions is not None:
            trial_config.training.repetitions = args.repetitions
        if args.epochs is not None:
            trial_config.training.epochs = args.epochs
        trial_config.training.save_every = 0
        trial_config.training.target_regular_checkpoints = 0
        if args.experiment_name:
            trial_config.experiment_name = args.experiment_name
        summary = AgeDecoupledTrainer(trial_config, config_path=args.config).train()
        if args.record_path:
            save_json(
                Path(args.record_path),
                {
                    "run_dir": summary["run_dir"],
                    "selected_checkpoint": summary["selected_checkpoint"],
                    "study": trial_info,
                    "k": trial_config.model.n_processes,
                },
            )
        print(
            "Training completed from best study trial. "
            f"Selected checkpoint: {summary['selected_checkpoint']} "
            f"(study={trial_info['study_name']}, trial={trial_info['number']})"
        )
        return

    if args.command == "tune":
        results = run_optuna_search(config)
        study_slug = config.tuning.study_name.replace("/", "_").replace(" ", "_")
        output_path = Path(config.paths.runs_dir) / f"optuna_{study_slug}_results.json"
        save_json(output_path, results)
        save_json(Path(config.paths.runs_dir) / "optuna_results.json", results)
        print(f"Optuna search completed. Results written to {output_path}")
        return

    if args.command == "serve":
        from .api.app import run_api

        run_api(config)
        return

    if args.command == "backfill-run-artifacts":
        results = backfill_analysis_artifacts(config, run_dir=args.run_dir, force=args.force)
        print(f"Backfilled analysis artifacts for {len(results)} run(s).")
        return

    if args.command == "select-best-model":
        results = select_best_runs(config=config, record_dir=args.record_dir, output_path=args.output_path)
        best = results.get("best_overall")
        if best is None:
            print("No completed runs were found for model selection.")
            return
        print(
            "Best model selection completed. "
            f"Best run: {best['run_name']} "
            f"(K={best['k']}, agreement={best['agreement_score']:.4f}, "
            f"selection={best['selection_score']:.4f}, pass={best['screen_pass']})"
        )
        return


if __name__ == "__main__":
    main()
