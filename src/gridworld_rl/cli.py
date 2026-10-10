"""Command-line interface for training, validation, and report generation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from . import __version__
from .config import SUPPORTED_ALGORITHMS, ExperimentConfig

COMMANDS = frozenset(
    {
        "train",
        "report",
        "benchmark",
        "validate",
        "verify",
        "inspect-checkpoint",
        "export-policy",
        "predict",
        "predict-ensemble",
        "compare-checkpoints",
        "audit-policy-support",
        "evaluate-checkpoint",
        "compare-evaluations",
        "compare-runs",
        "audit-run",
        "audit-benchmark",
        "export-supported-policy",
        "evaluate-supported-policy",
        "compare-datasets",
    }
)
ROOT_ONLY_OPTIONS = frozenset({"-h", "--help", "--version"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gridworld-rl",
        description="Reproducible offline RL baselines for Gridworld-10.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser(
        "train", help="train algorithms, evaluate them, and write artifacts"
    )
    _add_config_argument(train)
    _add_training_override_arguments(train, include_seed=True)
    train.add_argument("--output-dir", help="override output.directory")
    train.add_argument("--run-name", help="override output.run_name")
    train.add_argument(
        "--overwrite",
        action="store_true",
        help="atomically replace an existing run directory",
    )

    report = subparsers.add_parser(
        "report", help="regenerate figures and summary from a saved metrics file"
    )
    report.add_argument(
        "--run-dir", required=True, help="experiment artifact directory"
    )

    benchmark = subparsers.add_parser(
        "benchmark",
        help="run repeated seeded experiments and aggregate their metrics",
    )
    _add_config_argument(benchmark)
    _add_training_override_arguments(benchmark, include_seed=False)
    benchmark.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help="two or more declared random seeds are recommended",
    )
    benchmark.add_argument(
        "--output-dir",
        default="artifacts/benchmarks",
        help="benchmark output directory (default: artifacts/benchmarks)",
    )
    benchmark.add_argument(
        "--name",
        default="latest",
        help="benchmark directory name (default: latest)",
    )

    validate = subparsers.add_parser(
        "validate", help="validate configured CSV schemas and evaluation alignment"
    )
    _add_config_argument(validate)
    validate.add_argument(
        "--json", action="store_true", help="print dataset diagnostics as JSON"
    )
    verify = subparsers.add_parser(
        "verify", help="verify saved experiment or benchmark artifact integrity"
    )
    verify.add_argument("--run-dir", required=True, help="artifact directory to verify")
    verify.add_argument(
        "--json", action="store_true", help="print one JSON result to stdout"
    )
    inspect = subparsers.add_parser(
        "inspect-checkpoint", help="validate and summarize a saved model checkpoint"
    )
    inspect.add_argument("--checkpoint", required=True, help="saved model checkpoint")
    inspect.add_argument("--json", action="store_true", help="print metadata as JSON")
    export = subparsers.add_parser(
        "export-policy", help="export Q-values and greedy actions for every state"
    )
    export.add_argument("--checkpoint", required=True, help="saved model checkpoint")
    export.add_argument("--output", required=True, help="new JSON output file")
    export.add_argument(
        "--device", default="cpu", choices=["cpu", "auto", "cuda", "mps"]
    )
    export.add_argument("--batch-size", type=int, default=1024)
    supported_export = subparsers.add_parser(
        "export-supported-policy",
        help="export greedy actions restricted to logged actions at observed states",
    )
    supported_export.add_argument("--checkpoint", required=True)
    supported_export.add_argument("--train", required=True)
    supported_export.add_argument("--output", required=True)
    supported_export.add_argument(
        "--device", default="cpu", choices=["cpu", "auto", "cuda", "mps"]
    )
    supported_export.add_argument("--batch-size", type=int, default=1024)
    supported_eval = subparsers.add_parser(
        "evaluate-supported-policy",
        help="compare logged-support-constrained and original policy agreement",
    )
    supported_eval.add_argument("--checkpoint", required=True)
    supported_eval.add_argument("--train", required=True)
    supported_eval.add_argument("--challenge", required=True)
    supported_eval.add_argument("--solution", required=True)
    supported_eval.add_argument(
        "--device", default="cpu", choices=["cpu", "auto", "cuda", "mps"]
    )
    supported_eval.add_argument("--batch-size", type=int, default=1024)
    supported_eval.add_argument("--bootstrap-replicates", type=int, default=0)
    supported_eval.add_argument("--bootstrap-seed", type=int, default=0)
    supported_eval.add_argument("--confidence-level", type=float, default=0.95)
    predict = subparsers.add_parser(
        "predict", help="predict actions for states listed in a CSV file"
    )
    predict.add_argument("--checkpoint", required=True, help="saved model checkpoint")
    predict.add_argument("--input", required=True, help="CSV with one state column")
    predict.add_argument("--output", required=True, help="new predictions CSV file")
    predict.add_argument(
        "--device", default="cpu", choices=["cpu", "auto", "cuda", "mps"]
    )
    predict.add_argument("--batch-size", type=int, default=1024)
    predict.add_argument(
        "--train",
        help="restrict actions at observed states to those in this training CSV",
    )
    predict.add_argument(
        "--min-action-count",
        type=int,
        default=1,
        help="minimum logged count for an eligible action (requires --train)",
    )
    predict.add_argument(
        "--compact", action="store_true", help="omit per-action Q-value columns"
    )
    ensemble = subparsers.add_parser(
        "predict-ensemble",
        help="majority-vote CSV predictions from multiple checkpoints",
    )
    ensemble.add_argument("--checkpoints", nargs="+", required=True)
    ensemble.add_argument(
        "--weights",
        nargs="+",
        type=float,
        help="positive vote weights in the same order as --checkpoints",
    )
    ensemble.add_argument("--input", required=True, help="CSV with one state column")
    ensemble.add_argument("--output", required=True, help="new predictions CSV file")
    ensemble.add_argument(
        "--device", default="cpu", choices=["cpu", "auto", "cuda", "mps"]
    )
    ensemble.add_argument("--batch-size", type=int, default=1024)
    ensemble.add_argument(
        "--train", help="restrict each model vote to logged actions at observed states"
    )
    ensemble.add_argument(
        "--min-action-count",
        type=int,
        default=1,
        help="minimum logged count for an eligible action (requires --train)",
    )
    compare = subparsers.add_parser(
        "compare-checkpoints",
        help="compare saved greedy policies across every discrete state",
    )
    compare.add_argument("--checkpoints", nargs="+", required=True)
    compare.add_argument(
        "--train", help="validated training CSV for observed-state support diagnostics"
    )
    compare.add_argument(
        "--device", default="cpu", choices=["cpu", "auto", "cuda", "mps"]
    )
    compare.add_argument("--batch-size", type=int, default=1024)
    audit = subparsers.add_parser(
        "audit-policy-support",
        help="check whether greedy actions at observed states occur in training data",
    )
    audit.add_argument("--checkpoint", required=True)
    audit.add_argument(
        "--train", required=True, help="validated training transition CSV"
    )
    audit.add_argument(
        "--device", default="cpu", choices=["cpu", "auto", "cuda", "mps"]
    )
    audit.add_argument("--batch-size", type=int, default=1024)
    evaluate = subparsers.add_parser(
        "evaluate-checkpoint",
        help="score a saved checkpoint on aligned evaluation CSVs",
    )
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--challenge", required=True)
    evaluate.add_argument("--solution", required=True)
    evaluate.add_argument(
        "--train", help="training CSV for overlap and split diagnostics"
    )
    evaluate.add_argument(
        "--device", default="cpu", choices=["cpu", "auto", "cuda", "mps"]
    )
    evaluate.add_argument("--batch-size", type=int, default=1024)
    paired = subparsers.add_parser(
        "compare-evaluations",
        help="score multiple checkpoints on the same evaluation rows",
    )
    paired.add_argument("--checkpoints", nargs="+", required=True)
    paired.add_argument("--challenge", required=True)
    paired.add_argument("--solution", required=True)
    paired.add_argument("--train", help="training CSV for overlap diagnostics")
    paired.add_argument(
        "--device", default="cpu", choices=["cpu", "auto", "cuda", "mps"]
    )
    paired.add_argument("--batch-size", type=int, default=1024)
    paired.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=0,
        help="state-cluster bootstrap draws for paired accuracy intervals (default: off)",
    )
    paired.add_argument("--bootstrap-seed", type=int, default=0)
    paired.add_argument("--confidence-level", type=float, default=0.95)
    runs = subparsers.add_parser(
        "compare-runs", help="compare verified saved runs and their evaluation metrics"
    )
    runs.add_argument("--left", required=True, help="first experiment run directory")
    runs.add_argument("--right", required=True, help="second experiment run directory")
    run_audit = subparsers.add_parser(
        "audit-run", help="cross-check contents of a saved experiment run"
    )
    run_audit.add_argument("--run-dir", required=True)
    run_audit.add_argument(
        "--data-dir", help="original train and evaluation CSVs for inference checks"
    )
    benchmark_audit = subparsers.add_parser(
        "audit-benchmark",
        help="cross-check nested runs and a saved benchmark aggregate",
    )
    benchmark_audit.add_argument("--benchmark-dir", required=True)
    datasets = subparsers.add_parser(
        "compare-datasets",
        help="measure coverage and distribution shift between CSV logs",
    )
    datasets.add_argument("--left", required=True)
    datasets.add_argument("--right", required=True)
    datasets.add_argument("--num-states", type=int, default=100)
    datasets.add_argument("--num-actions", type=int, default=4)
    return parser


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        help=(
            "experiment JSON configuration; when omitted, use the built-in "
            "defaults represented by configs/default.json"
        ),
    )
    parser.add_argument(
        "--data-dir",
        help=(
            "directory containing train.csv, eval_challenge.csv, and eval_solution.csv"
        ),
    )


def _add_training_override_arguments(
    parser: argparse.ArgumentParser, *, include_seed: bool
) -> None:
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=SUPPORTED_ALGORITHMS,
        help="algorithms to train in the given order",
    )
    parser.add_argument("--epochs", type=int, help="override training.epochs")
    parser.add_argument(
        "--learning-rate", type=float, help="override training.learning_rate"
    )
    parser.add_argument("--batch-size", type=int, help="override training.batch_size")
    parser.add_argument("--learning-rate-schedule", choices=["constant", "cosine"])
    parser.add_argument(
        "--warmup-steps", type=int, help="optimizer updates for linear warmup"
    )
    parser.add_argument(
        "--min-learning-rate-ratio",
        type=float,
        help="cosine floor as a fraction of the base rate",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        help="minibatches per optimizer update (default: 1)",
    )
    parser.add_argument(
        "--max-optimizer-steps",
        type=int,
        help="stop after this many optimizer updates, including partial epochs",
    )
    parser.add_argument("--gamma", type=float, help="override training.gamma")
    parser.add_argument("--epsilon", type=float, help="override training.epsilon")
    parser.add_argument("--device", help="override training.device (auto/cpu/cuda/mps)")
    if include_seed:
        parser.add_argument("--seed", type=int, help="override training.seed")
    parser.add_argument("--cql-alpha", type=float, help="override training.cql_alpha")
    parser.add_argument(
        "--target-update-tau",
        type=float,
        help="target-network blend at each update (default: 1, full copy)",
    )


def _load_config(path: str | None, data_dir: str | None = None) -> ExperimentConfig:
    config = ExperimentConfig.from_json(path) if path else ExperimentConfig()
    if data_dir is not None:
        directory = Path(data_dir)
        config.dataset.train = str(directory / "train.csv")
        config.dataset.eval_challenge = str(directory / "eval_challenge.csv")
        config.dataset.eval_solution = str(directory / "eval_solution.csv")
        config.validate()
    return config


def _apply_training_overrides(
    config: ExperimentConfig, args: argparse.Namespace
) -> ExperimentConfig:
    for name in (
        "algorithms",
        "epochs",
        "learning_rate",
        "learning_rate_schedule",
        "warmup_steps",
        "min_learning_rate_ratio",
        "batch_size",
        "gradient_accumulation_steps",
        "max_optimizer_steps",
        "gamma",
        "epsilon",
        "device",
        "seed",
        "cql_alpha",
        "target_update_tau",
    ):
        value = getattr(args, name, None)
        if value is not None:
            setattr(config.training, name, value)
    config.validate()
    return config


def _apply_train_overrides(
    config: ExperimentConfig, args: argparse.Namespace
) -> ExperimentConfig:
    config = _apply_training_overrides(config, args)
    if args.output_dir is not None:
        config.output.directory = args.output_dir
    if args.run_name is not None:
        config.output.run_name = args.run_name
    if args.overwrite:
        config.output.overwrite = True
    config.validate()
    return config


def _normalize_argv(argv: Sequence[str] | None) -> list[str]:
    """Treat option-only invocations as the default ``train`` command."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if (
        arguments
        and arguments[0] not in COMMANDS
        and arguments[0] not in ROOT_ONLY_OPTIONS
        and arguments[0].startswith("-")
    ):
        return ["train", *arguments]
    return arguments


def _validate(config: ExperimentConfig) -> dict[str, object]:
    from .data import (
        evaluation_split_diagnostics,
        load_evaluation_data,
        load_transition_csv,
        transition_diagnostics,
    )

    train = load_transition_csv(
        config.dataset.train,
        num_states=config.dataset.num_states,
        num_actions=config.dataset.num_actions,
    )
    challenge, solution = load_evaluation_data(
        config.dataset.eval_challenge,
        config.dataset.eval_solution,
        num_states=config.dataset.num_states,
        num_actions=config.dataset.num_actions,
    )
    return {
        "schema_version": 1,
        "train_rows": len(train),
        "eval_rows": len(challenge),
        "training_diagnostics": transition_diagnostics(
            train,
            num_states=config.dataset.num_states,
            num_actions=config.dataset.num_actions,
        ),
        "evaluation_split_diagnostics": evaluation_split_diagnostics(
            train,
            solution,
            num_states=config.dataset.num_states,
            num_actions=config.dataset.num_actions,
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(_normalize_argv(argv))
    try:
        if args.command == "train":
            from .trainer import run_experiment

            config = _apply_train_overrides(
                _load_config(args.config, args.data_dir), args
            )
            run_dir = run_experiment(config)
            print(f"Experiment complete: {run_dir.resolve()}")
        elif args.command == "report":
            from .report import generate_report

            outputs = generate_report(Path(args.run_dir))
            print("Generated report files:")
            for output in outputs:
                print(f"  {output.resolve()}")
        elif args.command == "benchmark":
            from .benchmark import run_benchmark

            benchmark_dir = run_benchmark(
                _apply_training_overrides(
                    _load_config(args.config, args.data_dir), args
                ),
                seeds=args.seeds,
                output_directory=args.output_dir,
                name=args.name,
            )
            print(f"Benchmark complete: {benchmark_dir.resolve()}")
        elif args.command == "validate":
            result = _validate(_load_config(args.config, args.data_dir))
            if args.json:
                print(json.dumps(result, sort_keys=True, allow_nan=False))
            else:
                print(
                    f"Validated {result['train_rows']:,} training transitions and "
                    f"{result['eval_rows']:,} aligned evaluation transitions."
                )
                training = result["training_diagnostics"]
                evaluation = result["evaluation_split_diagnostics"]
                print(
                    "Observed state-action coverage: "
                    f"{100 * training['state_action_coverage']:.2f}%."
                )
                dynamics = training["dynamics"]
                print(
                    "State-action pairs with multiple recorded outcomes: "
                    f"{dynamics['variable_outcome_pairs']} "
                    f"({100 * dynamics['variable_outcome_row_fraction']:.2f}% "
                    "of training rows)."
                )
                print(
                    "Exact evaluation/training transition overlap: "
                    f"{100 * evaluation['exact_training_overlap_fraction']:.2f}%."
                )
        elif args.command == "export-policy":
            from .policy import export_policy

            output = export_policy(
                args.checkpoint,
                args.output,
                device=args.device,
                batch_size=args.batch_size,
            )
            print(f"Policy exported: {output.resolve()}")
        elif args.command == "export-supported-policy":
            from .supported_policy import export_supported_policy

            output = export_supported_policy(
                args.checkpoint,
                args.train,
                args.output,
                device=args.device,
                batch_size=args.batch_size,
            )
            print(f"Supported policy exported: {output.resolve()}")
        elif args.command == "evaluate-supported-policy":
            from .supported_evaluation import evaluate_supported_policy

            result = evaluate_supported_policy(
                args.checkpoint,
                args.train,
                args.challenge,
                args.solution,
                device=args.device,
                batch_size=args.batch_size,
                bootstrap_replicates=args.bootstrap_replicates,
                bootstrap_seed=args.bootstrap_seed,
                confidence_level=args.confidence_level,
            )
            print(json.dumps(result, sort_keys=True, allow_nan=False))
        elif args.command == "inspect-checkpoint":
            from .checkpoints import inspect_checkpoint

            details = inspect_checkpoint(args.checkpoint)
            if args.json:
                print(json.dumps(details, sort_keys=True))
            else:
                print(f"Algorithm: {details['algorithm']}")
                print(
                    f"Network: {details['num_states']} states, "
                    f"{details['num_actions']} actions, "
                    f"hidden layers {details['hidden_sizes']}"
                )
                print(f"Parameters: {details['parameter_count']:,}")
                print(
                    f"Seed: {details['seed']}; optimizer steps: {details['global_steps']}"
                )
                print(f"SHA-256: {details['checkpoint_sha256']}")
        elif args.command == "predict":
            from .inference import predict_csv

            output = predict_csv(
                args.checkpoint,
                args.input,
                args.output,
                device=args.device,
                batch_size=args.batch_size,
                compact=args.compact,
                train_csv=args.train,
                min_action_count=args.min_action_count,
            )
            print(f"Predictions exported: {output.resolve()}")
        elif args.command == "predict-ensemble":
            from .ensemble_inference import predict_ensemble_csv

            output = predict_ensemble_csv(
                args.checkpoints,
                args.input,
                args.output,
                device=args.device,
                batch_size=args.batch_size,
                train_csv=args.train,
                min_action_count=args.min_action_count,
                weights=args.weights,
            )
            print(f"Ensemble predictions exported: {output.resolve()}")
        elif args.command == "compare-checkpoints":
            from .comparison import compare_checkpoints

            result = compare_checkpoints(
                args.checkpoints,
                device=args.device,
                batch_size=args.batch_size,
                train_csv=args.train,
            )
            print(json.dumps(result, sort_keys=True, allow_nan=False))
        elif args.command == "audit-policy-support":
            from .support import audit_policy_support

            result = audit_policy_support(
                args.checkpoint,
                args.train,
                device=args.device,
                batch_size=args.batch_size,
            )
            print(json.dumps(result, sort_keys=True, allow_nan=False))
        elif args.command == "evaluate-checkpoint":
            from .checkpoint_evaluation import evaluate_checkpoint

            result = evaluate_checkpoint(
                args.checkpoint,
                args.challenge,
                args.solution,
                train_csv=args.train,
                device=args.device,
                batch_size=args.batch_size,
            )
            print(json.dumps(result, sort_keys=True, allow_nan=False))
        elif args.command == "compare-evaluations":
            from .paired_evaluation import compare_evaluations

            result = compare_evaluations(
                args.checkpoints,
                args.challenge,
                args.solution,
                train_csv=args.train,
                device=args.device,
                batch_size=args.batch_size,
                bootstrap_replicates=args.bootstrap_replicates,
                bootstrap_seed=args.bootstrap_seed,
                confidence_level=args.confidence_level,
            )
            print(json.dumps(result, sort_keys=True, allow_nan=False))
        elif args.command == "compare-runs":
            from .run_comparison import compare_runs

            result = compare_runs(args.left, args.right)
            print(json.dumps(result, sort_keys=True, allow_nan=False))
        elif args.command == "audit-run":
            from .run_audit import audit_run

            result = audit_run(args.run_dir, data_dir=args.data_dir)
            print(json.dumps(result, sort_keys=True, allow_nan=False))
            return 0 if result["valid"] else 1
        elif args.command == "audit-benchmark":
            from .benchmark_audit import audit_benchmark

            result = audit_benchmark(args.benchmark_dir)
            print(json.dumps(result, sort_keys=True, allow_nan=False))
            return 0 if result["valid"] else 1
        elif args.command == "compare-datasets":
            from .dataset_comparison import compare_datasets

            result = compare_datasets(
                args.left,
                args.right,
                num_states=args.num_states,
                num_actions=args.num_actions,
            )
            print(json.dumps(result, sort_keys=True, allow_nan=False))
        elif args.command == "verify":
            from .integrity import verify_artifacts

            try:
                result = verify_artifacts(args.run_dir)
            except (OSError, ValueError, RuntimeError) as exc:
                if not args.json:
                    raise
                print(
                    json.dumps(
                        {"schema_version": 1, "valid": False, "error": str(exc)},
                        sort_keys=True,
                    )
                )
                return 2
            if args.json:
                print(
                    json.dumps(
                        {"schema_version": 1, "valid": result.valid, **asdict(result)},
                        sort_keys=True,
                    )
                )
                return 0 if result.valid else 1
            if not result.valid:
                for category in ("missing", "modified", "unexpected"):
                    for name in getattr(result, category):
                        print(f"{category}: {name}", file=sys.stderr)
                return 1
            print(f"Integrity verified: {result.checked_files} files checked.")
        else:  # pragma: no cover - argparse enforces the choices
            parser.error(f"Unknown command: {args.command}")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
