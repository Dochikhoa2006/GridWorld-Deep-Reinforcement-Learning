"""Command-line interface for training, validation, and report generation."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .config import ExperimentConfig

COMMANDS = frozenset({"train", "report", "benchmark", "validate"})
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
    parser.add_argument("--epochs", type=int, help="override training.epochs")
    parser.add_argument(
        "--learning-rate", type=float, help="override training.learning_rate"
    )
    parser.add_argument("--batch-size", type=int, help="override training.batch_size")
    parser.add_argument("--gamma", type=float, help="override training.gamma")
    parser.add_argument("--epsilon", type=float, help="override training.epsilon")
    parser.add_argument("--device", help="override training.device (auto/cpu/cuda/mps)")
    if include_seed:
        parser.add_argument("--seed", type=int, help="override training.seed")
    parser.add_argument("--cql-alpha", type=float, help="override training.cql_alpha")


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
        "epochs",
        "learning_rate",
        "batch_size",
        "gamma",
        "epsilon",
        "device",
        "seed",
        "cql_alpha",
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


def _validate(config: ExperimentConfig) -> str:
    from .data import load_evaluation_data, load_transition_csv

    train = load_transition_csv(
        config.dataset.train,
        num_states=config.dataset.num_states,
        num_actions=config.dataset.num_actions,
    )
    challenge, _solution = load_evaluation_data(
        config.dataset.eval_challenge,
        config.dataset.eval_solution,
        num_states=config.dataset.num_states,
        num_actions=config.dataset.num_actions,
    )
    return (
        f"Validated {len(train):,} training transitions and "
        f"{len(challenge):,} aligned evaluation transitions."
    )


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
            print(_validate(_load_config(args.config, args.data_dir)))
        else:  # pragma: no cover - argparse enforces the choices
            parser.error(f"Unknown command: {args.command}")
    except (FileExistsError, FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
