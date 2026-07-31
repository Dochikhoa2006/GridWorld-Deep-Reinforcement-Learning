"""Repeated-seed benchmark orchestration and aggregate reporting."""

from __future__ import annotations

import copy
import json
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import ExperimentConfig
from .report import DISPLAY_NAMES
from .reproducibility import sha256_file
from .trainer import run_experiment


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _summary(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty metric.")
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def aggregate_runs(run_dirs: list[Path], seeds: list[int]) -> dict[str, Any]:
    """Aggregate compatible run metrics without reading solution labels."""

    if not run_dirs or len(run_dirs) != len(seeds):
        raise ValueError("run_dirs and seeds must be non-empty and equally sized.")
    metrics = [
        json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        for run_dir in run_dirs
    ]
    configurations = [
        json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        for run_dir in run_dirs
    ]
    configured_seeds = [
        configuration["training"]["seed"] for configuration in configurations
    ]
    if configured_seeds != seeds:
        raise ValueError(
            "Benchmark seed metadata does not match the declared seed schedule."
        )
    normalized_configurations = []
    for configuration in configurations:
        comparable = copy.deepcopy(configuration)
        comparable["training"].pop("seed", None)
        comparable.pop("output", None)
        normalized_configurations.append(comparable)
    if any(
        configuration != normalized_configurations[0]
        for configuration in normalized_configurations[1:]
    ):
        raise ValueError(
            "Benchmark runs use different experiment configurations beyond seed/output."
        )

    provenance_fields = ("device", "runtime", "source")
    for field in provenance_fields:
        if any(run.get(field) != metrics[0].get(field) for run in metrics[1:]):
            raise ValueError(f"Benchmark runs have inconsistent {field} metadata.")
    if any(
        run["dataset"]["sha256"] != metrics[0]["dataset"]["sha256"]
        for run in metrics[1:]
    ):
        raise ValueError("Benchmark runs use different dataset fingerprints.")
    if any(
        run["evaluation_split_diagnostics"]
        != metrics[0]["evaluation_split_diagnostics"]
        for run in metrics[1:]
    ):
        raise ValueError(
            "Benchmark runs have inconsistent evaluation-split diagnostics."
        )

    algorithm_sets = [set(run["algorithms"]) for run in metrics]
    if any(algorithms != algorithm_sets[0] for algorithms in algorithm_sets[1:]):
        raise ValueError("Benchmark runs do not contain the same algorithms.")

    algorithms: dict[str, Any] = {}
    for algorithm in sorted(algorithm_sets[0]):
        evaluations = [run["algorithms"][algorithm]["evaluation"] for run in metrics]
        recall_actions = sorted(
            evaluations[0]["per_action_recall"],
            key=int,
        )
        if any(
            sorted(evaluation["per_action_recall"], key=int) != recall_actions
            for evaluation in evaluations[1:]
        ):
            raise ValueError(
                f"Benchmark runs have inconsistent recall actions for {algorithm}."
            )
        algorithms[algorithm] = {
            "accuracy": _summary(evaluation["accuracy"] for evaluation in evaluations),
            "per_action_recall": {
                action: _summary(
                    evaluation["per_action_recall"][action]
                    for evaluation in evaluations
                )
                for action in recall_actions
            },
            "final_training_objective": _summary(
                run["algorithms"][algorithm]["training_history"]["total_loss"][-1]
                for run in metrics
            ),
        }

    return {
        "schema_version": 1,
        "seeds": seeds,
        "num_runs": len(run_dirs),
        "algorithms": algorithms,
        "dataset_sha256": metrics[0]["dataset"]["sha256"],
        "evaluation_split_diagnostics": metrics[0]["evaluation_split_diagnostics"],
        "provenance": {field: metrics[0].get(field) for field in provenance_fields},
        "run_directories": [str(path) for path in run_dirs],
    }


def generate_benchmark_report(
    benchmark_dir: str | Path,
    aggregate: dict[str, Any] | None = None,
) -> list[Path]:
    """Render aggregate action agreement and a concise Markdown benchmark table."""

    directory = Path(benchmark_dir)
    if aggregate is None:
        aggregate_path = directory / "aggregate_metrics.json"
        if not aggregate_path.is_file():
            raise FileNotFoundError(
                f"Aggregate metrics file not found: {aggregate_path}"
            )
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))

    algorithm_names = list(aggregate["algorithms"])
    labels = [DISPLAY_NAMES.get(name, name) for name in algorithm_names]
    means = [
        100.0 * aggregate["algorithms"][name]["accuracy"]["mean"]
        for name in algorithm_names
    ]
    deviations = [
        100.0 * aggregate["algorithms"][name]["accuracy"]["std"]
        for name in algorithm_names
    ]

    figure, axis = plt.subplots(figsize=(9, 5))
    bars = axis.bar(
        labels,
        means,
        yerr=deviations,
        capsize=5,
        color=plt.cm.tab10.colors[: len(labels)],
    )
    diagnostics = aggregate["evaluation_split_diagnostics"]
    majority_reference = 100.0 * diagnostics["training_majority"]["accuracy"]
    training_mode_reference = 100.0 * diagnostics["training_state_mode_accuracy"]
    evaluation_ceiling = 100.0 * diagnostics["evaluation_state_mode_ceiling"]
    axis.axhline(
        majority_reference,
        color="gray",
        linestyle="--",
        linewidth=1.25,
        label=f"training majority ({majority_reference:.2f}%)",
    )
    axis.axhline(
        training_mode_reference,
        color="black",
        linestyle=":",
        linewidth=1.5,
        label=f"training state mode ({training_mode_reference:.2f}%)",
    )
    axis.axhline(
        evaluation_ceiling,
        color="purple",
        linestyle="-.",
        linewidth=1.25,
        label=f"evaluation state-mode ceiling ({evaluation_ceiling:.2f}%)",
    )
    axis.set_title(
        f"Evaluation-split action agreement across {aggregate['num_runs']} seeded runs"
    )
    axis.set_ylabel("Agreement, mean ± sample standard deviation (%)")
    axis.set_ylim(0, max(100.0, max(means) * 1.2))
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="lower right", fontsize=8)
    axis.tick_params(axis="x", rotation=12)
    for bar, mean in zip(bars, means, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            mean + 1,
            f"{mean:.2f}%",
            ha="center",
            fontsize=9,
        )
    figure.tight_layout()
    figure_path = directory / "benchmark.png"
    figure.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    lines = [
        "# Multi-seed benchmark summary",
        "",
        f"- Seeds: `{', '.join(str(seed) for seed in aggregate['seeds'])}`",
        f"- Runs: `{aggregate['num_runs']}`",
        (
            "- Exact evaluation/training transition overlap: "
            f"`{100 * diagnostics['exact_training_overlap_fraction']:.2f}%`"
        ),
        f"- Training-majority reference: `{majority_reference:.2f}%`",
        f"- Training state-mode reference: `{training_mode_reference:.2f}%`",
        f"- Evaluation state-mode ceiling: `{evaluation_ceiling:.2f}%`",
        "",
        "| Algorithm | Agreement mean | Agreement std | Minimum | Maximum |",
        "|---|---:|---:|---:|---:|",
    ]
    for algorithm in algorithm_names:
        accuracy = aggregate["algorithms"][algorithm]["accuracy"]
        lines.append(
            f"| {DISPLAY_NAMES.get(algorithm, algorithm)} "
            f"| {100 * accuracy['mean']:.2f}% | {100 * accuracy['std']:.2f}% "
            f"| {100 * accuracy['minimum']:.2f}% "
            f"| {100 * accuracy['maximum']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "The metric is agreement on the provided, substantially overlapping "
            "evaluation split—not out-of-sample generalization or online return.",
            "Sample standard deviation is zero when only one seed is supplied.",
            "",
        ]
    )
    summary_path = directory / "benchmark.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return [figure_path, summary_path]


def run_benchmark(
    config: ExperimentConfig,
    *,
    seeds: list[int],
    output_directory: str | Path,
    name: str,
) -> Path:
    """Run one complete experiment per seed and aggregate their metrics."""

    if not seeds or any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        for seed in seeds
    ):
        raise ValueError("seeds must contain non-negative integers.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds cannot contain duplicates.")
    benchmark_name = Path(name)
    if (
        not name
        or benchmark_name.is_absolute()
        or len(benchmark_name.parts) != 1
        or name in {".", ".."}
    ):
        raise ValueError("name must be a single directory name.")

    benchmark_dir = Path(output_directory) / name
    runs_directory = benchmark_dir / "runs"
    benchmark_dir.parent.mkdir(parents=True, exist_ok=True)
    if benchmark_dir.exists():
        raise FileExistsError(
            f"Benchmark directory already exists: {benchmark_dir}. "
            "Choose a new benchmark name to preserve immutable evidence."
        )
    benchmark_dir.mkdir()
    runs_directory.mkdir()

    try:
        run_dirs: list[Path] = []
        for seed in seeds:
            run_config = copy.deepcopy(config)
            run_config.training.seed = seed
            run_config.output.directory = str(runs_directory)
            run_config.output.run_name = f"seed-{seed}"
            run_config.output.overwrite = False
            run_config.validate()
            run_dirs.append(run_experiment(run_config))

        aggregate = aggregate_runs(run_dirs, seeds)
        aggregate["run_directories"] = [
            str(run_dir.relative_to(benchmark_dir)) for run_dir in run_dirs
        ]
        _write_json(
            benchmark_dir / "benchmark_config.json",
            {
                "experiment": config.to_dict(),
                "benchmark": {
                    "seeds": seeds,
                    "output_directory": str(output_directory),
                    "name": name,
                },
            },
        )
        _write_json(benchmark_dir / "aggregate_metrics.json", aggregate)
        generated = generate_benchmark_report(benchmark_dir, aggregate)

        expected_files = [
            benchmark_dir / "benchmark_config.json",
            benchmark_dir / "aggregate_metrics.json",
            *generated,
            *(
                path
                for run_dir in run_dirs
                for path in run_dir.rglob("*")
                if path.is_file()
            ),
        ]
        manifest = {
            str(path.relative_to(benchmark_dir)): sha256_file(path)
            for path in sorted(expected_files)
        }
        _write_json(benchmark_dir / "manifest.json", {"sha256": manifest})
    except Exception:
        shutil.rmtree(benchmark_dir)
        raise
    return benchmark_dir
