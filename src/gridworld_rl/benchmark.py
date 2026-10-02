"""Repeated-seed benchmark orchestration and aggregate reporting."""

from __future__ import annotations

import copy
import json
import math
import shutil
import uuid
from collections.abc import Iterable
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import ExperimentConfig
from .evaluation import EVALUATION_SLICES
from .integrity import verify_artifacts
from .report import DISPLAY_NAMES, SLICE_NAMES
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


def _action_stability(
    run_dirs: list[Path],
    algorithms: set[str],
    seeds: list[int],
    num_actions: int,
    expected_rows: list[int | None],
) -> dict[str, Any] | None:
    """Compare saved actions on aligned evaluation rows across seed pairs."""

    paths = [run_dir / "predictions.json" for run_dir in run_dirs]
    if not any(path.exists() for path in paths):
        return None  # Older run bundles may have metrics without predictions.
    if not all(path.is_file() for path in paths):
        raise ValueError("Benchmark runs have inconsistent prediction artifacts.")
    predictions = []
    for path, row_count in zip(paths, expected_rows, strict=True):
        payload = json.loads(path.read_text(encoding="utf-8"))
        actions = payload.get("predictions") if isinstance(payload, dict) else None
        if not isinstance(actions, dict) or set(actions) != algorithms:
            raise ValueError(f"Invalid benchmark predictions: {path}")
        for algorithm, values in actions.items():
            if (
                not isinstance(values, list)
                or not values
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or not 0 <= value < num_actions
                    for value in values
                )
            ):
                raise ValueError(
                    f"Invalid benchmark predictions for {algorithm}: {path}"
                )
            if row_count is not None and len(values) != row_count:
                raise ValueError(
                    f"Benchmark predictions do not match evaluation rows for {algorithm}: {path}"
                )
        if len({len(values) for values in actions.values()}) != 1:
            raise ValueError(
                f"Benchmark predictions have inconsistent algorithm rows: {path}"
            )
        predictions.append(actions)

    result = {}
    for algorithm in sorted(algorithms):
        lengths = {len(run[algorithm]) for run in predictions}
        if len(lengths) != 1:
            raise ValueError(
                f"Benchmark predictions have inconsistent rows for {algorithm}."
            )
        pairs = []
        for first_index, second_index in combinations(range(len(seeds)), 2):
            first = predictions[first_index][algorithm]
            second = predictions[second_index][algorithm]
            agreement = sum(
                left == right for left, right in zip(first, second, strict=True)
            ) / len(first)
            pairs.append(
                {
                    "first_seed": seeds[first_index],
                    "second_seed": seeds[second_index],
                    "agreement": agreement,
                }
            )
        result[algorithm] = {
            "evaluation_rows": lengths.pop(),
            "pairwise_agreement": _summary(pair["agreement"] for pair in pairs)
            if pairs
            else None,
            "seed_pairs": pairs,
        }
    return result


def _aggregate_policy_agreement(
    metrics: list[dict[str, Any]], algorithms: set[str], seeds: list[int]
) -> dict[str, Any] | None:
    matrices = [run.get("policy_agreement") for run in metrics]
    if all(matrix is None for matrix in matrices):
        return None  # Legacy runs did not save this diagnostic.
    if any(not isinstance(matrix, dict) for matrix in matrices):
        raise ValueError("Benchmark runs have inconsistent policy agreement metrics.")
    order = matrices[0].get("algorithms")
    if (
        not isinstance(order, list)
        or any(not isinstance(name, str) for name in order)
        or set(order) != algorithms
        or len(order) != len(algorithms)
    ):
        raise ValueError("Benchmark policy agreement algorithms are inconsistent.")
    size = len(order)
    row_counts = set()
    for run, matrix in zip(metrics, matrices, strict=True):
        rows = matrix.get("num_examples")
        rates = matrix.get("agreement")
        counts = matrix.get("disagreements")
        if (
            matrix.get("algorithms") != order
            or isinstance(rows, bool)
            or not isinstance(rows, int)
            or rows <= 0
            or not isinstance(rates, list)
            or not isinstance(counts, list)
            or len(rates) != size
            or len(counts) != size
            or any(not isinstance(row, list) or len(row) != size for row in rates)
            or any(not isinstance(row, list) or len(row) != size for row in counts)
        ):
            raise ValueError("Invalid benchmark policy agreement matrix.")
        row_counts.add(rows)
        for algorithm in algorithms:
            evaluation_rows = run["algorithms"][algorithm]["evaluation"].get(
                "num_examples"
            )
            if evaluation_rows is not None and evaluation_rows != rows:
                raise ValueError(
                    "Benchmark policy agreement row count is inconsistent."
                )
        for first in range(size):
            for second in range(size):
                rate = rates[first][second]
                count = counts[first][second]
                if (
                    isinstance(rate, bool)
                    or not isinstance(rate, (int, float))
                    or not math.isfinite(rate)
                    or not 0 <= rate <= 1
                    or isinstance(count, bool)
                    or not isinstance(count, int)
                    or not 0 <= count <= rows
                    or abs(rate - (rows - count) / rows) > 1e-12
                    or rate != rates[second][first]
                    or count != counts[second][first]
                    or (first == second and count != 0)
                ):
                    raise ValueError("Invalid benchmark policy agreement matrix.")
    if len(row_counts) != 1:
        raise ValueError("Benchmark policy agreement row counts are inconsistent.")

    positions = {name: index for index, name in enumerate(order)}
    pairs = []
    for first, second in combinations(sorted(algorithms), 2):
        i, j = positions[first], positions[second]
        per_seed = [
            {
                "seed": seed,
                "agreement": matrix["agreement"][i][j],
                "disagreements": matrix["disagreements"][i][j],
            }
            for seed, matrix in zip(seeds, matrices, strict=True)
        ]
        pairs.append(
            {
                "first_algorithm": first,
                "second_algorithm": second,
                "agreement": _summary(item["agreement"] for item in per_seed),
                "disagreements": _summary(item["disagreements"] for item in per_seed),
                "per_seed": per_seed,
            }
        )
    return {"num_examples": row_counts.pop(), "pairs": pairs}


def aggregate_runs(run_dirs: list[Path], seeds: list[int]) -> dict[str, Any]:
    """Aggregate compatible run metrics without reading solution labels."""

    if not run_dirs or len(run_dirs) != len(seeds):
        raise ValueError("run_dirs and seeds must be non-empty and equally sized.")
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        for seed in seeds
    ) or len(set(seeds)) != len(seeds):
        raise ValueError("Benchmark seeds must be distinct non-negative integers.")
    manifests = [run_dir / "manifest.json" for run_dir in run_dirs]
    if any(path.exists() or path.is_symlink() for path in manifests):
        if not all(path.is_file() and not path.is_symlink() for path in manifests):
            raise ValueError("Benchmark runs have inconsistent integrity manifests.")
        for run_dir in run_dirs:
            verification = verify_artifacts(run_dir)
            if not verification.valid:
                raise ValueError(
                    f"Benchmark run failed integrity verification: {run_dir}: "
                    f"missing={verification.missing}, modified={verification.modified}, "
                    f"unexpected={verification.unexpected}."
                )
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
    if configured_seeds != seeds or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in configured_seeds
    ):
        raise ValueError(
            "Benchmark seed metadata does not match the declared seed schedule."
        )
    if any(
        isinstance(run.get("seed"), bool)
        or not isinstance(run.get("seed"), int)
        or run["seed"] != seed
        for run, seed in zip(metrics, seeds, strict=True)
    ):
        raise ValueError(
            "Benchmark metrics seeds do not match the declared seed schedule."
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
    policy_agreement = _aggregate_policy_agreement(metrics, algorithm_sets[0], seeds)
    stability = _action_stability(
        run_dirs,
        algorithm_sets[0],
        seeds,
        configurations[0]["dataset"]["num_actions"],
        [
            run["algorithms"][next(iter(algorithm_sets[0]))]["evaluation"].get(
                "num_examples"
            )
            for run in metrics
        ],
    )

    algorithms: dict[str, Any] = {}
    for algorithm in sorted(algorithm_sets[0]):
        evaluations = [run["algorithms"][algorithm]["evaluation"] for run in metrics]
        if any(
            isinstance(evaluation.get("accuracy"), bool)
            or not isinstance(evaluation.get("accuracy"), (int, float))
            or not 0 <= evaluation["accuracy"] <= 1
            for evaluation in evaluations
        ):
            raise ValueError(
                f"Benchmark accuracy must be finite and between 0 and 1 for {algorithm}."
            )
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
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
            for evaluation in evaluations
            for value in evaluation["per_action_recall"].values()
        ):
            raise ValueError(
                f"Benchmark per-action recall must be finite and between 0 and 1 for {algorithm}."
            )
        objective_histories = [
            run["algorithms"][algorithm]["training_history"]["total_loss"]
            for run in metrics
        ]
        if any(
            not isinstance(history, list) or not history
            for history in objective_histories
        ):
            raise ValueError(
                f"Benchmark final training objective is missing for {algorithm}."
            )
        final_objectives = [history[-1] for history in objective_histories]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in final_objectives
        ):
            raise ValueError(
                f"Benchmark final training objective must be finite for {algorithm}."
            )
        f1_values = [evaluation.get("macro_f1") for evaluation in evaluations]
        if any(value is not None for value in f1_values) and any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
            for value in f1_values
        ):
            raise ValueError(
                f"Benchmark macro F1 is invalid or missing for {algorithm}."
            )
        sliced = [evaluation.get("overlap_slices") for evaluation in evaluations]
        if any(value is not None for value in sliced):
            if any(
                not isinstance(value, dict) or set(value) != set(EVALUATION_SLICES)
                for value in sliced
            ):
                raise ValueError(
                    f"Benchmark overlap slices are inconsistent for {algorithm}."
                )
            for name in EVALUATION_SLICES:
                results = [value[name] for value in sliced]
                if (
                    any(
                        not isinstance(result, dict)
                        or isinstance(result.get("rows"), bool)
                        or not isinstance(result.get("rows"), int)
                        or result["rows"] < 0
                        or (
                            result.get("accuracy") is not None
                            and (
                                isinstance(result["accuracy"], bool)
                                or not isinstance(result["accuracy"], (int, float))
                                or not math.isfinite(result["accuracy"])
                                or not 0 <= result["accuracy"] <= 1
                            )
                        )
                        or (result["rows"] == 0) != (result["accuracy"] is None)
                        for result in results
                    )
                    or len({result["rows"] for result in results}) != 1
                ):
                    raise ValueError(
                        f"Benchmark overlap slice {name} is inconsistent for {algorithm}."
                    )
            if any(
                sum(value[name]["rows"] for name in EVALUATION_SLICES)
                != evaluation["num_examples"]
                for value, evaluation in zip(sliced, evaluations, strict=True)
            ):
                raise ValueError(
                    f"Benchmark overlap slice rows do not cover evaluation for {algorithm}."
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
            "final_training_objective": _summary(final_objectives),
        }
        if f1_values[0] is not None:
            algorithms[algorithm]["macro_f1"] = _summary(f1_values)
        if sliced[0] is not None:
            algorithms[algorithm]["overlap_slices"] = {
                name: {
                    "rows": sliced[0][name]["rows"],
                    "accuracy": (
                        _summary(value[name]["accuracy"] for value in sliced)
                        if sliced[0][name]["rows"]
                        else None
                    ),
                }
                for name in EVALUATION_SLICES
            }

    paired_comparisons = []
    for first, second in combinations(sorted(algorithm_sets[0]), 2):
        # Pair within each run; never compare independently sorted score lists.
        differences = [
            run["algorithms"][first]["evaluation"]["accuracy"]
            - run["algorithms"][second]["evaluation"]["accuracy"]
            for run in metrics
        ]
        paired_comparisons.append(
            {
                "first_algorithm": first,
                "second_algorithm": second,
                "accuracy_difference": _summary(differences),
                "per_seed": [
                    {"seed": seed, "accuracy_difference": difference}
                    for seed, difference in zip(seeds, differences, strict=True)
                ],
                "wins": sum(value > 0 for value in differences),
                "ties": sum(value == 0 for value in differences),
                "losses": sum(value < 0 for value in differences),
            }
        )

    aggregate = {
        "schema_version": 1,
        "seeds": seeds,
        "num_runs": len(run_dirs),
        "algorithms": algorithms,
        "paired_comparisons": paired_comparisons,
        "dataset_sha256": metrics[0]["dataset"]["sha256"],
        "evaluation_split_diagnostics": metrics[0]["evaluation_split_diagnostics"],
        "provenance": {field: metrics[0].get(field) for field in provenance_fields},
        "run_directories": [str(path) for path in run_dirs],
    }
    if stability is not None:
        aggregate["action_stability"] = stability
    if policy_agreement is not None:
        aggregate["policy_agreement"] = policy_agreement
    return aggregate


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

    has_f1 = all(
        "macro_f1" in aggregate["algorithms"][name] for name in algorithm_names
    )
    f1_header = " | Macro F1 mean | Macro F1 std" if has_f1 else ""
    f1_separator = "|---:|---:" if has_f1 else ""
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
        f"| Algorithm | Agreement mean | Agreement std | Minimum | Maximum{f1_header} |",
        f"|---|---:|---:|---:|---:{f1_separator}|",
    ]
    for algorithm in algorithm_names:
        accuracy = aggregate["algorithms"][algorithm]["accuracy"]
        f1 = aggregate["algorithms"][algorithm].get("macro_f1")
        f1_values = (
            f" | {100 * f1['mean']:.2f}% | {100 * f1['std']:.2f}%"
            if f1 is not None
            else ""
        )
        lines.append(
            f"| {DISPLAY_NAMES.get(algorithm, algorithm)} "
            f"| {100 * accuracy['mean']:.2f}% | {100 * accuracy['std']:.2f}% "
            f"| {100 * accuracy['minimum']:.2f}% "
            f"| {100 * accuracy['maximum']:.2f}%{f1_values} |"
        )
    if all(
        "overlap_slices" in aggregate["algorithms"][name] for name in algorithm_names
    ):
        lines.extend(
            [
                "",
                "## Agreement by training overlap",
                "",
                "Groups are disjoint and cover every evaluation row. "
                "Empty groups are shown as N/A.",
                "",
                "| Algorithm | Group | Rows | Mean agreement | Sample std |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for algorithm in algorithm_names:
            for name in EVALUATION_SLICES:
                result = aggregate["algorithms"][algorithm]["overlap_slices"][name]
                accuracy = result["accuracy"]
                mean = "N/A" if accuracy is None else f"{100 * accuracy['mean']:.2f}%"
                std = "N/A" if accuracy is None else f"{100 * accuracy['std']:.2f}%"
                lines.append(
                    f"| {DISPLAY_NAMES.get(algorithm, algorithm)} "
                    f"| {SLICE_NAMES[name]} | {result['rows']} | {mean} | {std} |"
                )
    comparisons = aggregate.get("paired_comparisons", [])
    if comparisons:
        lines.extend(
            [
                "",
                "## Paired seed comparisons",
                "",
                "Differences are first minus second on matching seeds, in percentage "
                "points (pp). Positive values favor the first algorithm. Wins, ties, "
                "and losses count seeds; ties require equal agreement. These are "
                "descriptive comparisons, not significance tests.",
                "",
                "| First | Second | Mean difference (pp) | Sample std (pp) "
                "| Minimum (pp) | Maximum (pp) | Wins / Ties / Losses |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for comparison in comparisons:
            first = comparison["first_algorithm"]
            second = comparison["second_algorithm"]
            difference = comparison["accuracy_difference"]
            lines.append(
                f"| {DISPLAY_NAMES.get(first, first)} "
                f"| {DISPLAY_NAMES.get(second, second)} "
                f"| {100 * difference['mean']:+.2f} "
                f"| {100 * difference['std']:.2f} "
                f"| {100 * difference['minimum']:+.2f} "
                f"| {100 * difference['maximum']:+.2f} "
                f"| {comparison['wins']} / {comparison['ties']} / {comparison['losses']} |"
            )
    policy_agreement = aggregate.get("policy_agreement")
    if policy_agreement and policy_agreement["pairs"]:
        lines.extend(
            [
                "",
                "## Agreement between algorithms across seeds",
                "",
                "Each algorithm pair is compared within the same seed on aligned "
                "evaluation rows. These action agreements do not use solution labels.",
                "",
                "| First | Second | Mean agreement | Sample std | Mean differing rows |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for pair in policy_agreement["pairs"]:
            first, second = pair["first_algorithm"], pair["second_algorithm"]
            agreement = pair["agreement"]
            lines.append(
                f"| {DISPLAY_NAMES.get(first, first)} "
                f"| {DISPLAY_NAMES.get(second, second)} "
                f"| {100 * agreement['mean']:.2f}% "
                f"| {100 * agreement['std']:.2f}% "
                f"| {pair['disagreements']['mean']:.2f} |"
            )
    stability = aggregate.get("action_stability")
    if stability and all(
        stability[name]["pairwise_agreement"] is not None for name in algorithm_names
    ):
        lines.extend(
            [
                "",
                "## Action stability across seeds",
                "",
                "Each seed pair is compared on the same evaluation rows. "
                "Agreement measures identical predicted actions without using labels.",
                "",
                "| Algorithm | Rows | Seed pairs | Mean agreement | Sample std |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for algorithm in algorithm_names:
            result = stability[algorithm]
            agreement = result["pairwise_agreement"]
            lines.append(
                f"| {DISPLAY_NAMES.get(algorithm, algorithm)} "
                f"| {result['evaluation_rows']} | {len(result['seed_pairs'])} "
                f"| {100 * agreement['mean']:.2f}% "
                f"| {100 * agreement['std']:.2f}% |"
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

    destination = Path(output_directory) / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"Benchmark directory already exists: {destination}. "
            "Choose a new benchmark name to preserve immutable evidence."
        )
    benchmark_dir = destination.parent / (
        f".{destination.name}.in-progress-{uuid.uuid4().hex}"
    )
    runs_directory = benchmark_dir / "runs"
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
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"Benchmark directory appeared during training: {destination}. "
                "The completed staging benchmark was not published."
            )
        benchmark_dir.replace(destination)
    except Exception:
        if benchmark_dir.exists():
            shutil.rmtree(benchmark_dir)
        raise
    return destination
