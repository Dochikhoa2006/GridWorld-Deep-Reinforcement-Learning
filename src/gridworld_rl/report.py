"""Deterministically render experiment metrics into portfolio-ready figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .reproducibility import sha256_file

DISPLAY_NAMES = {
    "dqn": "DQN",
    "double_dqn": "Double DQN",
    "expected_sarsa": "Expected SARSA",
    "cql": "CQL",
}


def _load_metrics(run_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    directory = Path(run_dir)
    metrics_path = directory / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"Metrics file not found: {metrics_path}. Run training before report generation."
        )
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid metrics JSON in {metrics_path}: {exc}") from exc
    if not isinstance(metrics.get("algorithms"), dict) or not metrics["algorithms"]:
        raise ValueError(f"{metrics_path} has no algorithm metrics.")
    return directory, metrics


def generate_report(run_dir: str | Path) -> list[Path]:
    """Regenerate figures and a Markdown summary using only saved metrics."""

    directory, metrics = _load_metrics(run_dir)
    algorithms = list(metrics["algorithms"])
    names = [DISPLAY_NAMES.get(name, name) for name in algorithms]

    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(13, 5))
    for algorithm in algorithms:
        history = metrics["algorithms"][algorithm]["training_history"]
        loss_axis.plot(
            range(1, len(history["total_loss"]) + 1),
            history["total_loss"],
            marker="o",
            markersize=3,
            label=DISPLAY_NAMES.get(algorithm, algorithm),
        )
    loss_axis.set_title("Offline training objective")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Mean loss")
    loss_axis.grid(alpha=0.25)
    loss_axis.legend()

    accuracies = [
        100.0 * metrics["algorithms"][algorithm]["evaluation"]["accuracy"]
        for algorithm in algorithms
    ]
    bars = accuracy_axis.bar(names, accuracies, color=plt.cm.tab10.colors[: len(names)])
    accuracy_axis.set_title("Provided evaluation-split action agreement")
    accuracy_axis.set_ylabel("Agreement (%)")
    accuracy_axis.set_ylim(0, max(100.0, max(accuracies) * 1.15))
    accuracy_axis.tick_params(axis="x", rotation=15)
    for bar, value in zip(bars, accuracies, strict=True):
        accuracy_axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.2f}%",
            ha="center",
            fontsize=9,
        )
    figure.suptitle("Gridworld-10 offline RL comparison")
    figure.tight_layout()
    overview_path = directory / "report.png"
    figure.savefig(overview_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    columns = 2
    rows = (len(algorithms) + columns - 1) // columns
    matrix_figure, axes = plt.subplots(
        rows, columns, figsize=(10, 4.5 * rows), squeeze=False
    )
    for axis, algorithm in zip(axes.flat, algorithms, strict=False):
        matrix = np.asarray(
            metrics["algorithms"][algorithm]["evaluation"]["confusion_matrix"]
        )
        image = axis.imshow(matrix, cmap="Blues")
        threshold = matrix.max() / 2 if matrix.size else 0
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column,
                    row,
                    str(int(matrix[row, column])),
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] > threshold else "black",
                )
        axis.set_title(DISPLAY_NAMES.get(algorithm, algorithm))
        axis.set_xlabel("Predicted action")
        axis.set_ylabel("True action")
        axis.set_xticks(range(matrix.shape[1]))
        axis.set_yticks(range(matrix.shape[0]))
        matrix_figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    for axis in list(axes.flat)[len(algorithms) :]:
        axis.axis("off")
    matrix_figure.suptitle("Provided evaluation-split confusion matrices")
    matrix_figure.tight_layout()
    confusion_path = directory / "confusion_matrices.png"
    matrix_figure.savefig(confusion_path, dpi=160, bbox_inches="tight")
    plt.close(matrix_figure)

    first_evaluation = metrics["algorithms"][algorithms[0]]["evaluation"]
    action_keys = sorted(first_evaluation["per_action_recall"], key=int)
    recall_headers = " | ".join(f"Recall a{action}" for action in action_keys)
    lines = [
        "# Experiment summary",
        "",
        f"- Seed: `{metrics['seed']}`",
        f"- Device: `{metrics['device']}`",
        f"- Training rows: `{metrics['dataset']['train_rows']}`",
        f"- Evaluation rows: `{metrics['dataset']['eval_rows']}`",
        (
            "- Observed state-action coverage: "
            f"`{metrics['dataset']['diagnostics']['observed_state_action_pairs']}/"
            f"{metrics['dataset']['diagnostics']['possible_state_action_pairs']}`"
        ),
        (
            "- Exact evaluation/training transition overlap: "
            f"`{100 * metrics['evaluation_split_diagnostics']['exact_training_overlap_fraction']:.2f}%`"
        ),
        (
            "- Unique evaluation transitions: "
            f"`{metrics['evaluation_split_diagnostics']['unique_full_transitions']}/"
            f"{metrics['evaluation_split_diagnostics']['rows']}`"
        ),
        (
            "- Evaluation states with conflicting actions: "
            f"`{metrics['evaluation_split_diagnostics']['states_with_conflicting_actions']}/"
            f"{metrics['evaluation_split_diagnostics']['unique_states']}`"
        ),
        (
            "- Training-majority reference: "
            f"`{100 * metrics['evaluation_split_diagnostics']['training_majority']['accuracy']:.2f}%`"
        ),
        (
            "- Training state-mode reference: "
            f"`{100 * metrics['evaluation_split_diagnostics']['training_state_mode_accuracy']:.2f}%`"
        ),
        (
            "- Evaluation-fitted state-mode ceiling: "
            f"`{100 * metrics['evaluation_split_diagnostics']['evaluation_state_mode_ceiling']:.2f}%`"
        ),
        "",
        f"| Algorithm | Agreement | {recall_headers} |",
        f"|---|---:|{'---:|' * len(action_keys)}",
    ]
    for algorithm in algorithms:
        evaluation = metrics["algorithms"][algorithm]["evaluation"]
        recalls = evaluation["per_action_recall"]
        recall_values = " | ".join(
            f"{100 * recalls[action]:.2f}%" for action in action_keys
        )
        lines.append(
            f"| {DISPLAY_NAMES.get(algorithm, algorithm)} "
            f"| {100 * evaluation['accuracy']:.2f}% "
            f"| {recall_values} |"
        )
    lines.extend(
        [
            "",
            "Agreement is measured on the provided evaluation split, which "
            "substantially overlaps training. It is neither an out-of-sample "
            "generalization score nor an online-policy return.",
            "",
        ]
    )
    summary_path = directory / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    # Reports may be regenerated later, so refresh the integrity manifest after
    # every render instead of leaving hashes tied to the first rendering.
    manifest_path = directory / "manifest.json"
    manifest = {
        str(path.relative_to(directory)): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != manifest_path.name
    }
    manifest_path.write_text(
        json.dumps({"sha256": manifest}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return [overview_path, confusion_path, summary_path, manifest_path]
