"""End-to-end training, evaluation, and artifact persistence."""

from __future__ import annotations

import copy
import importlib.metadata
import json
import math
import platform
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from . import __version__
from .algorithms import calculate_loss
from .config import ExperimentConfig
from .data import (
    TransitionDataset,
    evaluation_split_diagnostics,
    load_evaluation_data,
    load_transition_csv,
    make_dataloader,
    transition_diagnostics,
)
from .evaluation import (
    classification_metrics,
    overlap_sliced_agreement,
    policy_agreement_matrix,
    predict_actions,
)
from .models import QNetwork
from .report import generate_report
from .reproducibility import (
    resolve_device,
    set_global_seed,
    sha256_file,
    source_revision,
)
from .schedules import learning_rate_for_step


def _mean_history(
    batch_metrics: list[tuple[int, dict[str, float]]],
) -> dict[str, float]:
    if not batch_metrics:
        raise RuntimeError("Training produced no batches.")
    return {
        key: sum(size * batch[key] for size, batch in batch_metrics)
        / sum(size for size, _batch in batch_metrics)
        for key in ("td_loss", "cql_loss", "total_loss")
    }


def _require_finite(tensors: dict[str, torch.Tensor], context: str) -> None:
    """Reject invalid tensors with one device synchronization on the normal path."""

    checks = {name: torch.isfinite(value).all() for name, value in tensors.items()}
    if not torch.stack(list(checks.values())).all():
        names = ", ".join(name for name, finite in checks.items() if not finite)
        raise RuntimeError(f"Non-finite {names} during {context}.")


@torch.no_grad()
def update_target_model(target: nn.Module, online: nn.Module, tau: float) -> None:
    """Apply a full copy or blend to target parameters.

    Non-parameter buffers are copied at each update, since QNetwork currently
    has none and integer buffers cannot be blended.
    """

    if (
        isinstance(tau, bool)
        or not isinstance(tau, (int, float))
        or not math.isfinite(tau)
        or not 0 < tau <= 1
    ):
        raise ValueError("tau must be greater than 0 and at most 1.")
    if tau == 1:
        target.load_state_dict(online.state_dict())
        return
    for target_parameter, online_parameter in zip(
        target.parameters(), online.parameters(), strict=True
    ):
        target_parameter.lerp_(online_parameter, tau)
    for target_buffer, online_buffer in zip(
        target.buffers(), online.buffers(), strict=True
    ):
        target_buffer.copy_(online_buffer)


def train_model(
    algorithm: str,
    dataset: TransitionDataset,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[QNetwork, dict[str, Any]]:
    """Train one seeded baseline and return its model and history."""

    config.validate()
    effective_batch_size = (
        config.training.batch_size * config.training.gradient_accumulation_steps
    )
    planned_steps = config.training.epochs * (
        (len(dataset) + effective_batch_size - 1) // effective_batch_size
    )
    total_steps = min(
        planned_steps, config.training.max_optimizer_steps or planned_steps
    )
    # Validate the schedule before constructing models or doing any training.
    learning_rate_for_step(config.training, 0, total_steps)
    set_global_seed(config.training.seed)
    model = QNetwork(
        config.dataset.num_states,
        config.dataset.num_actions,
        config.network.hidden_sizes,
    ).to(device)
    target_model = copy.deepcopy(model).to(device)
    target_model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    loader = make_dataloader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        seed=config.training.seed,
        num_workers=config.training.num_workers,
        pin_memory=device.type == "cuda",
    )

    history: dict[str, list[float]] = {
        "td_loss": [],
        "cql_loss": [],
        "total_loss": [],
        "learning_rate": [],
    }
    global_step = 0
    target_syncs = 1
    completed_epochs = 0
    partial_epoch_batches = 0
    accumulation_steps = config.training.gradient_accumulation_steps
    model.train()
    for _epoch in range(config.training.epochs):
        epoch_metrics: list[tuple[int, dict[str, float]]] = []
        processed_batches = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(loader):
            context = (
                f"{algorithm} training (epoch {_epoch + 1}, batch {batch_index + 1}, "
                f"optimizer update {global_step + 1})"
            )
            states = batch["state"].to(device)
            actions = batch["action"].to(device)
            rewards = batch["reward"].to(device)
            next_states = batch["next_state"].to(device)
            dones = batch["done"].to(device)

            online_q = model(states)
            with torch.no_grad():
                # Terminal targets are exactly the observed reward. Filling
                # their bootstrap values with zero avoids needless inference.
                nonterminal = dones == 0
                has_nonterminal = bool(nonterminal.any())
                next_target_q = online_q.new_zeros(online_q.shape)
                if has_nonterminal:
                    active_next_states = next_states[nonterminal]
                    next_target_q[nonterminal] = target_model(active_next_states)
                if algorithm in {"double_dqn", "expected_sarsa"}:
                    next_online_q = online_q.new_zeros(online_q.shape)
                    if has_nonterminal:
                        next_online_q[nonterminal] = model(active_next_states)
                else:
                    next_online_q = next_target_q
            _require_finite(
                {
                    "online Q-values": online_q,
                    "next online Q-values": next_online_q,
                    "target Q-values": next_target_q,
                },
                context,
            )
            loss, parts = calculate_loss(
                algorithm,
                online_q,
                actions,
                rewards,
                dones,
                next_online_q,
                next_target_q,
                config.training.gamma,
                config.training.epsilon,
                config.training.cql_alpha,
            )
            _require_finite({"loss": loss}, context)
            if any(not math.isfinite(value) for value in parts.values()):
                raise RuntimeError(f"Non-finite loss metrics during {context}.")

            # Weight by transitions, including a short final minibatch/window.
            # Each window has the same objective as its concatenated full batch.
            window_start = (batch_index // accumulation_steps) * accumulation_steps
            window_samples = min(
                accumulation_steps * config.training.batch_size,
                len(dataset) - window_start * config.training.batch_size,
            )
            (loss * (len(states) / window_samples)).backward()
            if (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(
                loader
            ):
                try:
                    nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config.training.gradient_clip_norm,
                        error_if_nonfinite=True,
                    )
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"Gradient clipping failed during {context}: {exc}"
                    ) from exc
                learning_rate = learning_rate_for_step(
                    config.training, global_step, total_steps
                )
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                optimizer.step()
                _require_finite(dict(model.named_parameters()), context)
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % config.training.target_update_interval == 0:
                    update_target_model(
                        target_model, model, config.training.target_update_tau
                    )
                    target_syncs += 1
            epoch_metrics.append((len(states), parts))
            processed_batches += 1
            if global_step == total_steps:
                break

        averages = _mean_history(epoch_metrics)
        for key, value in averages.items():
            history[key].append(value)
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])
        if processed_batches == len(loader):
            completed_epochs += 1
        else:
            partial_epoch_batches = processed_batches
        if global_step == total_steps:
            break

    return model, {
        "training_history": history,
        "global_steps": global_step,
        "planned_global_steps": planned_steps,
        "completed_epochs": completed_epochs,
        "partial_epoch_batches": partial_epoch_batches,
        "target_synchronizations": target_syncs,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _dataset_fingerprints(config: ExperimentConfig) -> dict[str, str]:
    paths = {
        "train": Path(config.dataset.train).expanduser(),
        "eval_challenge": Path(config.dataset.eval_challenge).expanduser(),
        "eval_solution": Path(config.dataset.eval_solution).expanduser(),
    }
    for path in paths.values():
        if not Path(path).is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")
    return {name: sha256_file(path) for name, path in paths.items()}


def _require_stable_datasets(
    config: ExperimentConfig, expected: dict[str, str]
) -> None:
    current = _dataset_fingerprints(config)
    changed = [name for name in expected if current[name] != expected[name]]
    if changed:
        raise RuntimeError(
            "Dataset file(s) changed during the run: " + ", ".join(changed)
        )


def run_experiment(config: ExperimentConfig) -> Path:
    """Validate data, train configured algorithms, and save all artifacts."""

    config.validate()
    set_global_seed(config.training.seed)
    device = resolve_device(config.training.device)
    source_metadata = source_revision()

    run_dir = Path(config.output.directory) / config.output.run_name
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    if run_dir.exists() and not config.output.overwrite:
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. "
            "Choose a new run name or pass --overwrite."
        )
    if run_dir.exists() and (not run_dir.is_dir() or run_dir.is_symlink()):
        raise FileExistsError(f"Run path exists and is not a directory: {run_dir}")

    staging_dir = run_dir.parent / (f".{run_dir.name}.in-progress-{uuid.uuid4().hex}")
    checkpoints_dir = staging_dir / "checkpoints"
    staging_dir.mkdir()
    checkpoints_dir.mkdir()

    try:
        dataset_sha256 = _dataset_fingerprints(config)
        train_frame = load_transition_csv(
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
        _require_stable_datasets(config, dataset_sha256)
        dataset = TransitionDataset(
            train_frame,
            source=config.dataset.train,
            num_states=config.dataset.num_states,
            num_actions=config.dataset.num_actions,
        )
        config.to_json(staging_dir / "config.json")

        metrics: dict[str, Any] = {
            "schema_version": 1,
            "seed": config.training.seed,
            "device": str(device),
            "runtime": {
                "python": platform.python_version(),
                "numpy": importlib.metadata.version("numpy"),
                "pandas": importlib.metadata.version("pandas"),
                "torch": importlib.metadata.version("torch"),
                "matplotlib": importlib.metadata.version("matplotlib"),
                "gridworld_offline_rl": __version__,
            },
            "source": source_metadata,
            "evaluation_split_diagnostics": evaluation_split_diagnostics(
                train_frame,
                solution,
                num_states=config.dataset.num_states,
                num_actions=config.dataset.num_actions,
            ),
            "dataset": {
                "train_rows": len(train_frame),
                "eval_rows": len(challenge),
                "diagnostics": transition_diagnostics(
                    train_frame,
                    num_states=config.dataset.num_states,
                    num_actions=config.dataset.num_actions,
                ),
                "sha256": dataset_sha256,
            },
            "algorithms": {},
        }
        predictions_payload: dict[str, list[int]] = {}
        eval_states = challenge["state"].to_numpy(dtype=np.int64)
        eval_targets = solution["action"].to_numpy(dtype=np.int64)

        for algorithm in config.training.algorithms:
            model, training_metrics = train_model(algorithm, dataset, config, device)
            predictions = predict_actions(
                model,
                eval_states,
                device=device,
                batch_size=max(config.training.batch_size, 256),
            )
            evaluation = classification_metrics(
                eval_targets,
                predictions,
                num_actions=config.dataset.num_actions,
            )
            evaluation["overlap_slices"] = overlap_sliced_agreement(
                train_frame, solution, eval_targets, predictions
            )
            checkpoint_path = checkpoints_dir / f"{algorithm}.pt"
            torch.save(
                {
                    "format_version": 1,
                    "algorithm": algorithm,
                    "model_state_dict": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "network": config.network.__dict__,
                    "num_states": config.dataset.num_states,
                    "num_actions": config.dataset.num_actions,
                    "seed": config.training.seed,
                    "global_steps": training_metrics["global_steps"],
                },
                checkpoint_path,
            )
            metrics["algorithms"][algorithm] = {
                **training_metrics,
                "evaluation": evaluation,
                "checkpoint": str(checkpoint_path.relative_to(staging_dir)),
            }
            predictions_payload[algorithm] = predictions.tolist()

        metrics["policy_agreement"] = policy_agreement_matrix(
            predictions_payload, num_actions=config.dataset.num_actions
        )
        _write_json(staging_dir / "metrics.json", metrics)
        _write_json(
            staging_dir / "predictions.json",
            {
                "predictions": predictions_payload,
                "note": (
                    "Predictions follow evaluation-challenge row order; "
                    "labels are not copied."
                ),
            },
        )
        generate_report(staging_dir)
        _require_stable_datasets(config, dataset_sha256)
        _publish_run_directory(
            staging_dir,
            run_dir,
            overwrite=config.output.overwrite,
        )
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise
    return run_dir


def _publish_run_directory(
    staging_dir: Path,
    run_dir: Path,
    *,
    overwrite: bool,
) -> None:
    """Atomically publish a complete run while preserving an existing run on error."""

    if run_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Run directory appeared during training: {run_dir}. "
            "The completed staging run was not published."
        )
    if run_dir.exists() and (not run_dir.is_dir() or run_dir.is_symlink()):
        raise FileExistsError(f"Run path exists and is not a directory: {run_dir}")
    if not run_dir.exists():
        staging_dir.replace(run_dir)
        return

    backup_dir = run_dir.parent / (f".{run_dir.name}.backup-{uuid.uuid4().hex}")
    run_dir.replace(backup_dir)
    try:
        staging_dir.replace(run_dir)
    except Exception:
        backup_dir.replace(run_dir)
        raise
    shutil.rmtree(backup_dir)
