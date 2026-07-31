# Changelog

All notable project changes are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project intends to use [Semantic Versioning](https://semver.org/) for future
tagged releases.

## [Unreleased]

### Added

- A `src/gridworld_rl` package with explicit configuration, data, model, algorithm,
  training, evaluation, checkpoint, reporting, benchmarking, reproducibility, and
  CLI responsibilities.
- A discrete Conservative Q-Learning baseline for offline value regularization.
- Seeded execution and configurable algorithm, epochs, learning rate, batch size,
  discount factor, epsilon, device, target synchronization, and CQL strength.
- PyTorch `DataLoader` training, Huber TD loss, gradient clipping, and target-network
  synchronization.
- Dataset schema, type, range, finite-value, and evaluation-alignment validation.
- Logged-data diagnostics for observed states, state-action coverage, per-action
  counts, terminal fraction, and reward statistics.
- Saved checkpoints, resolved JSON configurations, package/runtime versions, anchored
  source revision and working-tree metadata, dataset fingerprints, metrics,
  confusion matrices, and per-action recall/support.
- Report generation from existing run artifacts.
- Multi-seed benchmark orchestration with aggregate JSON, mean/standard-deviation
  figures, Markdown summaries, nested per-seed runs, and integrity manifests.
- Unit and integration test suites with an enforced coverage threshold.
- Ruff linting/formatting, pre-commit hooks, editor settings, build/smoke checks, and
  a Makefile for local quality gates.
- GitHub Actions matrices for Python 3.11 and 3.12, distribution builds, and
  installed-wheel smoke tests.
- Structured issue forms, a pull-request template, dependency-update configuration,
  and a Code of Conduct.
- A non-root Docker runtime and pinned dependencies.
- MIT license, security policy, contributing guide, citation metadata, architecture
  and reproducibility guides, a technical audit, and a migration guide.
- A provenance-labelled five-seed diagnostic snapshot that preserves class-level
  failure evidence instead of presenting only aggregate accuracy.
- Staged single-run publication with explicit atomic replacement, plus immutable
  benchmark destinations that protect completed evidence from accidental overwrite.
- A `--data-dir` convenience option, option-only invocations that default to
  training, and benchmark hyperparameter overrides for reproducible release runs.

### Changed

- Replaced the legacy one-command `main.py` workflow with
  `python -m gridworld_rl train --config configs/default.json`.
- Adopted the standard `src/` package layout and split tests into `unit/` and
  `integration/` suites; retained `main.py` only as a compatibility shim.
- Defined evaluation as provided-split action agreement; added overlap, duplication,
  ambiguity, and trivial-reference diagnostics; and separated it from both
  out-of-sample generalization and environment-return claims.
- Rewrote the README around reproducible portfolio evidence and explicit limitations.
- Archived the untraceable legacy comparison image under `docs/legacy/` and labelled
  it as historical rather than current evidence.
- Documented that the repository's MIT license does not cover third-party dataset
  files.

### Fixed

- Double DQN now uses the online network for next-action selection and the target
  network for evaluation.
- Expected SARSA now computes the full expectation under the configured
  epsilon-greedy policy.
- DQN terminal transitions are explicitly masked in the tested Bellman target.
- Missing or malformed datasets now fail with actionable validation errors.

## Historical implementation

Before this changelog, the repository used a single `main.py` script and saved a
fixed comparison PNG without a versioned configuration, checkpoint, or metrics
manifest. No release tag is assigned retroactively. See
[`docs/TECHNICAL_AUDIT.md`](docs/TECHNICAL_AUDIT.md) for the detailed review and
[`docs/MIGRATION.md`](docs/MIGRATION.md) for the upgrade procedure.

[Unreleased]: https://github.com/Dochikhoa2006/GridWorld-Deep-Reinforcement-Learning/commits/main
