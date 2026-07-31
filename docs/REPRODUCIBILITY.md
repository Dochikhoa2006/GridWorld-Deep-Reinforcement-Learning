# Reproducibility Protocol

[README](../README.md) ·
[Architecture](ARCHITECTURE.md) ·
[Technical audit](TECHNICAL_AUDIT.md) ·
[Migration](MIGRATION.md)

## Reproducibility claim

This project provides a **repeatable experiment pipeline**: declared random seeds,
deterministic controls where supported, pinned dependencies, resolved configuration,
dataset fingerprints, runtime metadata, structured outputs, and multi-seed
aggregation.

That is not the same as claiming universal bit-for-bit identity. Exact values can
differ across operating systems, CPU instruction sets, accelerator backends, PyTorch
builds, or kernels without deterministic implementations.

It is useful to separate three levels of evidence:

| Level | Question | What this repository provides |
| --- | --- | --- |
| **Traceable** | Can a result be connected to its inputs and settings? | Source revision/dirty state when available, resolved config, runtime versions, data hashes, checkpoints, metrics, manifest |
| **Repeatable** | Can the same setup rerun the same procedure? | Seeded RNGs, deterministic data order, pinned dependencies, CPU/Docker paths |
| **Statistically defensible** | Does an observed difference persist across independent runs and a predeclared protocol? | Multi-seed tooling; the researcher must choose and disclose the protocol |

## Reference workflow

### 1. Record the source revision

Start from a clean checkout and record the exact commit:

```bash
git status --short
git rev-parse HEAD
```

Each run probes from the installed/editable package path and records
`source.git_commit` plus `source.git_dirty` in `metrics.json`; it never attributes
the caller's current working directory. Dirty status includes tracked changes and
untracked, nonignored files. When no Git checkout is discoverable, including in the
production container, both fields are `null`. The installed
`runtime.gridworld_offline_rl` version is always recorded; retain an external release
tag, source-archive hash, or container image digest when Git fields are unavailable.

### 2. Create an isolated environment

Python 3.11 is the reference runtime:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Confirm the command surface and run the synthetic test suite:

```bash
gridworld-rl --version
make check
```

For the strongest cross-run comparison, use the same dependency file, Python minor
version, and device class.

### 3. Obtain and identify the dataset

Download the Gridworld-10 files from the
[dataset owner's page](https://www.kaggle.com/datasets/sainideeshk/gridworld-10-an-offline-rl-dataset).
Do not commit or redistribute the CSVs under this project's MIT license.

Place the required files at the configured paths:

```text
Gridworld-10_Dataset/
├── train.csv
├── eval_challenge.csv
└── eval_solution.csv
```

Validate the schema and evaluation alignment:

```bash
gridworld-rl validate --config configs/default.json
```

Each completed run writes SHA-256 fingerprints for all three configured CSVs into
`metrics.json`. It also records observed state/action coverage, action counts,
terminal fraction, and reward summary statistics without copying transition rows.
Evaluation diagnostics record exact full-transition overlap with training,
duplicates, conflicting per-state actions, two training-derived references, and an
explicitly evaluation-fitted state-mode ceiling.
Dataset row counts and summary statistics do not uniquely identify a snapshot; use
the hashes when comparing runs.

### 4. Freeze the experiment definition

Copy the reference configuration rather than editing it in place:

```bash
cp configs/default.json configs/showcase.json
```

Review and commit the experiment config before training. At minimum, declare:

- included algorithms;
- network hidden sizes;
- epochs and batch size;
- learning rate and discount factor;
- Expected SARSA epsilon;
- CQL penalty strength;
- target synchronization interval and gradient clipping;
- device class; and
- output naming.

CLI overrides are saved into a single run's resolved `config.json`, but a committed
JSON file remains the clearest specification for a shared benchmark.

### 5. Predeclare a seed schedule

Choose seeds before examining aggregate results. The command below is an example
schedule, not a statistically privileged default:

```bash
gridworld-rl benchmark \
  --config configs/showcase.json \
  --seeds 11 22 33 44 55 \
  --output-dir artifacts/benchmarks \
  --name showcase
```

The benchmark command:

1. deep-copies the base configuration for each seed;
2. writes complete runs under `runs/seed-<seed>/`;
3. checks configuration, source/runtime metadata, data fingerprints, evaluation
   diagnostics, and algorithm compatibility;
4. aggregates action agreement, per-action recall, and final objective;
5. reports mean, sample standard deviation, minimum, and maximum; and
6. hashes the aggregate outputs and all nested run artifacts.

Five seeds demonstrate the mechanism; more or fewer may be appropriate depending on
the expected variance and claim. Decide the sample size and comparison rule before
reading the evaluation results. Benchmark destinations are immutable; the command
refuses an existing name, so assign a new name to every comparison.

### 6. Inspect evidence, not only the headline

Review:

- `benchmark_config.json` for the base experiment and seed schedule;
- `aggregate_metrics.json` for the exact summary values;
- `benchmark.md` and `benchmark.png` for presentation;
- each run's `metrics.json` for source/runtime provenance, dataset hashes and
  coverage, and per-seed behavior;
- overlap/duplicate/conflicting-label diagnostics and the training-derived
  references;
- each confusion matrix for majority-action collapse or asymmetric errors; and
- `manifest.json` for file integrity.

Do not select an algorithm only from mean accuracy. Inspect variability, per-action
support/recall, optimization histories, and whether every run used identical data
fingerprints.

### 7. Regenerate presentation artifacts

Per-run reports are deterministic functions of saved metrics:

```bash
gridworld-rl report \
  --run-dir artifacts/benchmarks/showcase/runs/seed-11
```

Regeneration refreshes the per-run manifest. It does not retrain a model or reread
the solution CSV. If that run is nested inside a benchmark directory, the changed
files no longer match the benchmark's top-level manifest. Preserve a published
benchmark unchanged or rerun it under a new name.

## Determinism controls

For each algorithm, the pipeline seeds:

- Python's `random`;
- NumPy;
- PyTorch CPU;
- all available CUDA devices;
- `DataLoader` shuffling; and
- Python/NumPy RNG state inside data-loading workers.

When deterministic mode is enabled, PyTorch deterministic algorithms are requested
with warnings for unavailable deterministic implementations. cuDNN benchmarking is
disabled and deterministic cuDNN behavior is requested.

The seed is reset before each algorithm, giving matching initialization and data
ordering where architecture and execution path are shared. This is useful for a
controlled comparison, but the algorithms can still consume random operations
differently if their implementations diverge.

### Recommended comparison device

Use `--device cpu` when portability and repeatability matter more than throughput:

```bash
gridworld-rl train \
  --config configs/showcase.json \
  --device cpu \
  --seed 11 \
  --run-name cpu-seed-11
```

For accelerator runs, record the GPU model, driver/runtime, PyTorch build, and
whether deterministic warnings appeared. Do not combine CPU, CUDA, and MPS runs in a
single aggregate without treating device as an experimental factor.

## Artifact provenance

### Per-run record

| Evidence | Location |
| --- | --- |
| Resolved parameters | `config.json` |
| Source commit and working-tree dirty state, when Git is available | `metrics.json` |
| Package, Python, library versions, and device | `metrics.json` |
| Dataset row counts, diagnostics, and SHA-256 hashes | `metrics.json` |
| Seed and optimizer-step counts | `metrics.json` |
| Model state and architecture metadata | `checkpoints/*.pt` |
| Challenge-order model outputs | `predictions.json` |
| Exact metrics and histories | `metrics.json` |
| Human-readable derived output | `summary.md`, `report.png`, `confusion_matrices.png` |
| Integrity hashes | `manifest.json` |

### Benchmark record

| Evidence | Location |
| --- | --- |
| Base experiment and seed schedule | `benchmark_config.json` |
| Aggregate exact values | `aggregate_metrics.json` |
| Human-readable aggregate | `benchmark.md`, `benchmark.png` |
| Complete per-seed evidence | `runs/seed-*/` |
| Top-level and nested integrity hashes | `manifest.json` |

SHA-256 detects accidental or unreported artifact changes. It is not a digital
signature and does not establish authorship.

Single-run writes use a hidden staging directory and publish only complete outputs.
Existing runs are refused unless `--overwrite` is explicit; replacement preserves
the previous directory if publication fails. Benchmarks never overwrite an existing
destination. These rules protect evidence from partial or accidental replacement,
but they do not replace external backups.

## Evaluation protocol

The current evaluator measures:

```text
action agreement = mean(argmax_a Q(s,a) == supplied_action)
```

Confusion matrices use true actions as rows and predicted actions as columns.
Per-action recall is zero when an action has no support, and support is always
reported so that zero can be interpreted correctly.

This is a **provided evaluation-split diagnostic**, not a held-out generalization
score. In the inspected snapshot, 5,491/5,544 rows (99.04%) exactly match a full
training transition, only 534 full transitions are unique, and 77/91 states have
conflicting action labels. The pipeline persists those facts beside every score.

It also reports:

- a training-majority action scored on the evaluation split;
- a per-state training mode scored on the evaluation split; and
- an evaluation-fitted per-state mode ceiling, clearly labelled as an oracle
  diagnostic rather than a predictive baseline.

### Prevent evaluation leakage

- Do not train on `eval_solution.csv`.
- Do not choose epochs, architecture, seed, epsilon, or CQL strength by repeatedly
  maximizing the same solution-set accuracy and then call that set “untouched.”
- If the solution influences a decision, label it as a validation set.
- For a confirmatory comparison, reserve a separate hidden test split or define an
  environment-based evaluation before tuning.

### Interpret claims narrowly

Supported wording:

> Across the declared seeds and fixed dataset snapshot, method A had higher mean
> agreement with the provided evaluation labels than method B; the split overlap,
> trivial references, variability, and per-action recall are reported in the linked
> artifact. This does not measure out-of-sample generalization.

Unsupported without additional evidence:

- “Method A learns the optimal policy.”
- “Method A generalizes to unseen transitions.”
- “Method A achieves higher online return.”
- “CQL is safe.”
- “Method A is statistically superior” based only on one run or overlapping error
  bars.

If statistical hypothesis testing is added, predeclare the test, use paired seeds
where appropriate, report effect sizes and uncertainty, and account for multiple
comparisons.

## Docker protocol

The CPU container offers a consistent user-space environment and runs as a non-root
user:

```bash
docker build -t gridworld-rl .

docker run --rm \
  -v "$PWD/Gridworld-10_Dataset:/app/Gridworld-10_Dataset:ro" \
  -v "$PWD/artifacts:/app/artifacts" \
  gridworld-rl benchmark \
    --config configs/default.json \
    --seeds 11 22 33 44 55 \
    --output-dir artifacts/benchmarks \
    --name docker-showcase
```

The container improves environment consistency, but image rebuilds can still change
if the base-image tag is updated upstream. Record the built image digest for
long-lived published evidence.

## Publication checklist

Before linking a result from GitHub or LinkedIn:

- [ ] The source commit is public and CI passes on that commit.
- [ ] The working tree used for training was clean, or deviations are documented.
- [ ] The exact config and seed schedule were fixed before evaluation.
- [ ] All compared runs have identical dataset SHA-256 fingerprints.
- [ ] Python, dependency, and device metadata are retained.
- [ ] Multiple independent seeds are reported with variability.
- [ ] Confusion matrices and per-action support/recall were reviewed.
- [ ] Train/evaluation overlap, duplicates, conflicting labels, and trivial
      references are disclosed.
- [ ] Action agreement is not described as environment return.
- [ ] Action agreement on this split is not described as out-of-sample
      generalization.
- [ ] Any tuning against evaluation labels is disclosed.
- [ ] No dataset CSV, secret, private path, or solution labels were published.
- [ ] Manifests match the files being shared.
- [ ] Historical legacy output is not presented as a result of the current pipeline.

## Known gaps

- Git metadata is unavailable in some installed/container contexts, and container
  image digests are not automatically embedded.
- The benchmark reports descriptive statistics; it does not run a hypothesis test
  or confidence-interval procedure.
- Aggregate logged state-action coverage is reported, but policy-selected
  out-of-distribution action rates and state-conditional coverage quality are not.
- The repository does not provide a validated online Gridworld environment.
- Cross-platform bitwise equality is not promised.

These gaps limit the claim boundary; they do not prevent the pipeline from producing
traceable, repeatable baseline experiments.
