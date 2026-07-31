# Technical Audit: Legacy Implementation

> [!NOTE]
> This document records the engineering review that motivated version 2.0. For the
> current design, see the [architecture guide](ARCHITECTURE.md). For a defensible
> experiment workflow, see the [reproducibility guide](REPRODUCIBILITY.md).

## Executive summary

The original repository demonstrated initiative and a useful problem choice, but the
single-file implementation was not ready to support reproducible technical claims.
Two target calculations were materially inconsistent with their named algorithms,
the validation routine only printed values, experiment randomness was uncontrolled,
and the saved figure could not be traced to a configuration or checkpoint.

The refactor separates data, learning targets, training, evaluation, reporting, and
command-line concerns. It adds an offline-specific CQL baseline and machine-readable
artifacts. The intent is not to retroactively validate the historical results; all
portfolio metrics should be regenerated with the new pipeline.

This audit concerns the legacy `main.py` state that preceded the modular
`gridworld_rl` package. It is a code and reproducibility review, not a security
certification or a review of the dataset's authorship.

## Severity definitions

| Severity | Meaning |
| --- | --- |
| High | Can change the algorithm being implemented or invalidate a core result |
| Medium | Can prevent reproduction, conceal invalid data, or materially weaken evaluation |
| Low | Primarily affects maintainability, portability, or presentation |

## Findings and resolutions

### 1. Double DQN selected actions with a target network — High

**Legacy behavior.** In the Double DQN branch, both the action-selection tensor and
the action-evaluation tensor came from copied target models. One target model selected
the maximizing next action and the other target model evaluated it. A randomly chosen
member of a two-model pair was then optimized.

**Why it matters.** Canonical Double DQN decouples selection and evaluation in a
specific way:

```text
a* = argmax_a Q_online(s', a)
y  = r + γ(1-done) Q_target(s', a*)
```

Selecting with a lagged target network removes the intended online-selection step.
The two-model random-update scheme also describes a different variant from the one
claimed in the README.

**Resolution.** The refactored Double DQN target selects with the online network and
evaluates the selected action with the target network. The operation is isolated and
unit-tested.

### 2. Expected SARSA did not compute an epsilon-greedy expectation — High

**Legacy behavior.** The code computed the mean of four next-action values, then
multiplied that mean by `0.25`. Because the mean already divides by four, this
produced one sixteenth of the sum. It also assigned no additional probability to the
greedy action and did not expose epsilon.

**Why it matters.** With `n` actions and epsilon `ε`, the target-policy probabilities
are:

```text
π(a|s') = ε/n + (1-ε)  if a is the selected greedy action
π(a|s') = ε/n          otherwise
```

and the bootstrap value is `Σ_a π(a|s')Q_target(s',a)`. The legacy expression was
neither this expectation nor a correctly scaled uniform expectation.

**Resolution.** The expectation is computed from explicit epsilon-greedy
probabilities, terminal masking is applied, epsilon is configurable, and boundary
cases are tested.

### 3. DQN correctness was implicit and untested — Medium

**Legacy behavior.** The DQN branch did use the recognizable expression
`r + γ(1-done) max Q_target(s',a)`. This audit did not find a fundamental algebraic
error in that branch. However, the equation was embedded inside a multi-algorithm
training function, coupled to global data, and had no test for terminal masking or
target-network use.

**Why it matters.** A correct-looking expression is easy to regress when it shares
control flow with unrelated algorithms. Terminal transitions must not bootstrap, and
the maximization must use the target network.

**Resolution.** The refactor makes the DQN Bellman target an explicit, testable
operation:

```text
y = r + γ(1-done) max_a Q_target(s',a)
```

### 4. No method addressed offline distribution shift — Medium

**Legacy behavior.** All methods could assign high values to actions that were rare
or absent for a state in the fixed dataset. The code included no offline-RL
regularizer or diagnostic for this failure mode.

**Why it matters.** Bootstrapping and maximization can favor out-of-distribution
actions whose estimates have not been corrected by data.

**Resolution.** The project adds a discrete CQL baseline. Its training objective is
the Huber TD loss plus:

```text
α [logsumexp_a Q(s,a) - Q(s,a_data)]
```

This penalizes relative value assigned to actions outside the observed dataset
action. The run record also summarizes logged state-action coverage and action
counts. These additions reduce a known incentive for overestimation and expose broad
coverage gaps, but they do not measure policy-selected out-of-distribution rates or
guarantee conservative behavior for every state.

### 5. The dataset “cleaning” function did not validate or clean — High

**Legacy behavior.** `check_and_clean_dataset()` printed null counts, unique values,
and dtypes. It did not reject missing columns, nulls, invalid state/action ranges,
non-finite rewards, malformed booleans, or mismatched challenge/solution rows.
Execution continued after suspicious input.

**Why it matters.** Invalid action indices can fail deep inside tensor operations,
while row misalignment can yield a plausible but meaningless accuracy.

**Resolution.** Loading now fails early with contextual errors. It validates required
columns, conversions, finite values, state and action ranges, and row-level alignment
between evaluation challenge and solution data. A missing dataset produces an
actionable message rather than a raw import-time traceback.

### 6. Randomness was uncontrolled — Medium

**Legacy behavior.** Network initialization and NumPy batch sampling used default
random state. The repository recorded no seed and offered no deterministic execution
controls.

**Why it matters.** Two executions could produce different training order, weights,
predictions, and figures without any visible configuration change.

**Resolution.** A declared seed is applied to supported Python, NumPy, PyTorch, and
data-loading sources. The resolved seed is saved with the run. Backend-level
determinism limitations are documented rather than hidden.

### 7. Training was a monolithic, fixed, partial pass — Medium

**Legacy behavior.** `training()` hard-coded the learning rate, discount, batch size,
and synchronization cadence. It sampled independent batches for
`floor(dataset_size/batch_size)` updates rather than iterating a shuffled epoch, so
some rows could repeat and others could be skipped. There was no epoch control.

**Why it matters.** The effective training distribution was opaque, important
hyperparameters could not be reproduced from a command, and the data pipeline could
not be tested independently.

**Resolution.** The refactor uses `Dataset`/`DataLoader`, configurable epochs and
optimization settings, explicit target synchronization, Huber TD loss, and gradient
clipping.

### 8. Data was loaded at module import time — Medium

**Legacy behavior.** Importing `main.py` immediately read six relative CSV paths.
Any missing file prevented reuse of model or metric code, and behavior depended on
the caller's working directory.

**Why it matters.** Import side effects make unit tests, library reuse, IDE tooling,
and deployment brittle.

**Resolution.** Data is loaded only through explicit APIs and CLI execution. Paths
come from configuration and errors identify the missing resource and expected layout.

### 9. Evaluation mixed distinct claims — High

**Legacy behavior.** The project reported row-wise action agreement as “accuracy”
beside a hand-written cumulative-return simulation. That simulation tried actions in
descending Q order until one was accepted, rather than applying exactly one policy
action per step. Horizontal moves were checked only against the flattened `0..99`
range, allowing row-boundary wraparound. There was no episode step cap, and results
were not stored with the trained models.

**Why it matters.** A policy given several ranked attempts per state is not being
evaluated under the stated four-action MDP. Action-label agreement and environment
return answer different questions and should not be presented as interchangeable
evidence of policy quality.

**Resolution.** The reproducible report focuses on clearly named provided-split
action agreement, a confusion matrix, and per-action recall. It also exposes exact
train/evaluation overlap, duplicates, conflicting state labels, and trivial
references. The inspected split has 99.04% row overlap, so it is explicitly not
presented as out-of-sample evidence. Any future return evaluation should use a
separately tested environment with explicit boundary transitions, one action per
step, termination rules, and declared seeds.

### 10. Evaluation labels were available without protocol safeguards — Medium

**Legacy behavior.** The solution CSV was loaded alongside training data and no
protocol prevented repeated tuning against it.

**Why it matters.** Even when solution labels are not directly optimized, repeated
hyperparameter choices based on the same solution set turn it into a validation set
and bias reported performance.

**Resolution.** Training and scoring paths are separated conceptually and in code.
The documentation requires disclosure of tuning and recommends an additional held-out
set for confirmatory comparisons.

### 11. Artifacts were not sufficient to reproduce a result — Medium

**Legacy behavior.** The script saved one PNG to a fixed filename. It did not save
weights, a seed, hyperparameters, per-model metrics, dependency versions, or a
run-specific manifest. A later run could silently overwrite the plot.

**Why it matters.** The figure could not be traced back to the exact model or
configuration that produced it.

**Resolution.** Runs save a checkpoint, resolved JSON configuration, source and
runtime metadata, dataset hashes/diagnostics, predictions, metrics, and report inputs
in a dedicated directory. Integrity manifests cover generated files, reports can be
regenerated from metrics, and repeated-seed runs can be aggregated without manually
copying values. Single runs publish from a staging directory and refuse replacement
without explicit opt-in; benchmark destinations are immutable.

### 12. The README contained inaccurate or unsupported statements — Medium

**Legacy behavior.**

- DQN was called on-policy even though Q-learning is off-policy.
- Double DQN was described as two independent networks without documenting the
  noncanonical update actually used.
- Expected SARSA was described as a probability-weighted expectation despite the
  scaling error.
- A 42% result and qualitative stability claims were presented without a seed,
  checkpoint, configuration, or uncertainty across runs.
- The citation named an unrelated “MovieLens 100k” project.
- The software license statement conflicted with the requested distribution model
  and did not distinguish source code from third-party data.

**Resolution.** The portfolio README defines targets, labels evaluation semantics,
states limitations, confines its diagnostic numbers to a provenance-labelled run,
supplies correct citation metadata, and separates the MIT software license from
dataset rights.

### 13. Project structure prevented focused testing — Medium

**Legacy behavior.** Data access, neural networks, three target rules, optimization,
evaluation, plotting, and entry-point logic lived in one `main.py`. Most functions
depended on globals.

**Why it matters.** A reviewer could not test an algorithm target without satisfying
filesystem and global-state assumptions. Changes in one concern could break another.

**Resolution.** The `src/` package exposes separate modules with narrow
responsibilities. Unit tests cover targets, data validation, and metrics; integration
tests cover training, artifacts, checkpoint loading, reports, CLI behavior,
deterministic reruns, and multi-seed aggregation.

### 14. Deployment and dependency controls were weak — Medium

**Legacy behavior.** Dependencies were unpinned. The container installed build tools
and ran the application as root. Its default command invoked the legacy script, and
the dataset/output contract was undocumented.

**Why it matters.** Builds could change over time, root execution enlarged the
container's impact, and users could not predict where outputs would go.

**Resolution.** Runtime and development dependencies are pinned; CI checks Ruff,
Python 3.11/3.12 tests with a coverage floor, builds a wheel, and smoke-tests the
installed distribution. The Docker image uses an unprivileged user, and
dataset/artifact mounts are documented.

## What the refactor does not prove

The engineering corrections make experiments inspectable; they do not establish that:

- one algorithm is statistically superior;
- action labels are uniquely optimal;
- the evaluation set was never used during informal tuning;
- the dataset covers actions selected by every learned policy; or
- results transfer beyond this dataset and state representation.

Those claims require a predeclared evaluation protocol, multiple seeds, uncertainty
reporting, and—where return is claimed—a validated environment.

## Portfolio release checklist

Before linking the repository publicly:

- run CI from a clean clone;
- obtain the dataset through its authorized distribution channel;
- train each reported configuration from scratch;
- retain the commit identifier, config, metrics, and checkpoint for every run;
- generate reports from saved artifacts;
- run the declared multi-seed benchmark and disclose any hyperparameter selection;
- inspect confusion matrices for majority-action collapse;
- confirm that no dataset CSV or sensitive local path is committed; and
- treat the historical PNG as provenance, not a current benchmark.
