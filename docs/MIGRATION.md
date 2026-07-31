# Migration Guide

> [!NOTE]
> This guide is for maintainers upgrading the original single-file project. New
> users should start with the repository [README](../README.md).

This guide moves a clone of the legacy `main.py` project to the modular
`gridworld_rl` workflow without treating historical outputs as comparable results.

## Behavioral changes

| Legacy workflow | Refactored workflow |
| --- | --- |
| `python main.py` | `python -m gridworld_rl train --config configs/default.json` |
| Data loaded on import | Data paths resolved and validated at command execution |
| Hard-coded hyperparameters | Versioned JSON configuration |
| One implicit sampling pass | Configurable `DataLoader` epochs |
| DQN, noncanonical DDQN, mis-scaled Expected SARSA | Tested DQN, Double DQN, epsilon-greedy Expected SARSA, and CQL |
| Fixed PNG output | Per-run checkpoint, resolved config, provenance, metrics, report, and manifest |
| Manual single-run comparison | Repeated-seed benchmark command and aggregate report |
| Ad hoc console inspection | Exceptions with dataset/file/field context |

Legacy model objects and the historical PNG are not compatible experiment artifacts.
Retrain with the new code; do not relabel the old figure as a refactored result.

## 1. Preserve the current repository state

Start in the repository root. If `git status` shows work you want to retain, commit or
stash it before switching branches.

```bash
git status --short
git switch main
git pull --ff-only
git tag legacy-main-before-offline-rl-refactor
git switch -c refactor/offline-rl-pipeline
```

The local tag provides a convenient reference to the pre-migration code. Push it only
if you intentionally want that tag on the remote:

```bash
git push origin legacy-main-before-offline-rl-refactor
```

## 2. Keep the dataset local

### Dataset history blocker

The current working tree ignores `Gridworld-10_Dataset/`, but the legacy repository
history contains all six CSV files. Deleting or ignoring a path in a later commit
does not remove its earlier blobs, and rewriting history cannot retract existing
clones or caches.

Before presenting the repository as a clean software-only showcase, choose one path:

1. **Recommended: publish a new clean-history showcase repository.** Export only the
   reviewed working tree, exclude `.git`, datasets, generated artifacts, and local
   environments, initialize a new repository, inspect its first commit, then push to
   a new remote.
2. **Rewrite the existing remote only with explicit maintainer approval.** Make a
   recoverable mirror backup, use a purpose-built history-filtering tool to remove
   `Gridworld-10_Dataset/` from every ref, verify all branches/tags and blob objects,
   coordinate the required force-push, and notify every collaborator to reclone.
3. **Retain history only if redistribution is authorized.** Record the dataset
   owner's permission and keep the software and dataset licenses visibly separate.

Do not run a history rewrite as a routine migration command. It changes commit IDs,
branches, tags, open pull requests, and collaborators' clones. The safest portfolio
path is a new clean repository with a curated initial commit.

Audit both the current tree and all reachable history before publishing:

```bash
git ls-files 'Gridworld-10_Dataset/*'
git log --all --name-only -- 'Gridworld-10_Dataset'
git rev-list --all --objects | rg 'Gridworld-10_Dataset/'
```

Continue with the local dataset layout only after deciding how the historical blobs
will be handled.

The refactor expects this layout:

```text
Gridworld-10_Dataset/
├── train.csv
├── eval_challenge.csv
└── eval_solution.csv
```

The dataset's sample map files may remain beside these CSVs, but the current pipeline
does not consume them.

Do not use `git add -f` on the dataset. The repository's MIT license does not cover
third-party CSV files. Verify ignore behavior before committing:

```bash
git check-ignore -v Gridworld-10_Dataset/train.csv
git status --short
```

If the first command prints no ignore rule, stop and correct `.gitignore` before
staging files.

## 3. Build a clean environment

Recreate the virtual environment so legacy, unpinned packages do not mask dependency
problems:

```bash
python3.11 -m venv .venv-refactor
source .venv-refactor/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv-refactor
.venv-refactor\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

## 4. Validate the migration

Run fast checks first, then one complete configured experiment:

```bash
VENV=.venv-refactor make check
python -m gridworld_rl --help
python -m gridworld_rl validate --config configs/default.json
python -m gridworld_rl train --config configs/default.json
```

If the train command cannot find the dataset, confirm the configured data directory
and the exact case of each filename. If an accelerator produces nondeterministic
behavior, rerun on CPU and record that device choice in the config.

The new runner refuses an existing output directory by default and publishes through
a hidden staging directory. Choose a unique `output.run_name` for evidence. Use
`--overwrite` only for an intentional atomic replacement; benchmark destinations
cannot be overwritten.

## 5. Review generated artifacts

Before using a result, confirm that its run directory contains:

- a resolved JSON configuration with algorithm, seed, data paths, and optimization
  settings;
- source revision/working-tree metadata when Git is available, package/runtime
  versions, dataset fingerprints, and aggregate coverage diagnostics;
- a model checkpoint;
- machine-readable metrics;
- a confusion matrix and per-action recall in the report output; and
- no embedded evaluation-solution labels or dataset copy.

Action agreement is not environment return. Do not compare the new metric with the
legacy hand-written rollout plot as if they were the same measurement.

## 6. Stage and commit the refactor

Review both unstaged and staged content, particularly data and artifact paths:

```bash
git status --short
git diff --check
git diff
git add --all
git status --short
git diff --cached --check
git diff --cached --stat
git diff --cached
git commit -m "refactor: add reproducible offline RL pipeline"
git push -u origin refactor/offline-rl-pipeline
```

Open a pull request into `main`. In the pull-request description, include:

- the target-equation corrections;
- the test command and result;
- the dataset provenance and license caveat;
- the configuration and seeds used for any reported metric; and
- a statement that historical results were not carried forward.

Merge only after CI passes. Prefer the hosting platform's protected merge controls
over a direct push to `main`.

## 7. Update an existing clone after merge

Once the refactor is merged:

```bash
git switch main
git pull --ff-only
python -m pip install -e '.[dev]'
python -m pytest
```

Keep old artifacts outside `artifacts/latest` or rename their directory. They lack the
new resolved-configuration and metrics contract and should not be consumed by the
report command. The default run name will be refused if that directory already
exists.

## 8. Tag version 2.0.0

After the pull request is merged and CI passes on `main`, create the release tag from
the merged commit. First confirm that the name is unused:

```bash
git switch main
git pull --ff-only
git tag --list v2.0.0
```

If the last command produces no output, create and publish the annotated tag:

```bash
git tag -a v2.0.0 -m "Release 2.0.0"
git push origin v2.0.0
```

Do not move or replace a published version tag. If `v2.0.0` already exists but points
to the wrong commit, stop and decide on a new patch version with the maintainer.

## Configuration mapping

| Legacy constant or behavior | New setting |
| --- | --- |
| `a = 0.005` | `training.learning_rate` |
| `batch_size = 128` | `training.batch_size` |
| `gamma = 0.99` | `training.gamma` |
| one implicit pass | `training.epochs` |
| sync every 100 loop iterations | `training.target_update_interval` |
| implicit uniform Expected SARSA logic | `training.epsilon` |
| no conservative term | `training.cql_alpha` |
| current hardware chosen implicitly | `training.device` |
| uncontrolled random state | `training.seed` |
| fixed output silently replaced | `output.run_name` plus opt-in `output.overwrite` |

Use the keys and defaults in `configs/default.json`; do not assume the values above
remain the project defaults.

## Rollback

The migration branch leaves `main` and the legacy tag intact. To inspect the old code
without changing the refactor branch:

```bash
git show legacy-main-before-offline-rl-refactor:main.py
```

To abandon an unmerged migration branch, first preserve any work you need, switch
away from it, then delete only that named branch:

```bash
git switch main
git branch -d refactor/offline-rl-pipeline
```

Git will refuse `-d` if the branch contains unmerged commits. Review those commits
instead of forcing deletion.
