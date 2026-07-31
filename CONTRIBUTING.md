# Contributing

Thank you for helping improve Gridworld-10 Offline Reinforcement Learning. Changes
that make algorithm behavior easier to verify, experiments easier to reproduce, or
limitations clearer are especially welcome.

Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md). Report
security concerns through the private process in [SECURITY.md](SECURITY.md), not a
public issue.

## Before opening a change

- Search existing issues and pull requests.
- Open an issue before a large algorithm, artifact-format, or CLI change.
- Do not commit the third-party dataset, model checkpoints, generated reports, or
  credentials.
- Keep benchmark claims separate from engineering changes and include the exact
  configuration and seeds behind any result.

By submitting a contribution, you agree that it may be distributed under the
repository's MIT License and that you have the right to submit it.

## Development setup

Use Python 3.11:

```bash
make install-dev
source .venv/bin/activate
make check
```

Without Make, create a virtual environment and install
`python -m pip install -e '.[dev]'`, then run the Ruff, pytest, build, and CLI smoke
commands defined in the `Makefile`.

Obtain the Gridworld-10 CSV files from their authorized source and place them under
`Gridworld-10_Dataset/` only when an integration test or local experiment needs
them. Unit tests should use small synthetic fixtures and must not depend on a
developer's private dataset checkout.

## Making a pull request

1. Create a focused branch from the latest default branch.
2. Make the smallest coherent change.
3. Add or update tests for changed behavior.
4. Update the README, relevant file under `docs/`, or changelog when user-facing
   behavior changes.
5. Run the checks below.
6. Review the staged diff for data, artifacts, absolute paths, and secrets.
7. Open a pull request that explains the motivation, implementation, test evidence,
   and compatibility impact.

```bash
git switch -c type/short-description
make check
git diff --check
git status --short
```

## Engineering expectations

### Algorithm changes

For a Bellman target or loss change:

- write the equation in the pull-request description;
- include a hand-checkable tensor example in a unit test;
- test terminal masking;
- make online-network and target-network roles explicit;
- test CPU behavior before adding accelerator-specific logic; and
- explain whether the method is online, off-policy, or adapted to fixed offline data.

Do not use the evaluation solution as optimization data. If it influences
hyperparameter selection, disclose that and avoid describing the same set as an
untouched test set.

### Data changes

Validation errors should identify the file and offending field without dumping the
full dataset. New parsing behavior needs tests for accepted input and failure cases,
including range errors, non-finite values, and challenge/solution misalignment.

### Reproducibility

Any new source of randomness must accept or derive from the configured seed. New
experiment controls belong in the resolved configuration and should be persisted
with the run.

### Documentation

Use plain language, runnable relative-path commands, and no untraceable performance
claims. Distinguish action agreement from environment return. Cite external methods
or datasets at their primary source when adding research discussion.

## Commit and review guidance

Clear, imperative commit subjects are preferred, for example:

```text
fix: use online selection in Double DQN target
test: reject misaligned evaluation rows
docs: clarify dataset licensing
```

A reviewer may request changes when a contribution:

- changes a target without a numerical test;
- makes results depend on import order or ambient random state;
- silently changes the artifact schema;
- includes generated/binary output without justification; or
- reports a benchmark without sufficient provenance.

All pull requests must pass GitHub Actions before merge.
