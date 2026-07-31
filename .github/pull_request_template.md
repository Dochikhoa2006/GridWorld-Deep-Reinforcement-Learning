## Summary

<!-- Explain what changed and why. Focus on the user, research, or maintenance problem being solved. -->

## Related issue

<!-- Use "Closes #123" when this pull request fully resolves an issue. -->

Closes #

## Type of change

- [ ] Bug fix
- [ ] New feature or algorithm
- [ ] Refactor with no intended behavior change
- [ ] Tests or continuous integration
- [ ] Documentation
- [ ] Dependency, packaging, or container update

## Technical approach

<!-- Describe the important implementation choices and tradeoffs. For an algorithm or loss change, include the equation and make online/target network roles explicit. -->

## Verification

<!-- List the exact commands run and summarize their results. Add a small reproducible example for behavior changes. -->

```text
python -m pytest -q
```

## Reproducibility and compatibility

<!-- Note effects on random seeds, defaults, CLI options, persisted configurations, artifact schemas, checkpoints, and prior results. Write "No impact" only after checking each relevant area. -->

## Evidence

<!-- Include concise logs, metrics, or screenshots when useful. State the configuration, seed, and data split behind benchmark results. Do not include private datasets or sensitive local information. -->

## Checklist

- [ ] I read and followed `CONTRIBUTING.md`.
- [ ] I kept the change focused and documented non-obvious decisions.
- [ ] I added or updated tests for changed behavior, or explained why tests are not applicable.
- [ ] I updated user-facing documentation and `CHANGELOG.md`, or confirmed that no update is needed.
- [ ] I did not commit credentials, third-party datasets, generated reports, checkpoints, or other unintended artifacts.
- [ ] I reviewed the diff for absolute local paths and sensitive information.
- [ ] I preserved evaluation-data isolation and disclosed any benchmark tuning.
- [ ] I ran the relevant local checks and expect the continuous-integration checks to pass.
