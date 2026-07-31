# Five-Seed, 50-Epoch Release-Candidate Benchmark

[README](../../README.md) ·
[Reproducibility](../REPRODUCIBILITY.md) ·
[Architecture](../ARCHITECTURE.md)

## Status

This benchmark was generated locally on 31 July 2026 from clean source commit
`7ba76df867462235b1efa6d22578438914e1b4b9` on branch
`refactor/linkedin-showcase`. Every run recorded `git_dirty: false`; the aggregate
manifest verified all 59 retained files after generation.

The results are release-candidate evidence, not a claim that the branch has already
passed GitHub-hosted CI or been merged into `main`.

## Protocol

```bash
gridworld-rl benchmark \
  --data-dir Gridworld-10_Dataset \
  --epochs 50 \
  --device cpu \
  --cql-alpha 1.0 \
  --seeds 11 22 33 44 55 \
  --output-dir artifacts/benchmarks \
  --name linkedin-50-epoch-five-seed
```

| Dimension | Value |
| --- | --- |
| Seeds | 11, 22, 33, 44, 55 |
| Training rows | 22,172 |
| Evaluation rows | 5,544 |
| Exact train/evaluation transition overlap | 5,491 / 5,544 rows (99.04%) |
| Unique full evaluation transitions | 534 (5,010 duplicate rows) |
| Evaluation states with conflicting actions | 77 / 91 |
| Logged state-action coverage | 346 / 400 pairs (86.5%) |
| Algorithms | DQN, Double DQN, Expected SARSA, CQL |
| Epochs | 50 |
| Network | One-hot state input; hidden widths 128 and 64; four outputs |
| Optimizer | Adam, learning rate 0.001 |
| Batch size | 128 |
| Discount | 0.99 |
| Expected SARSA epsilon | 0.1 |
| CQL alpha | 1.0 |
| Target synchronization | Every 250 optimizer steps |
| Gradient clipping | Maximum norm 10.0 |
| Runtime | Python 3.12.13; PyTorch 2.11.0; NumPy 2.4.4; pandas 3.0.2; Matplotlib 3.10.8 |
| Device | CPU |

Dataset SHA-256 fingerprints:

```text
train           58a517870fe94fdec3e5bc53790a436f68fc5bbabf303adb031adbe5a1aba686
eval_challenge  dc82210503d3c0aa084c6b194b0b372f51bdbdd10121c36fd53a7f51806ef2dc
eval_solution   a49bf745e7ac3d9db082e21010e0bc36e530c49b0c7255ad13c4273b234d19f1
```

## Aggregate action agreement

Values are mean ± sample standard deviation across the five declared seeds.

| Algorithm | Mean ± std | Minimum | Maximum |
| --- | ---: | ---: | ---: |
| CQL | 41.38% ± 0.01% | 41.38% | 41.40% |
| Expected SARSA | 33.79% ± 5.02% | 29.98% | 42.30% |
| Double DQN | 32.97% ± 5.29% | 28.14% | 42.05% |
| DQN | 31.37% ± 2.34% | 29.64% | 35.37% |
| Training-majority reference (always action 1) | 39.23% | 39.23% | 39.23% |
| Training per-state-mode reference | 41.63% | 41.63% | 41.63% |
| Evaluation-fitted state-mode ceiling | 44.66% | 44.66% | 44.66% |

The first two references are selected only from training labels and then scored on
the evaluation split. The ceiling chooses the most frequent evaluation label for
each evaluation state, so it is an oracle diagnostic rather than a fair predictive
baseline.

![Provided evaluation-split action agreement across five seeded runs](five-seed-action-agreement.png)

## Per-action diagnostic

Mean recall ± sample standard deviation across seeds was:

| Algorithm | Action 0 | Action 1 | Action 2 | Action 3 |
| --- | ---: | ---: | ---: | ---: |
| CQL | 0.57% ± 0.00% | 17.31% ± 0.28% | 0.19% ± 0.07% | 98.72% ± 0.28% |
| Expected SARSA | 4.27% ± 5.34% | 44.00% ± 26.50% | 0.82% ± 1.16% | 45.46% ± 19.28% |
| Double DQN | 5.79% ± 8.79% | 31.41% ± 33.80% | 1.55% ± 1.53% | 56.41% ± 23.41% |
| DQN | 3.55% ± 4.63% | 27.74% ± 19.63% | 3.19% ± 4.57% | 56.15% ± 15.43% |

The overall CQL score is exceptionally consistent, but the class-level evidence
shows why that should not be described as broad policy stability: its learned
greedy policy almost always recovers action 3 and nearly never recovers actions 0
or 2. The other methods vary substantially across seeds and also perform poorly on
actions 0 and 2.

The training data is imbalanced toward actions 1 and 3. More fundamentally, this
evaluation split cannot estimate out-of-sample generalization: 99.04% of its rows
exactly reproduce a transition present in training, it contains only 534 unique
full transitions across 5,544 rows, and 77 of 91 states carry conflicting action
labels. The 44.66% evaluation-fitted state-mode ceiling quantifies the ambiguity
for any deterministic policy that receives only the state.

## Interpretation

- CQL had the highest mean agreement and the smallest observed standard deviation,
  but its per-action recall reveals a strongly collapsed action distribution.
- No learned method exceeded the 41.63% training per-state-mode reference on mean
  agreement.
- The small evaluation ceiling and contradictory labels limit the attainable score
  for deterministic state-only policies.
- These five seeds describe this declared protocol. They do not establish
  statistical superiority, safety, optimality, or online-return performance.

## Claim boundary

This benchmark supports the engineering statement that the repository reproducibly
runs four offline value-learning baselines across declared seeds and retains
configuration, source, data, runtime, class-level, and integrity evidence.

It does not support claims about online cumulative return, out-of-sample policy
generalization, an optimal or safe policy, statistical superiority, or transfer to
another dataset. See the [reproducibility protocol](../REPRODUCIBILITY.md) before
publishing another comparison.
