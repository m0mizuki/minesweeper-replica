# Minesweeper CSP

Minesweeper を planted CSP として扱い、Belief Propagation と MCMC の比較から
solution-space の統計構造を調べる研究用コードです。研究方針と数理モデルは
[`docs/README_minesweeper_research.md`](docs/README_minesweeper_research.md) を参照してください。

現在は Milestone 1 を実装しています。

- 各セルを独立な Bernoulli(`rho`) で生成（総地雷数は固定しない）
- ground truth と独立に観測プロトコルを適用
- Moore 近傍（最大8セル）の clue を生成
- 未観測セルを変数、観測済み安全セルを制約とする factor graph を構築
- planted configuration の CSP 整合性を検査

観測プロトコルは次の2種類です。

- `all_safe`: 全ての安全セルを clue として観測
- `bernoulli_safe`: 各安全セルを `observation_rate` の確率で独立に観測

いずれも地雷セルは clue として観測しません。

```python
from minesweeper_csp import (
    assert_csp_consistent,
    build_factor_graph,
    generate_clues,
    generate_ground_truth,
    generate_observation_mask,
)

instance = generate_ground_truth(Lx=6, Ly=8, rho=0.2, seed=123)
observed = generate_observation_mask(
    instance,
    protocol="bernoulli_safe",
    observation_rate=0.5,
    seed=456,
)
clues = generate_clues(instance.ground_truth, observed)
graph = build_factor_graph(observed, clues)
assert_csp_consistent(graph, instance.ground_truth)
```

テストは、パッケージを editable install した後に標準ライブラリだけで実行できます。

```powershell
python -m pip install -e .
python -m unittest discover -s tests -v
```

