# Minesweeper CSP

Minesweeper を planted CSP として扱い、Belief Propagation と MCMC の比較から
solution-space の統計構造を調べる研究用コードです。研究方針と数理モデルは
[`docs/README_minesweeper_research.md`](docs/README_minesweeper_research.md) を参照してください。

現在は Milestone 5 まで実装しています。

- 各セルを独立な Bernoulli(`rho`) で生成（総地雷数は固定しない）
- ground truth と独立に観測プロトコルを適用
- Moore 近傍（最大8セル）の clue を生成
- 未観測セルを変数、観測済み安全セルを制約とする factor graph を構築
- planted configuration の CSP 整合性を検査
- 小規模CSPのfeasible statesを完全列挙
- Bernoulli事前分布を含む厳密な分配関数・周辺確率を計算
- planted overlapと独立2レプリカ間の厳密な overlap 分布を計算
- 正規化・log-domain更新・dampingを備えたBelief Propagation
- BPの収束状態、iteration数、message residualを記録
- 複数初期値によるfixed-point依存性の比較
- BP marginalとexact marginalの誤差評価
- factor graph上の局所BFS blockを用いるblocked Gibbs sampler
- burn-in、thinning、chain長、変更率、自己相関時間、ESSを記録
- 複数chainのR-hat、chain間marginal差、replica overlap trace
- MCMC marginalとexact marginalの誤差評価
- `rho` sweepと複数disorder sampleの再開可能な実験実行
- runごとの厳密なconfig・seed・git commit・timestamp記録
- 平均、標準偏差、標準誤差、quantile、sample間揺らぎの集約
- BP/MCMC/Exact、overlap、収束性、自己相関時間のSVG可視化

観測プロトコルは次の2種類です。

- `all_safe`: 全ての安全セルを clue として観測
- `bernoulli_safe`: 各安全セルを `observation_rate` の確率で独立に観測

いずれも地雷セルは clue として観測しません。

```python
from minesweeper_csp import (
    BPConfig,
    assert_csp_consistent,
    build_factor_graph,
    generate_clues,
    generate_ground_truth,
    generate_observation_mask,
    MCMCConfig,
    run_blocked_gibbs,
    run_bp,
    solve_exact,
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

exact = solve_exact(
    graph,
    rho=instance.rho,
    planted_ground_truth=instance.ground_truth,
    max_variables=20,
)
print(exact.number_of_feasible_solutions)
print(exact.marginals)
print(exact.replica_overlap.mean)

bp = run_bp(
    graph,
    rho=instance.rho,
    config=BPConfig(damping=0.2, initialization="random", seed=789),
)
print(bp.status, bp.iterations, bp.marginals)

mcmc = run_blocked_gibbs(
    graph,
    rho=instance.rho,
    initial_assignment=instance.ground_truth,
    config=MCMCConfig(block_size=4, burn_in=1000, samples=5000, seed=1011),
    planted_ground_truth=instance.ground_truth,
)
print(mcmc.marginals, mcmc.mine_density_effective_sample_size)
```

`solve_exact` は計算量・メモリ量とも変数数に対して指数的です。誤って大規模系を
全列挙しないよう、デフォルトでは未知変数が20個を超えると停止します。
replica overlap 分布は全solution pairの二重ループではなく、XOR自己相関の
Walsh–Hadamard変換で厳密に求めます。

BPのmessageは各更新で正規化されます。積によるunderflowを避けるため更新は
log-domainで行い、hard constraintやpriorによって両状態の質量がゼロになる場合は
一様分布で隠さず`infeasible`として返します。`run_bp_multiple`に複数の
`BPConfig`を渡すことで、初期値ごとの収束とfixed pointの差を比較できます。

MCMCはsingle-site Metropolisではなく、factor graph上で近い変数をBFSで集め、
block内の全状態から外部を固定した厳密な条件付き分布を作るblocked Gibbsです。
`acceptance_rate`はGibbs更新には該当しないため`None`とし、実際に状態が変わった
更新の割合を`changed_update_rate`へ保存します。定数traceのESSだけでは固定変数と
stuck chainを区別できないため、変更率と複数chainのR-hatを併用してください。

## Phase scan

実験入力には標準ライブラリで厳密に読み込めるJSONを使用します。短い動作確認用と
小規模pilot scan用の設定を用意しています。

```powershell
python scripts/run_sweep.py configs/sweeps/rho_smoke.json --output results/rho_smoke
python scripts/run_sweep.py configs/sweeps/rho_small.json --output results/rho_small
```

完了済みの`run.json`はデフォルトで再利用されるため、中断したsweepを同じコマンドで
再開できます。異なるconfigを同じ出力先へ混在させることは、`--overwrite`の有無に
かかわらず禁止されます。既存runを同一configで明示的に再計算する場合だけ
`--overwrite`を指定してください。

各runには次を保存します。

- `run.json`: 条件、全seed、solver設定、収束・mixing診断、比較指標、metadata
- `arrays.npz`: 盤面、clue、marginal、message residual、MCMC samples、全trace

sweep全体には`aggregate.json`と`phase_scan.svg`を生成します。保存済みrunだけを
再集約する場合は次を実行します。

```powershell
python scripts/aggregate.py results/rho_small
```

テストは、パッケージを editable install した後に標準ライブラリだけで実行できます。

```powershell
python -m pip install -e .
python -m unittest discover -s tests -v
```
