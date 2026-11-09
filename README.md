# Minesweeper CSP

Minesweeper を planted CSP として扱い、Belief Propagation と MCMC の比較から
solution-space の統計構造を調べる研究用コードです。研究方針と数理モデルは
[`docs/README_minesweeper_research.md`](docs/README_minesweeper_research.md) を参照してください。

現在は Milestone 7 まで実装しています。

資料生成用のローカル作業フォルダ（`.codex-*`、`.chart-data-*`）は
`build/presentations/` にまとめています。既存の生成スクリプトは
`build/presentations/.codex-pptx-build/build_deck.mjs` にあり、最終成果物は
引き続き `outputs/` に出力します。`build/` 以下は Git 管理対象外です。

- 各セルを独立な Bernoulli(`rho`) で生成（総地雷数は固定しない）
- ground truth と独立に観測プロトコルを適用
- Moore 近傍（最大8セル）の clue を生成
- 未観測セルを変数、観測済み安全セルを制約とする factor graph を構築
- planted configuration の CSP 整合性を検査
- 小規模CSPのfeasible statesを完全列挙
- Bernoulli(`rho`) 事前分布で重み付けした feasible states の厳密な分配関数・周辺確率を計算
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
- 複数診断を組み合わせた候補密度領域の優先順位付け
- 候補領域の高密度sweepとsystem-size dependence
- disorder平均した完全なMCMC/Exact `P(q)` と分布幅・entropy・mode数
- BP fixed-point数、初期値spread、MCMC mixingの有限サイズ比較
- BP fixed pointでのBethe分配関数・free entropy・entropy
- factor graphのcycle rankと長さ4のshort-loop数
- undamped BP mapの数値Jacobianによる有限instance局所安定性
- Exactに対するBethe誤差とsystem-size dependenceの可視化

観測プロトコルは次の2種類です。

- `all_safe`: 全ての安全セルを clue として観測
- `bernoulli_safe`: 各安全セルを `observation_rate` の確率で独立に観測

いずれも地雷セルは clue として観測しません。

推論の target distribution は、Bernoulli(`rho`) 事前分布を clue 制約で条件付けした分布です。
`P(x | c) = Z_rho^-1 product_i rho^x_i (1-rho)^(1-x_i) product_a delta(sum_i x_i, c_a)`
を Exact・BP・MCMC 共通で使用します。制約のない変数の地雷周辺確率は `rho` です。
BP の既定初期値は変数→因子が `[1-rho, rho]`、因子→変数が `[0.5, 0.5]` です。
`uniform` と `random` は初期値依存性を調べるために明示的に選択できます。
厳密分配関数 `Z_rho` は事前分布の下で制約が満たされる確率で、解数とは区別します。
ρ=0,1 で制約の事前確率がゼロなら Exact は質量ゼロ、BP は `infeasible` を返し、
MCMC は事前確率ゼロの初期配置を拒否します。

添付の「スライド草稿.md」に合わせ、overlap は未開示変数上の一致率
`q = mean(x_i == x'_i)`（0〜1）を使います。Exact・MCMC の planted/replica
overlap、集計、図示はすべて同じ定義です。未知変数が0個の場合は `q=1` とします。
照合結果と保存形式の変更は
[`docs/SLIDE_MODEL_ALIGNMENT.md`](docs/SLIDE_MODEL_ALIGNMENT.md) にまとめています。

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
log-domainで行い、hard constraintによって両状態の質量がゼロになる場合は
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

現行の `run.json` は schema version 3 で、推論測度と overlap の定義を記録します。
旧モデル・旧 overlap の結果は再利用・集計・理論解析時に拒否します。
旧結果を残し、新しい出力先（例：`results/rho_smoke_matching_fraction`）で再計算してください。

各runには次を保存します。

- `run.json`: 条件、全seed、solver設定、収束・mixing診断、比較指標、metadata
- `arrays.npz`: 盤面、clue、marginal、message residual、MCMC samples、全trace

sweep全体には`aggregate.json`と`phase_scan.svg`を生成します。保存済みrunだけを
再集約する場合は次を実行します。

```powershell
python scripts/aggregate.py results/rho_small
```

## Candidate-region analysis

粗いphase scanの結果から追加計算の候補領域を作成します。候補スコアは
BP–MCMC差、BP非収束、fixed-point依存性、chain間差、R-hat、overlap自己相関時間、
ESSを組み合わせたscan内の相対的な優先度です。RSBの確率や判定値ではありません。

```powershell
python scripts/find_candidates.py results/rho_small/aggregate.json `
  --output results/candidate_plan.json `
  --sizes 4x4,6x6,8x8 `
  --dense-points 11 `
  --top-k 3 `
  --padding-points 1 `
  --disorder-samples 20 `
  --bp-random-restarts 6 `
  --mcmc-chains 4

python scripts/run_candidate_study.py results/candidate_plan.json `
  --output results/candidate_study
```

候補studyはサイズごとに独立したseed streamを使います。生成物は次の通りです。

- `candidate_analysis.json`: 全サイズ・密度の診断値とbinned `P(q)`
- `candidate_summary.svg`: BP–MCMC差、fixed-point spread、overlap緩和、`P(q)`幅
- `overlap_distributions.svg`: 代表密度におけるMCMC/Exactの完全な`P(q)`

保存済みのsize sweepから解析だけをやり直す場合は次を使用します。

```powershell
python scripts/analyze_candidates.py results/candidate_study --overlap-bins 41
```

`mode_count`は平滑化したhistogramに基づく探索用heuristicです。有限サイズで離散的な
overlap値が複数あるだけでも増えるため、多峰性の証拠として単独使用してはいけません。
またoverlapは各instanceの未知変数集合上で定義され、観測率に応じて変数数が変わります。

## RS / RSB theory diagnostics

候補studyの保存済みrunから、Bethe量、factor graph topology、BP fixed pointの局所安定性を
計算します。

```powershell
python scripts/analyze_rs_theory.py results/candidate_study --max-edges 256
```

各runの`theory.json`に加え、study全体の`rs_theory_analysis.json`と
`rs_theory_summary.svg`を生成します。Jacobianのspectral radiusが1未満であることは、
その有限instanceのundamped BP fixed pointが局所安定であることだけを意味します。
これはquenched無限系のAT eigenvalueではなく、1を超えただけでAT線やRSB転移とは
判定しません。Exactに対するBethe誤差、short loop、MCMC mixing、`P(q)`を合わせて
解釈してください。

現行posteriorにおけるobservation-maskの扱い、格子short loopによるcavity法の制限、
random-graph surrogateや1RSBへ進む条件は
[`docs/RS_RSB_theory_notes.md`](docs/RS_RSB_theory_notes.md)にまとめています。
実際に得られた数値、考察、限界、次の検証項目は
[`docs/MILESTONE7_RESULTS_DISCUSSION.md`](docs/MILESTONE7_RESULTS_DISCUSSION.md)を参照してください。

同文書の報告値は過去のモデル・overlap 規約で得たものです。
次の設定は現行 Bernoulli 事後分布向けに更新済みですが、数値結果は未再計算です。
schema version 4 より前の保存結果は混合集計や再開の対象にせず、新しい出力先で再計算してください。

```powershell
python scripts/run_candidate_study.py `
  configs/studies/rsb_candidate_rho_010_050_n20.json `
  --output results/rsb_candidate_rho_010_050_n20_bernoulli_v4

python scripts/analyze_rs_theory.py `
  results/rsb_candidate_rho_010_050_n20_bernoulli_v4 `
  --max-edges 256
```

テストは、パッケージを editable install した後に標準ライブラリだけで実行できます。

```powershell
python -m pip install -e .
python -m unittest discover -s tests -v
```
