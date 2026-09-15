# Minesweeper CSP: BP / MCMC / RS–RSB Study

## 1. このプロジェクトの目的

本プロジェクトでは、**Minesweeper を planted CSP（Constraint Satisfaction Problem）として定式化し、Belief Propagation (BP) と Markov Chain Monte Carlo (MCMC) を比較することで、Replica Symmetry (RS) が破れる可能性のある領域を数値的に探索する**。

最終的には、数値計算で得られた転移候補に対して、可能であればレプリカ法・cavity method を用いた RS / RSB の理論解析を行う。

当面の研究フローは以下で固定する。

1. Minesweeper を CSP として定式化する
2. BP を実装し、RS 仮定に基づく解を調べる
3. MCMC を十分慎重に実装し、平衡分布を数値的に評価する
4. BP と MCMC の一致・不一致領域を調べる
5. overlap、緩和時間、BP の収束性などを合わせて RSB 転移候補を探す
6. 可能ならレプリカ法で RS / RSB を解析する

---

## 2. 現在のスコープ

### 今扱う地雷生成

各セルの planted ground truth を

$$
x_i^0 \sim \mathrm{Bernoulli}(\rho),
\qquad
x_i^0 \in \{0,1\}
$$

として独立に生成する。

- $x_i^0=1$ : 地雷
- $x_i^0=0$ : 非地雷
- $\rho$ : 地雷密度

したがって地雷数そのものは固定しない。

### 今は扱わないもの

将来的には

$$
\sum_i x_i^0 = N\rho
$$

のように**総地雷数を固定した ensemble** へ拡張する予定だが、現段階では実装・解析対象に含めない。

CodeX は、固定地雷数制約を現在のモデルへ暗黙に追加してはならない。

---

## 3. Minesweeper の CSP 定式化

セル集合を $V$  とする。

各未知セル $i\in V$  に binary variable

$$
x_i \in \{0,1\}
$$

を対応させる。

観測された clue cell の集合を $A$  とし、clue $a\in A$  の周囲にある変数集合を $\partial a$  とする。

ground truth $x^0$  から clue

$$
y_a = \sum_{i\in\partial a} x_i^0
$$

を生成する。

各 clue は CSP constraint

$$
\sum_{i\in\partial a} x_i = y_a
$$

を与える。

したがって posterior / Gibbs measure は

$$
P(\mathbf{x}\mid \mathbf{y})
=
\frac{1}{Z}
\left[
\prod_{a\in A}
\mathbf{1}
\left(
\sum_{i\in\partial a}x_i=y_a
\right)
\right]
\left[
\prod_{i\in V}
\rho^{x_i}(1-\rho)^{1-x_i}
\right].
$$

ここで $Z$  は規格化定数である。

この分布が、BP と MCMC が比較すべき**共通の target distribution** である。

### factor graph

- variable node: $x_i$ 
- factor node: clue constraint $a$ 
- edge: $i\in\partial a$ 

として factor graph を構成する。

---

## 4. 観測条件について

地雷配置の生成と、どの clue を観測するかは別の処理として実装する。

最低限、コード上では以下を分離すること。

```text
generate_ground_truth(...)
generate_observation_mask(...)
generate_clues(...)
build_factor_graph(...)
```

観測 clue の選び方は研究結果に影響するため、ハードコードしない。

experiment config から切り替え可能な構造にする。

現時点で観測方法が確定していない実験については、README や config に明示し、別の observation protocol を混同しないこと。

---

## 5. Belief Propagation

BP は factor graph 上で行う。

variable-to-factor message:

$$
m_{i\to a}(x_i)
\propto
P_0(x_i)
\prod_{b\in\partial i\setminus a}
m_{b\to i}(x_i)
$$

where

$$
P_0(x_i)
=
\rho^{x_i}(1-\rho)^{1-x_i}.
$$

factor-to-variable message:

$$
m_{a\to i}(x_i)
\propto
\sum_{\{x_j:j\in\partial a\setminus i\}}
\mathbf{1}
\left(
x_i+\sum_{j\in\partial a\setminus i}x_j=y_a
\right)
\prod_{j\in\partial a\setminus i}
m_{j\to a}(x_j).
$$

最終 marginal:

$$
P_i(x_i)
\propto
P_0(x_i)
\prod_{a\in\partial i}
m_{a\to i}(x_i).
$$

### 実装上の要件

- message は必ず正規化する
- zero division / underflow を防ぐ
- damping を設定可能にする
- convergence tolerance と max iteration を config 化する
- convergence したかどうかを必ず記録する
- iteration 数を保存する
- 複数初期値から BP を走らせられるようにする
- 異なる初期値から異なる fixed point に入るか確認できるようにする

Minesweeper の factor degree は小さいため、factor update は最初は全列挙でもよい。
必要になれば dynamic programming / convolution に置き換える。

### BP についての解釈

ここで BP は RS cavity approximation に対応するものとして扱う。

ただし Minesweeper の通常の格子 factor graph には短い loop が存在するため、

**BP と MCMC の不一致だけを根拠に RSB と判断してはいけない。**

不一致は単純な loopy-BP approximation error でも発生しうる。

---

## 6. MCMC

MCMC は BP の「正解」と機械的にみなすのではなく、

**target posterior を十分に平衡化してサンプリングできた場合の数値的 benchmark**

として使用する。

### 重要

hard constraint

$$
\sum_{i\in\partial a}x_i=y_a
$$

の下では、単純な single-site flip MCMC は非エルゴード的になったり、極端に動きにくくなったりする可能性がある。

そのため「とりあえず Metropolis を実装して比較する」という方針は禁止する。

### 推奨する段階的実装

#### Phase 1: 小規模系

小規模 $N$  では feasible states を完全列挙し、

- exact marginal
- exact overlap distribution
- exact partition function / number of solutions

を求める。

これを BP と MCMC の両方の unit test / benchmark にする。

#### Phase 2: blocked MCMC

局所 block を選び、block 内の binary variables を全列挙して、外部変数を固定した conditional distribution から更新する。

block size は config 化する。

#### Phase 3: 必要なら高度な sampler

混合が悪い場合には候補として

- larger blocked Gibbs
- replica exchange / parallel tempering
- soft-constraint auxiliary ensemble
- cluster / constraint-preserving move

などを検討する。

ただし sampler を変更した場合も、最終的に目的とする posterior が同一であることを検証する。

### MCMC diagnostics

最低限以下を保存する。

- chain length
- burn-in
- thinning
- acceptance rate（該当する sampler の場合）
- integrated autocorrelation time
- effective sample size
- 複数 chain 間の比較
- observable の trace
- overlap の trace
- 初期状態依存性

複数の独立 chain を使用する。

MCMC と BP が一致しない場合には、まず MCMC が十分に mixing しているかを疑う。

---

## 7. BP と MCMC の比較

各 parameter point で、少なくとも以下を比較する。

### marginal

BP:

$$
p_i^{\mathrm{BP}} = P_{\mathrm{BP}}(x_i=1)
$$

MCMC:

$$
p_i^{\mathrm{MCMC}} = P_{\mathrm{MCMC}}(x_i=1)
$$

比較指標の例:

$$
\mathrm{MAE}
=
\frac{1}{N}
\sum_i
\left|
p_i^{\mathrm{BP}}-p_i^{\mathrm{MCMC}}
\right|
$$

$$
\mathrm{RMSE}
=
\sqrt{
\frac{1}{N}
\sum_i
\left(
p_i^{\mathrm{BP}}-p_i^{\mathrm{MCMC}}
\right)^2
}.
$$

必要に応じて correlation、KL divergence、Jensen-Shannon divergence 等も用いる。

### planted configuration との overlap

spin 表現

$$
s_i = 2x_i-1,
\qquad
s_i^0=2x_i^0-1
$$

を用いて

$$
q_0
=
\frac{1}{N}
\sum_i
s_i s_i^0
$$

を定義する。

### replica overlap

独立な posterior sample $\mathbf{x}^{(1)},\mathbf{x}^{(2)}$  に対して

$$
q_{12}
=
\frac{1}{N}
\sum_i
s_i^{(1)}s_i^{(2)}
$$

を計算する。

特に

$$
P(q_{12})
$$

の形状を記録する。

RSB 候補を探す際には、平均 overlap だけでなく distribution 全体を見る。

---

## 8. RSB 転移候補を探す観測量

parameter sweep を行い、以下の変化を同時に見る。

### BP 側

- convergence / non-convergence
- convergence iteration 数
- 初期値依存性
- 複数 fixed point の存在
- BP marginal

### MCMC 側

- autocorrelation time
- relaxation time
- effective sample size
- chain 間の不一致
- metastability
- overlap distribution $P(q)$ 

### BP vs MCMC

- marginal MAE / RMSE
- overlap
- entropy に関連する量
- observable の一致・不一致

### RSB 候補とみなすための考え方

例えば、

- MCMC は十分平衡化している
- BP の結果が MCMC と系統的にずれる
- BP fixed point が初期値に依存する
- relaxation time が急増する
- $P(q)$  が広がる、または多峰化する

などが同じ parameter region で現れる場合、RSB 転移候補として詳しく調べる。

ただし、有限サイズ効果や格子の short loop による BP 誤差を必ず切り分ける。

---

## 9. parameter sweep

最初に主要 parameter として地雷密度

$$
\rho
$$

を sweep する。

観測率など他の parameter を導入する場合も config から指定可能にする。

各 parameter point で複数 disorder sample を生成する。

つまり、

```text
for parameter point:
    for disorder sample:
        generate planted instance
        run exact solver if small
        run BP
        run MCMC
        compute observables
    aggregate disorder statistics
```

という構造にする。

平均値だけでなく

- standard deviation
- standard error
- quantile
- sample-to-sample fluctuation

も可能な範囲で保存する。

---

## 10. disorder average と再現性

この研究では random instance ensemble を扱うため、単一盤面の結果だけで結論を出さない。

乱数 seed を必ず保存する。

推奨:

```python
seed_instance
seed_bp
seed_mcmc
```

を分離する。

実験結果から、使用した config と seed を使って完全に再現できるようにする。

---

## 11. 小規模 exact solver は必須

小さい $N$  では全状態列挙を実装する。

exact solver は以下のために使用する。

1. CSP implementation の検証
2. BP implementation の検証
3. MCMC implementation の検証
4. finite-size の正解との比較
5. BP–MCMC disagreement の原因切り分け

exact solver が扱えるサイズでは、

```text
Exact
vs
BP
vs
MCMC
```

の三者比較を必ず行う。

---

## 12. 将来のレプリカ法

数値実験で RS / RSB 転移候補が確認できた場合、

$$
\mathbb{E}[\log Z]
$$

の quenched average を解析対象とする。

replica trick

$$
\mathbb{E}[\log Z]
=
\lim_{n\to 0}
\frac{\mathbb{E}[Z^n]-1}{n}
$$

を用いた RS saddle point、必要であれば 1RSB などの解析を検討する。

ただし、Minesweeper は空間構造を持つため、標準的なランダムグラフ CSP と同じ形で解析できるとは限らない。

したがって、

1. ensemble を明確に定義する
2. 数値的に転移候補を確認する
3. 解析可能な order parameter を特定する
4. RS ansatz
5. 必要なら RSB ansatz

の順で進める。

レプリカ解析を先に無理に実装する必要はない。

---

## 13. 推奨ディレクトリ構成

```text
.
├── README.md
├── pyproject.toml
├── configs/
│   ├── default.yaml
│   └── sweeps/
├── src/
│   └── minesweeper_csp/
│       ├── __init__.py
│       ├── instance.py
│       ├── observation.py
│       ├── factor_graph.py
│       ├── exact.py
│       ├── bp.py
│       ├── mcmc.py
│       ├── observables.py
│       ├── diagnostics.py
│       └── io.py
├── scripts/
│   ├── run_single.py
│   ├── run_sweep.py
│   └── aggregate.py
├── notebooks/
│   ├── 01_instance_check.ipynb
│   ├── 02_bp_validation.ipynb
│   ├── 03_mcmc_validation.ipynb
│   └── 04_phase_scan.ipynb
├── tests/
│   ├── test_instance.py
│   ├── test_exact.py
│   ├── test_bp.py
│   └── test_mcmc.py
└── results/
```

`results/` の巨大な raw data は必要に応じて Git 管理対象外にする。

---

## 14. 出力フォーマット

各 run について少なくとも以下を保存する。

```yaml
model:
  Lx:
  Ly:
  N:
  rho:
  observation_protocol:

instance:
  seed:
  number_of_mines:
  number_of_constraints:

bp:
  converged:
  iterations:
  tolerance:
  damping:
  marginals:

mcmc:
  sampler:
  seed:
  burn_in:
  samples:
  autocorrelation_time:
  effective_sample_size:
  marginals:

comparison:
  marginal_mae:
  marginal_rmse:
  planted_overlap:
  replica_overlap_mean:

metadata:
  git_commit:
  timestamp:
```

実際には JSON / YAML / NPZ / Parquet 等を適切に使い分けてよい。

重要なのは、後から結果だけを見て実験条件が分かることである。

---

## 15. 最初に実装する順番

CodeX は以下の順番で進める。

### Milestone 1 — instance / CSP

- Bernoulli($\rho$ ) で planted configuration を生成
- clue を生成
- CSP consistency check
- factor graph 構築
- unit test

### Milestone 2 — exact solver

- 小規模全列挙
- exact marginal
- exact overlap statistics

### Milestone 3 — BP

- BP message update
- damping
- convergence detection
- exact solver との比較
- 複数 initialization

### Milestone 4 — MCMC

- blocked sampler
- diagnostics
- exact solver との比較
- multiple chains

### Milestone 5 — phase scan

- $\rho$  sweep
- disorder average
- BP / MCMC comparison
- overlap
- relaxation / autocorrelation time
- visualization

### Milestone 6 — RSB candidate analysis

- suspicious region を高密度 sweep
- system-size dependence
- $P(q)$ 
- BP fixed-point dependence
- MCMC mixing check

### Milestone 7 — theory

- 数値結果をもとに RS / RSB の解析可能性を検討

実装済みの有限サイズ診断:

- BP fixed pointにおけるBethe $\log Z$、free entropy、entropy
- Exactに対するBethe誤差
- factor graphのcycle rankと長さ4のshort-loop数
- undamped BP mapの数値Jacobianとspectral radius
- system-size・$\rho$依存性の集約と可視化

ここでspectral radiusは有限instanceの局所安定性であり、AT線とは同一視しない。
格子のshort loop、observation-maskをposterior likelihoodに含めない現行規約、
1RSBへ進む判定条件の詳細は
[`RS_RSB_theory_notes.md`](RS_RSB_theory_notes.md)を参照する。

---

## 16. CodeX への実装方針

実装時は以下を守る。

1. **数式上のモデルとコード上の target distribution を一致させる**
2. **Bernoulli ensemble と固定地雷数 ensemble を混ぜない**
3. **小規模 exact result を基準に BP / MCMC を検証する**
4. **MCMC の非収束を物理現象と誤認しない**
5. **BP の非収束を即座に RSB と解釈しない**
6. **乱数 seed・config・git commit を保存する**
7. **研究用コードなので、速度より最初は correctness を優先する**
8. **parameter sweep の前に単一 instance の validation を十分行う**
9. **グラフ・数値データの双方を保存する**
10. **将来の固定地雷数 ensemble へ拡張しやすい設計にするが、現時点では実装しない**

---

## 17. 最初の成功条件

まず以下が成立すれば Phase 1 の成功とする。

- 小規模 instance で CSP が正しく生成される
- planted configuration が全 constraint を満たす
- exact solver が feasible solutions を列挙できる
- BP marginal が exact marginal に近い領域を確認できる
- MCMC marginal が exact marginal と統計誤差内で一致する
- 同じ instance に対して BP / MCMC / Exact を一括比較できる

その後、system size と $\rho$  を増やして phase structure の探索へ進む。

---

## 18. 研究上の中心的な問い

このプロジェクトで最終的に答えたい問いは以下である。

> Minesweeper の planted CSP ensemble において、どの parameter region まで RS 的な記述が成立し、どの領域から複雑な solution-space structure や RSB を示唆する挙動が現れるのか。

そのため、単に Minesweeper を解く solver を作ることが目的ではない。

**solution space の統計構造、BP の成立範囲、MCMC dynamics、overlap structure を調べることが研究目的である。**
