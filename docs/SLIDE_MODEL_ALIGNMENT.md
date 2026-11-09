# 推論モデルの更新とスライド草稿との対応

2026-11-09の指定により、事後分布を次に変更した。

$$
P(\boldsymbol{x}\mid\boldsymbol{c})=
\frac{1}{Z_\rho}\prod_i P_0(x_i)
\prod_a\delta\!\left(\sum_{i\in\partial a}x_i,c_a\right),
\qquad P_0(x_i)=\rho^{x_i}(1-\rho)^{1-x_i}.
$$

変数は未開示セル、制約は開示済み安全セルの数字とする。総地雷数は固定しない。
観測集合を固定し、mask の選択尤度は含めない。mask の生成過程まで条件付けする完全な
Bayes posterior との違いは `RS_RSB_theory_notes.md` を参照。

## BP

$$
m_{i\to a}(x_i)\propto P_0(x_i)
\prod_{b\in\partial i\setminus a}m_{b\to i}(x_i),
\qquad
b_i(x_i)\propto P_0(x_i)\prod_{a\in\partial i}m_{a\to i}(x_i).
$$

事前因子は変数更新と belief にそれぞれ1回含める。因子→変数の更新式は従来どおり。
既定の変数→因子初期値は `[1−ρ, ρ]`、因子→変数は中立な `[0.5, 0.5]`。
`initialization="prior"` を既定とし、`uniform` / `random` は初期値依存性の診断用に残す。
制約のない変数の地雷確率はρとなる。木では厳密周辺確率に一致し、ループでは近似となる。

## Blocked Gibbs

$$
P(\boldsymbol{x}_B\mid\boldsymbol{x}_{\setminus B},\boldsymbol{c})
\propto \prod_{i\in B}P_0(x_i)
\prod_{a:\partial a\cap B\ne\emptyset}
\delta\!\left(\sum_{i\in\partial a}x_i,c_a\right).
$$

外側の事前因子とブロックに接続しない制約は定数なので打ち消せる。
外側を固定し、接続する全制約を満たす候補を列挙し、ブロック内地雷数kに対する
`ρ^k (1−ρ)^(|B|−k)` で重み付けして抽出する。研究用実装のBFSブロック選択は
配置に依存しないため、各更新は目標分布に対する詳細釣り合いを満たす。
ただし小さなブロックでは解空間を移動できない場合がある。定常分布の妥当性と
エルゴード性・混合速度は別に診断する。

ρ=1/2、または有効候補の地雷数がすべて同じ場合、条件付き分布は一様になる。

## Exact・Bethe・端点

Exact の周辺確率・planted/replica overlap は正規化した事前重みで集計する。
`number_of_feasible_solutions` と `mine_count_histogram` は重みを付けない制約解の数を保つ。
`partition_function` は解数でなく事前の下で制約が満たされる確率となる。
ρ=1/2でも `Z_rho = number_of_feasible_solutions / 2^N` である。

対数重みと log-sum-exp で計算し、分配関数がアンダーフローしても対数分配関数と
正規化した確率は保持する。ρ=0,1 は決定論的な事前分布として扱う。
制約の事前質量がゼロなら Exact は `has_posterior_mass=False`、BP は `infeasible`。
MCMC は制約を満たし事前確率が正の初期配置を要求する。

Bethe の変数規格化定数と安定性解析の BP map にも事前因子を含める。
エントロピーは

$$
H_B=\log Z_B-\sum_i[b_i(1)\log\rho+b_i(0)\log(1-\rho)]
$$

となる（0 log 0 は0）。

## 保存結果と検証

`run.json` は schema version 4、`inference.target_distribution` は
`bernoulli_posterior`、`rho_used_as_inference_prior` は true とする。
旧結果の再開・混合集計・理論診断を拒否する。新しい出力先で再計算すること。

```powershell
python scripts/run_sweep.py configs/sweeps/rho_smoke.json --output results/rho_smoke_bernoulli_v4
```

overlap は未開示変数上の一致率 `q=mean(x_i == x'_i)`（空配置では1）を保つ。
旧スピン定義の overlap を変換するだけでは、旧一様測度の結果を現行測度へ変換できない。

`tests/test_slide_model.py` では `x0+x1=1, x1+x2=1` の解010と101を用い、
確率がそれぞれ1−ρとρ、周辺確率が `[ρ, 1−ρ, ρ]` となることを確認する。
BP・Exact・MCMC の比較、端点、アンダーフロー、Bethe entropy、旧結果の拒否も検証する。
大規模な候補 study の数値結果はこの変更では再計算していない。
