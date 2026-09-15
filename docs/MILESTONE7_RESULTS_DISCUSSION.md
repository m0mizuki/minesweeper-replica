# Milestone 7: 得られた結果と考察

## 1. この文書の位置づけ

Milestone 6–7 の動作確認として実行した小規模 candidate study と、その保存済み run に対する有限 instance の RS / Bethe 診断をまとめる。解析元は次のファイルである。

- `results/candidate_smoke_v2/candidate_plan.json`
- `results/candidate_smoke_v2/candidate_analysis.json`
- `results/candidate_smoke_v2/rs_theory_analysis.json`

以下の結果は、実装が end-to-end で動作し、どの観測量を本計算で追うべきかを確認するための **smoke study** である。各 $(L,\rho)$ に disorder sample が2個しかないため、相転移や RSB の統計的証拠とはみなさない。

## 2. 実験条件

| 項目 | 設定 |
|---|---|
| 系サイズ | $3\times3$, $4\times4$ |
| mine density | $\rho=0.10, 0.25, 0.40$ |
| disorder sample | 各点2 |
| observation | `bernoulli_safe`, observation rate $0.6$ |
| BP | damping $0.2$, tolerance $10^{-9}$, 最大300 iteration |
| BP初期値 | prior, uniform, random の3種類 |
| MCMC | local BFS blocked Gibbs, block size 6 |
| MCMC長 | burn-in 100、300 samples、thinning 1、2 chains |
| Exact | 最大12未知変数 |
| overlap histogram | 11 bins |
| RS安定性 | undamped BP mapの数値Jacobian、最大256 edges |

観測 mask は quenched design として扱い、その生成確率を posterior likelihood には含めていない。従って、この結果に Nishimori identity を仮定していない。

## 3. BP・MCMC・overlap の結果

表中の値は disorder 平均である。`q幅` は replica-overlap histogram の $q_{95}-q_{05}$、$\tau_q$ は planted-overlap trace の integrated autocorrelation time である。

| サイズ | $\rho$ | BP–MCMC marginal MAE | BP fixed-point spread | $\tau_q$ | $q$幅 | mode数 |
|---:|---:|---:|---:|---:|---:|---:|
| 3×3 | 0.10 | $2.1\times10^{-13}$ | $4.4\times10^{-14}$ | 1.00 | 0.000 | 1 |
| 3×3 | 0.25 | $1.7\times10^{-12}$ | $4.3\times10^{-12}$ | 1.00 | 0.000 | 1 |
| 3×3 | 0.40 | 0.0101 | $4.3\times10^{-13}$ | 2.73 | 0.545 | 3 |
| 4×4 | 0.10 | $2.1\times10^{-12}$ | $1.1\times10^{-11}$ | 1.00 | 0.000 | 1 |
| 4×4 | 0.25 | 0.0522 | $1.4\times10^{-10}$ | 5.13 | 0.727 | 2 |
| 4×4 | 0.40 | 0.1699 | $1.9\times10^{-10}$ | 9.14 | 1.000 | 2 |

観測された傾向は次の通りである。

1. $\rho=0.10$ では両サイズとも BP–MCMC 差、overlap緩和、$P(q)$幅が小さい。
2. $4\times4$ では $\rho=0.25$ から BP–MCMC MAE、$\tau_q$、$q$幅が同時に増え、$\rho=0.40$ でさらに増大した。
3. 3種類の初期値間の BP marginal spread は全点で $10^{-10}$ 程度以下であり、この sample から複数の明確な BP fixed point は確認できなかった。
4. mode数は高密度側で増えたが、11-bin histogram と少数 sample に依存する探索用 heuristic である。多峰的な pure-state structure の証拠として単独では使えない。

## 4. RS / Bethe 診断の結果

$\lambda_{\mathrm{BP}}$ は undamped BP map のJacobian spectral radius、`Bethe誤差`は $|\log Z_{\mathrm B}-\log Z|/|V|$ である。括弧内は値を計算できた instance 数であり、BP非収束などにより2未満の場合がある。

| サイズ | $\rho$ | BP収束率 | $\lambda_{\mathrm{BP}}$ | 局所不安定率 | Bethe誤差 | boundary message率 | 4-cycle密度 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3×3 | 0.10 | 1.00 | 0.000 (2/2) | 0.00 | $1.4\times10^{-17}$ (2/2) | 0.385 | 1.750 |
| 3×3 | 0.25 | 1.00 | 0.000 (2/2) | 0.00 | $7.4\times10^{-17}$ (2/2) | 0.670 | 1.583 |
| 3×3 | 0.40 | 0.50 | 0.000 (1/2) | 0.00 | 0.000 (1/2) | 0.667 | 0.771 |
| 4×4 | 0.10 | 1.00 | 0.000 (2/2) | 0.00 | $7.4\times10^{-17}$ (2/2) | 0.841 | 1.229 |
| 4×4 | 0.25 | 0.50 | 1.430 (1/2) | 1.00 | 0.0360 (1/2) | 0.500 | 0.550 |
| 4×4 | 0.40 | 1.00 | 0.716 (2/2) | 0.50 | 0.0996 (2/2) | 0.409 | 1.266 |

ここで $4\times4,\rho=0.40$ の $\lambda_{\mathrm{BP}}=0.716$ は2 instanceの平均であり、局所不安定率0.5が示すように、少なくとも一方は $\lambda_{\mathrm{BP}}>1$ である。平均値だけで局所安定と判定してはならない。

## 5. 考察

### 5.1 数値的に疑わしい領域

$4\times4$ の $\rho=0.25$–$0.40$ では、BP–MCMC差、overlap autocorrelation、$P(q)$幅、Bethe誤差、局所BP不安定性の複数指標が同時に悪化した。この一致は、本計算で優先的に調べる領域としては妥当である。

ただし $\rho=0.25$ では BP と理論量を評価できたのが1/2 instanceのみである。$\rho=0.40$ でも局所安定性は instance 間で一致していない。現状から「臨界密度が0.25付近にある」と推定することはできない。

### 5.2 RSB以外の説明

factor graph には1変数あたりおよそ0.55–1.75個の4-cycleがあり、short loop は希薄ではない。従って Bethe誤差や BP–MCMC差は、locally-tree-like 仮定の破れによる loopy-BP approximation error でも説明できる。

また、MCMCは高密度側で $\tau_q$ が増えている。burn-in 100、保持 sample 300という短い chain では、平衡化不足や metastability によって MCMC marginal と $P(q)$ が歪む可能性がある。現時点では「BPが誤っている」のか「MCMCが混合していない」のかを完全には切り分けられていない。

### 5.3 boundary messageの影響

messageが0または1に近い instance が多く、boundary message率は最大0.84であった。この場合、cavity log odds の有限差分では clipping の影響が大きい。$\lambda_{\mathrm{BP}}=0$ は強く凍結された message によって得られる場合があり、RS安定性の強い証拠ではない。spectral radius は boundary fraction と必ず併記する。

### 5.4 現段階の結論

今回の結果は、$4\times4$ の中・高密度側を追加検証する理由を与えるが、RSB転移の存在やAT線を示してはいない。特に、disorder数、system size、MCMC長が不足し、short loop と boundary clipping の影響も大きい。

従って Milestone 7 の結論は次のように限定する。

> 有限instanceのBethe精度と局所BP安定性を、MCMC mixing、$P(q)$、short-loop topologyと同じ parameter grid 上で比較できるようになった。smoke studyでは $4\times4,\rho=0.25$–$0.40$ に複合的な警告が見られたが、RSBの判定には至らない。

## 6. 次に必要な検証

1. $\rho=0.20$–$0.45$ を細分化し、各点20以上の disorder sample を取る。
2. $6\times6$, $8\times8$ へ拡大し、同じ異常がサイズとともに強まるか調べる。
3. MCMCのburn-inとchain長を増やし、chain別trace、R-hat、ESS、初期状態依存性を再確認する。
4. Jacobianの有限差分幅とclipping probabilityを変え、spectral radiusの数値安定性を確認する。
5. Exact可能サイズでは Bethe誤差を4-cycle密度で層別化し、short-loop errorとの相関を測る。
6. 格子モデルを保つcluster variational法と、別モデルとして定義したlocally-tree-like surrogateのどちらを解析対象にするか決定する。
