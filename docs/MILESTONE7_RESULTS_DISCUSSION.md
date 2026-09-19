# Milestone 7: $\rho=0.10$–$0.50$, 20 disorder sample の結果と考察

> **旧モデルによる結果（再計算が必要）**
>
> この文書の数値は、未知変数を Bernoulli($\rho$) 事前分布で重み付けしていた
> 旧 target distribution で計算したものである。現行モデルは clue 制約を満たす
> 配置上の一様分布であり、推論重みに $\rho$ を含めない。そのため、以下の数値を
> 現行モデルの結果として引用・比較してはならない。設定ファイルは現行モデル向けに
> 更新済みであり、同じ条件で Exact・BP・MCMC・Bethe 診断を再実行する必要がある。

## 1. 解析の位置づけ

RSB候補探索の統計を改善するため、mine densityを

$$
\rho\in\{0.10,0.20,0.30,0.40,0.50\}
$$

とし、各 $(L,\rho)$ で20個の独立disorder instanceを計算した。$3\times3$と$4\times4$を合わせて合計200 runである。従来の各点2 sampleのsmoke結果は、本解析で置き換える。

再現用設定と解析元は次の通りである。

- `configs/studies/rsb_candidate_rho_010_050_n20.json`
- `results/rsb_candidate_rho_010_050_n20/candidate_analysis.json`
- `results/rsb_candidate_rho_010_050_n20/rs_theory_analysis.json`

`results/`はGit管理対象外である。全200 runには`run.json`、`arrays.npz`、`theory.json`が保存されている。理論解析時のBP再実行と保存済みmarginalの最大差は全runで0であり、seedから完全に再現できた。

## 2. 実験条件

| 項目 | 設定 |
|---|---|
| 系サイズ | $3\times3$, $4\times4$ |
| mine density | 0.10から0.50まで0.10刻み |
| disorder sample | 各 $(L,\rho)$ で20、合計200 run |
| observation | `bernoulli_safe`, observation rate 0.6 |
| BP | damping 0.2、tolerance $10^{-9}$、最大300 iteration |
| BP初期値 | uniform、random（複数 restart） |
| MCMC | local BFS blocked Gibbs、block size 6 |
| MCMC長 | burn-in 100、300 retained samples、thinning 1、2 chains |
| Exact | 最大12未知変数 |
| overlap histogram | 11 bins |
| RS安定性 | undamped BP mapの数値Jacobian、最大256 edges |

変更したのは $\rho$ gridとdisorder sample数であり、従来のsmoke studyと比較できるよう、BP・MCMC・観測条件は維持した。

## 3. BP・MCMC・overlapの結果

表の値は20 disorder instanceの平均である。BP収束率は3初期値に対する収束割合、$\tau_q$はplanted-overlap traceのintegrated autocorrelation time、$q$幅はinstanceごとの $q_{95}-q_{05}$ の平均である。

| サイズ | $\rho$ | BP収束率 | BP–MCMC MAE | fixed-point spread | $\tau_q$ | $q$幅 | mean max finite R-hat | mean min ESS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3×3 | 0.10 | 1.000 | 0.0048 | $2.7\times10^{-11}$ | 2.84 | 0.136 | 1.011 | 243.5 |
| 3×3 | 0.20 | 0.917 | 0.0040 | $6.8\times10^{-11}$ | 2.12 | 0.264 | 1.010 | 232.2 |
| 3×3 | 0.30 | 0.900 | 0.0202 | 0.0278 | 3.10 | 0.600 | 1.012 | 186.9 |
| 3×3 | 0.40 | 1.000 | 0.0178 | $2.7\times10^{-10}$ | 3.58 | 0.518 | 1.012 | 179.5 |
| 3×3 | 0.50 | 0.850 | 0.0257 | $1.9\times10^{-10}$ | 4.45 | 0.636 | 1.016 | 118.3 |
| 4×4 | 0.10 | 0.967 | 0.0087 | $6.4\times10^{-10}$ | 4.92 | 0.145 | 1.014 | 207.8 |
| 4×4 | 0.20 | 0.867 | 0.0164 | $4.3\times10^{-11}$ | 6.59 | 0.345 | 1.026 | 116.4 |
| 4×4 | 0.30 | 0.917 | 0.0320 | 0.0843 | 5.44 | 0.391 | 1.023 | 164.9 |
| 4×4 | 0.40 | 0.817 | 0.0460 | $7.5\times10^{-11}$ | 6.57 | 0.455 | 1.024 | 129.2 |
| 4×4 | 0.50 | 0.850 | 0.0357 | $3.3\times10^{-10}$ | 10.22 | 0.573 | 1.070 | 75.6 |

### 3.1 観測された傾向

1. $4\times4$のBP–MCMC差は $\rho=0.10$から0.40まで0.0087、0.0164、0.0320、0.0460と増加した。0.50では0.0357へ低下したが、この点ではMCMC診断が最も悪いため、物理的な減少とはまだ解釈できない。
2. $4\times4$の$q$幅は0.145から0.573まで密度とともに広がった。$\tau_q$も $\rho=0.50$ で10.22へ増加した。
3. 明確なBP初期値依存性は $\rho=0.30$ に集中した。marginal spreadが $10^{-6}$ を超えたinstanceは、$3\times3$で1/20、$4\times4$で4/20であり、それ以外の密度では0/20だった。
4. BP収束率の最小値は $4\times4,\rho=0.40$ の0.817だった。fixed-point multiplicityのピークと非収束率のピークは同じ密度ではない。

## 4. MCMC mixingの評価

$4\times4,\rho=0.50$では20 instance中13件で最大finite R-hatが1.01を超え、7件では1.05を超えた。最大値は1.442だった。mean min ESSも75.6まで低下し、chain間marginal spreadは0.1875だった。

さらに、全200 instanceに少なくとも1個のinfinite R-hat変数があった。本実装でinfinite R-hatになるのは、各chain内では変数が定数のままなのにchain間でその定数値が異なる場合であり、初期状態依存性または非混合を明示的に表す。表の`max finite R-hat`はこの成分を除いた最大値なので、それだけを見るとmixingを過大評価する。

従って高密度側だけでなく、今回のMCMCによるBP–MCMC差と$P(q)$全体を平衡posteriorの確定値として解釈するには、現在のburn-in 100、300 samples、2 chainsでは不足している。infinite R-hat変数数、finite R-hat、ESS、変更率、chain別overlap traceを併用し、block sizeの拡大またはsamplerの改良後に再評価する必要がある。

今回のscanでは「RSBらしい遅化を確定した」のではない。全密度でsampler検証が必要であり、その中でも $\rho=0.50$ の有限R-hat、ESS、$\tau_q$が最も悪い。

## 5. RS / Bethe診断

$\lambda_{\mathrm{BP}}$はundamped BP mapのJacobian spectral radiusである。局所不安定率は、BPが収束して$\lambda_{\mathrm{BP}}$を計算できたinstanceのうち $\lambda_{\mathrm{BP}}>1$ となった割合である。Bethe誤差は $|\log Z_{\mathrm B}-\log Z|/|V|$。`Exact数`はBethe–Exact比較が可能だったinstance数である。

| サイズ | $\rho$ | primary BP収束率 | mean $\lambda_{\mathrm{BP}}$ | 局所不安定率 | boundary率 | Bethe誤差 | Exact数 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3×3 | 0.10 | 1.00 | 0.000 | 0.000 | 0.227 | $3.5\times10^{-17}$ | 20 |
| 3×3 | 0.20 | 0.90 | 0.094 | 0.056 | 0.497 | 0.0066 | 18 |
| 3×3 | 0.30 | 0.90 | 0.231 | 0.111 | 0.289 | 0.0198 | 18 |
| 3×3 | 0.40 | 1.00 | 0.136 | 0.100 | 0.215 | 0.0079 | 20 |
| 3×3 | 0.50 | 0.85 | 0.141 | 0.000 | 0.120 | 0.0100 | 17 |
| 4×4 | 0.10 | 0.95 | 0.130 | 0.053 | 0.570 | 0.0050 | 19 |
| 4×4 | 0.20 | 0.85 | 0.077 | 0.059 | 0.658 | 0.0033 | 17 |
| 4×4 | 0.30 | 0.90 | 0.211 | 0.111 | 0.625 | 0.0275 | 17 |
| 4×4 | 0.40 | 0.80 | 0.219 | 0.125 | 0.593 | 0.0112 | 16 |
| 4×4 | 0.50 | 0.85 | 0.257 | 0.176 | 0.300 | 0.0159 | 13 |

### 5.1 解釈

- mean $\lambda_{\mathrm{BP}}$ は全点で1未満だが、平均は少数の局所不安定instanceを隠す。$4\times4$の局所不安定率は高密度側で0.111、0.125、0.176と増えた。
- Bethe誤差は単調ではなく、最大は $4\times4,\rho=0.30$ の0.0275だった。この点はBP fixed-point dependenceも最大であり、追加検証の第一候補である。
- boundary message率は $4\times4$ の0.10–0.40で0.57–0.66と高い。message clippingの影響が残るため、小さいspectral radiusを強いRS安定性とは解釈しない。
- 4-cycle密度は全点で有限であり、格子factor graphはlocally tree-likeではない。Bethe誤差やBP–MCMC差にはshort-loop errorも含まれる。

## 6. 現段階の考察

20 disorder sampleへ増やしたことで、従来の2 sampleで見えた単一の大きな値がensemble全体の代表ではないことが分かった。警告は1点に集中せず、異なる密度で異なる形で現れている。

- $\rho=0.30$: BP fixed-point dependenceとBethe誤差が最大
- $\rho=0.40$: BP収束率が最低、BP–MCMC MAEが最大
- $\rho=0.50$: MCMC relaxation、finite R-hat、ESSが最も悪く、局所BP不安定率も最大

このため、現時点で単一の「RSB候補点」を決めるより、BP側の診断から $\rho=0.30$–$0.50$を暫定候補領域として扱う方が妥当である。MCMC側は全200 instanceで初期状態依存性が残っており、特に $\rho=0.50$ は非平衡の影響を強く受けている可能性がある。

> 本scanは、$4\times4$の $\rho=0.30$–$0.50$ にBP多重解、非収束、局所不安定性、overlap緩和の複合的な警告が存在することを示した。しかし、system sizeは2種類のみで、MCMC mixingとshort-loop効果も十分に分離できていないため、RSB転移またはAT線の証拠とはしない。

## 7. 次の優先計算

1. まず小規模Exact可能系でblock sizeを増やすか、より大域的なconstraint-preserving moveを導入し、infinite R-hat変数を解消する。
2. $\rho=0.30,0.40,0.50$でMCMCを4 chains以上、burn-inとsample数を現在の少なくとも10倍にして再計算する。
3. infinite R-hatがなく、finite R-hat $<1.01$、十分なESS、初期状態間一致を満たしたinstanceでBP–MCMC差と$P(q)$を再評価する。
4. $6\times6$、$8\times8$を追加し、BP多重解率、非収束率、$\lambda_{\mathrm{BP}}>1$率、overlap susceptibilityのサイズ依存性を調べる。
5. $\rho=0.25$–$0.45$を0.025または0.05刻みに細分化し、0.30付近のfixed-point dependenceが有限幅の領域か確認する。
6. Jacobianの有限差分幅とclipping probabilityを変え、boundary messageを含むspectral radiusの頑健性を確認する。
