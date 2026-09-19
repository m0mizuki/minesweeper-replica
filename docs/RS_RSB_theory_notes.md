# Milestone 7: RS / RSB 理論解析の実行可能性

## 結論

現段階で実装・検証できるのは、各有限 instance に対する Bethe 近似と BP fixed point の局所安定性である。これらを Exact、MCMC、overlap、factor graph の short loop と同時に比較することで、RS 記述が破綻する候補を絞り込める。

一方、通常の Minesweeper 格子は locally tree-like ではなく、長さ4の loop が密に存在する。したがって、有限 instance の BP Jacobian の固有値が1を超えることを AT 線と呼んだり、BP–MCMC の差を 1RSB の証拠と解釈したりはしない。Milestone 7 は、そのような主張に進む前の理論的・数値的な判定基盤を構築する段階とする。

## 1. 解析対象となる測度

コードが比較する target distribution は、観測 clue の hard constraint を満たす配置上の一様分布である。

$$
P(\boldsymbol{x}\mid\boldsymbol{y},A)
=\frac{1}{Z}
\prod_{a\in A}\mathbf 1\!\left(\sum_{i\in\partial a}x_i=y_a\right)
.
$$

$\rho$ は planted instance の生成密度であり、この推論測度の外部パラメータではない。
同じ観測 factor graph に対する Exact・BP・MCMC の重みは $\rho$ に依存しない。

ここでは観測集合 $A$ を quenched design として固定し、その生成確率を posterior likelihood に含めない。現行の `bernoulli_safe` は ground truth が安全なセルだけを観測するため、$A$ 自体が ground truth と相関する。従って「観測 mask も含めた完全な Bayes posterior」を扱うなら測度を再定義する必要がある。現行測度のまま Nishimori identity を仮定してはならない。

熱力学極限を議論する場合は、少なくとも次を固定する必要がある。

- $L_x,L_y\to\infty$ の取り方と境界条件
- mine density $\rho$ と observation rate
- observation mask を平均する ensemble、および mask likelihood を含めるか
- quenched free entropy $N^{-1}\mathbb E[\log Z]$ の平均対象

## 2. 有限 instance の Bethe 量

正規化した BP message から局所規格化定数 $Z_i,Z_a,Z_{ia}$ を作り、

$$
\log Z_{\mathrm B}
=\sum_i\log Z_i+\sum_a\log Z_a-\sum_{(i,a)}\log Z_{ia}
$$

を計算する。hard constraint 上では許容状態の factor energy は0なので、Bethe entropy は

$$
H_{\mathrm B}=\log Z_{\mathrm B}
$$

となる。tree factor graph では Exact と一致する。loopy graph では差

$$
\Delta f=\frac{|\log Z_{\mathrm B}-\log Z|}{|V|}
$$

を Exact が可能なサイズで測り、loop topology と一緒に保存する。負の Bethe entropy や大きな $\Delta f$ は警告だが、それだけでは RSB を意味しない。

## 3. BP fixed point の局所 RS 安定性

factor graph の全 edge に対する variable-to-factor cavity log odds を $h$ とし、damping を除いた BP の1反復を $F(h)$ とする。収束 fixed point $h^\star$ の周りで

$$
J_{ee'}=\left.\frac{\partial F_e}{\partial h_{e'}}\right|_{h^\star}
$$

を中心差分で構成する。スペクトル半径 $\lambda_{\max}=\rho(J)$ が1未満なら、その有限 instance・その fixed point における undamped update は局所的に安定である。

この量には次の制限がある。

- 有限 instance の deterministic Jacobian であり、quenched ensemble の replicon eigenvalue ではない
- damping は数値的収束を変えるので、物理診断では undamped map を線形化する
- hard constraint により message が0または1に達する場合は log odds を clip するため、boundary fraction を併記する
- dense Jacobian の計算量が大きいため edge 数に guard を設ける

従って $\lambda_{\max}=1$ は局所 BP instability の目安であり、AT 線という名称は用いない。

## 4. short loop の切り分け

各 factor graph について独立 cycle rank

$$
r=|E|-|V_{\mathrm{bipartite}}|+C
$$

と長さ4の cycle 数を計算する。Minesweeper の2個の clue が2個以上の未知セルを共有すると4-cycleが生じる。格子を拡大してもこの密度が消えない場合、通常の locally-tree-like random factor graph の cavity equation をそのまま用いる正当性はない。

解析では少なくとも、次の4系列を同じ $(L,\rho)$ 上で比較する。

1. BP Jacobian のスペクトル半径
2. Exact に対する Bethe $\log Z$ 誤差
3. cycle rank と4-cycleの密度
4. MCMC mixing、BP fixed-point dependence、完全な $P(q)$

局所安定な BP でも Exact 誤差だけが loop 密度と共に残るなら、RSB より loopy-BP error が自然である。逆にサイズと共に安定性、mixing、$P(q)$、fixed-point dependence が同じ領域で系統的に悪化する場合に、初めて RSB 解析の優先度が上がる。

## 5. 次の解析ルート

### 空間格子を保つルート

- plaquette/region graph を使う cluster variational method
- 幅を固定した strip の transfer matrix と幅方向 extrapolation
- periodic boundary を含む有限サイズ scaling

これは元の空間モデルに忠実だが、標準的な scalar cavity order parameter には落ちにくい。

### random-graph surrogate ルート

variable/factor degree と clue 分布を合わせた locally-tree-like ensemble を別モデルとして定義すれば、population dynamics による RS density evolution と bug-proliferation stability を検討できる。ただしこれは Minesweeper 格子そのものではなく surrogate であり、数値結果を混同しない。

### 1RSB へ進む条件

1RSB では単一 message ではなく pure state 間の message 分布（survey）を order parameter とし、Parisi parameter $m$ に依存する generalized free entropy と complexity $\Sigma$ を求める必要がある。実装コストが大きいため、次が確認されるまでは着手しない。

- disorder sample 数と system size を増やしても候補領域が再現する
- sampler の mixing 不良だけでは説明できない
- Exact サイズで short-loop error と候補シグナルを切り分けられる
- current lattice または明示した surrogate のどちらを理論対象にするか決定する

## 6. Milestone 7 の判定基準

現時点では「RSB 転移を導出した」とは結論しない。`rs_theory_analysis.json` と `rs_theory_summary.svg` により、Bethe 精度、局所 BP 安定性、short-loop 密度の finite-size trend を保存できることを完了条件とする。十分な disorder average を持つ候補 study で上記の複合シグナルが再現した後、cluster approximation または明示的な random-graph surrogate のどちらへ進むか判断する。
