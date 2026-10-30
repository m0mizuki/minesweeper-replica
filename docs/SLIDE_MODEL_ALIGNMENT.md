# スライド草稿との問題設定の照合

2026-10-30に提示された「スライド草稿.md」の本文のCSP・BP・MCMCを基準とする。
草稿中で今後の拡張として述べられている総地雷数制約や、別案のBernoulli事前分布は、
今回の推論測度には含めない。

| 項目 | 現行の設定と照合結果 |
| --- | --- |
| 真の配置 | 各セル独立に Bernoulli(`rho`) で生成。既存実装が一致 |
| 変数と制約 | 未開示セルが二値変数、開示済み安全セルの周囲8近傍の和が数字に一致。既存実装が一致 |
| 推論測度 | `P(x given c) = Z^-1 product_a delta(sum_i x_i, c_a)`。既存のExact・BP・MCMCが一致 |
| 総地雷数 | 固定せず、推論にも大域制約を入れない。既存実装が一致 |
| BP | 一様初期値は `[0.5, 0.5]`。変数メッセージ・beliefに生成密度の事前因子を掛けない。既存実装が一致 |
| MCMC | 因子グラフ上のBFSで局所ブロックを作り、外側を固定して全接続制約を満たす候補から一様抽出。既存の研究用実装が一致 |
| BP–MCMC差 | 未開示変数上の平均絶対誤差 `comparison.bp_vs_mcmc.mae` が草稿の `D_BP-MCMC` に対応 |
| overlap | 旧スピン積から、草稿の一致率 `q = mean(x_i == x'_i)` へ変更。Exact・MCMC・集計・図示を0〜1に統一 |

`N` はoverlap・周辺確率の比較では未開示変数数を意味する。
保存済みの `model.N` は盤面セル数なので、比較時の分母は
`instance.number_of_variables` を使う。未知変数が0個の場合、overlapは1、
周辺確率の誤差は0とする。

草稿では開示マスの選び方と境界条件は指定されていないため、既存の
`all_safe` / `bernoulli_safe` と、盤面外を隣接に含めない境界条件を使用する。
観測率は引き続き実験設定で指定する。

## 保存結果の扱い

`run.json` のschema versionを3にし、`inference` に一様測度・事前因子なし・
総地雷数固定なし、`overlap` に `matching_fraction`・未開示変数集合・空配置の規約を記録する。
同じ情報を正規化した `config.json`、`aggregate.json`、`candidate_analysis.json` に残す。
入力configで別の測度やoverlapを明示するとエラーにする。
モデル情報を省略した従来の入力configは、新規計算の設定として使用できる。

旧runの再開、現行runとの混合集計、旧runへの理論診断はエラーにする。
既存結果を保持したまま、新しい出力先で再計算すること。

```powershell
python scripts/run_sweep.py configs/sweeps/rho_smoke.json --output results/rho_smoke_matching_fraction
```

同じ測度の旧overlapなら `q = (q_spin + 1)/2`、平均も同じ変換、
分散は旧値の1/4、分布幅は旧値の1/2になる。ただし旧Bernoulli事前測度の結果は、
この変換だけで現行結果にはならない。保存済みデータの自動変換は行わない。

## 検証

`tests/test_slide_model.py` で `x0+x1=1, x1+x2=1` を使い、
地雷数が異なる解 `010` と `101` の重みが各1/2になること、BPとExactの一致、
MCMCでの両解のサンプリング、同じグラフの推論が `rho` に依存しないことを確認する。
その他の回帰テストで、Exactのoverlap分布、MCMCの一致率、候補解析、旧結果の拒否を検証する。

大規模な候補領域の再計算や、mixing改善・RSB転移の確定はこの変更には含めない。
