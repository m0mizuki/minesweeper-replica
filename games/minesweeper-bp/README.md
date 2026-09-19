# Bernoulli Minesweeper + Belief Propagation

各セルの地雷変数を独立に

\[
x_i^0 \sim \operatorname{Bernoulli}(\rho)
\]

から生成する Web 版マインスイーパーです。実現した総地雷数への条件付けは行いません。開示済みの数字を局所制約として、リポジトリ本体の `minesweeper_csp.run_bp`（sum-product BP）で、制約を満たす配置上の一様分布における未開示セルの地雷周辺確率を求めます。$\rho$ は盤面生成だけに使い、BP の事前分布には使いません。

## 仕様

- 初手を安全にする条件付けは行わず、ゲーム作成時に全セルを独立生成
- 総地雷数をゲーム生成にも BP の因子にも使用しない
- BP は Bernoulli 事前分布を使わず、制約のないセルの周辺確率は 0.5
- フラグはプレイヤー用の印であり、BP の観測証拠として使用しない
- SQA / QUBO / OpenJij への依存なし
- BP の収束状態と反復数を画面に表示
- 画面のスライダーで地雷密度 $\rho$ を設定
- BP 実行後にセルを開いても再推論はせず、未開示セルの確率表示を維持

## 起動

リポジトリルートから:

```powershell
uv venv
uv pip install -r games/minesweeper-bp/requirements.txt
uv run uvicorn --app-dir games/minesweeper-bp main:app --reload
```

ブラウザで <http://127.0.0.1:8000> を開きます。

## テスト

```powershell
uv run python -m unittest discover -s games/minesweeper-bp/tests -v
```
