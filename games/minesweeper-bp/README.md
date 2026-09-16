# Bernoulli Minesweeper + Belief Propagation

各セルの地雷変数を独立に

\[
x_i^0 \sim \operatorname{Bernoulli}(\rho)
\]

から生成する Web 版マインスイーパーです。実現した総地雷数への条件付けは行いません。開示済みの数字を局所制約として、リポジトリ本体の `minesweeper_csp.run_bp`（sum-product BP）で未開示セルの地雷周辺確率を求めます。

## 仕様

- 初手を安全にする条件付けは行わず、ゲーム作成時に全セルを独立生成
- 総地雷数をゲーム生成にも BP の因子にも使用しない
- フラグはプレイヤー用の印であり、BP の観測証拠として使用しない
- SQA / QUBO / OpenJij への依存なし
- BP の収束状態、反復数、最終 message residual、因子数を画面に表示

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
