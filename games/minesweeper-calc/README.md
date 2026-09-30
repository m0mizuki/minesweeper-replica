# Minesweeper Inference Lab

2×2または3×3のマインスイーパーを因子グラフとして表し、次の2手法の計算を追跡する静的ブラウザアプリです。セットアップ画面で推論手法、盤面サイズ、地雷密度、開示マスを選択できます。

- sum-product BP: 2種類のメッセージ更新を1サイクルずつ表示
- blocked Gibbs sampling: 数字制約で連結した未開示マスをブロックとし、候補列挙、制約判定、重みの正規化、乱数による採択を1ステップずつ表示

## 起動

リポジトリルートから次のいずれかを実行し、表示されたURLの `/games/minesweeper-calc/` を開きます。

```powershell
uv run python -m http.server 8000
```

または、`games/minesweeper-calc` をルートとして配信します。

```powershell
uv run python -m http.server 8000 --directory games/minesweeper-calc
```

ES Modules を使用しているため、`index.html` を直接開くのではなくHTTPサーバー経由で表示してください。

## テスト

```powershell
node --test games/minesweeper-calc/tests/*.test.mjs
```

## 構成

- `js/bp-engine.mjs`: 盤面、因子グラフ、BP更新、計算トレースを扱う独立モジュール
- `js/gibbs-engine.mjs`: ブロック構築、条件付き分布、Gibbs更新、計算トレースを扱う独立モジュール
- `js/app.mjs`: 画面状態と描画
- `js/math-renderer.mjs`: KaTeXによるLaTeX数式描画とフォールバック
- `styles.css`: 盤面・計算トレースの表示

数式表示には [KaTeX](https://katex.org/) 0.18.9 をCDN経由で使用します。CDNを読み込めない場合はLaTeXソース表示へフォールバックします。

メッセージ計算の詳細は「軽量」と「KaTeX」を切り替えられます。軽量モードでは詳細式をLaTeXソースのまま表示し、盤面上の式と2つのメッセージ定義式だけをKaTeXで描画します。

blocked Gibbs sampling では、現在のブロック外の標本を固定し、数字制約を満たすブロック内割当だけに Bernoulli 事前分布の重みを付けます。初期配置は全数字制約を満たす割当から決定し、各ステップ後に標本から経験周辺確率を更新します。
