# BP Message Lab

2×2または3×3のマインスイーパーを因子グラフとして表し、sum-product BP の2種類のメッセージ更新を1サイクルずつ追跡する静的ブラウザアプリです。セットアップ画面で盤面サイズ、地雷密度、開示マスを選択できます。

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
node --test games/minesweeper-calc/tests/bp-engine.test.mjs
```

## 構成

- `js/bp-engine.mjs`: 盤面、因子グラフ、BP更新、計算トレースを扱う独立モジュール
- `js/app.mjs`: 画面状態と描画
- `js/math-renderer.mjs`: KaTeXによるLaTeX数式描画とフォールバック
- `styles.css`: 盤面・計算トレースの表示

数式表示には [KaTeX](https://katex.org/) 0.18.9 をCDN経由で使用します。CDNを読み込めない場合はLaTeXソース表示へフォールバックします。

メッセージ計算の詳細は「軽量」と「KaTeX」を切り替えられます。軽量モードでは詳細式をLaTeXソースのまま表示し、盤面上の式と2つのメッセージ定義式だけをKaTeXで描画します。

将来 MCMC 表示を追加する場合も、BP 固有の計算は `bp-engine.mjs` に閉じているため、別エンジン・別表示パネルとして追加できます。
