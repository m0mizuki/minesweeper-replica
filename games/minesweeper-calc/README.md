# BP Message Lab

3×3 マインスイーパーを因子グラフとして表し、sum-product BP の2種類のメッセージ更新を1サイクルずつ追跡する静的ブラウザアプリです。

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
- `styles.css`: 盤面・計算トレースの表示

将来 MCMC 表示を追加する場合も、BP 固有の計算は `bp-engine.mjs` に閉じているため、別エンジン・別表示パネルとして追加できます。
