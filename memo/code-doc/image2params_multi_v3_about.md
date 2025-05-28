# 📚 ノートブック全体構成ドキュメント（パラメトリック画像学習編）

このプロジェクトでは、p5.jsで生成されたジェネラティブ画像から  
複数のパラメータ（形状・色相など）を予測するCNNを学習します。

モデルのロス関数には、**単体モデルによって収集された val_loss の履歴**を活用し、  
各出力に適切な重みを割り当てることで、多タスク学習の最適化を図ります。

---

## 🎯 `image2param_single.ipynb`：単体モデルによる val_loss 記録

### 📌 目的
- 各パラメータ（dipCount, hue_cos, hue_sin, circleCount）に対して個別に学習を実行し、
  **検証ロス (val_loss)** を `val_losses.json` に追記記録する

### 📁 出力
- `loss_stats/val_losses.json`：
  - 各パラメータの val_loss を配列形式で蓄積
  - 例：
    ```json
    {
      "dipCount": [0.203, 0.187],
      "hue_cos": [0.110, 0.097],
      ...
    }
    ```

### 🔁 実行制御
- 各パラメータに対して `REPEAT` 回数だけ学習を行う（例：REPEAT=10）

### 🧩 主な構成ステップ
1. 初期設定（REPEAT/パス設定/val_losses.json 読み込み）
2. CSV読み込み・対象抽出
3. Dataset定義（target_colを指定）
4. train/val分割とDataLoader作成
5. 単体CNNでの学習 + val_loss測定（REPEAT分）
6. val_losses.jsonへの追記保存
7. 統計の確認・可視化（平均・分散・ヒストグラム）

---

## 🧠 `image2params_v3.ipynb`：重み付きマルチタスク学習

### 📌 目的
- `val_losses.json` を参照し、出力ごとの val_loss に応じた重みを導出
- 4出力（dipCount, hue_cos, hue_sin, circleCount）同時回帰モデルを学習

### ⚙️ 重みモード
- `VAL_LOSS_MODE` で切り替え可能：
  - `"latest"`：最新の1つを使う
  - `"average"`：履歴の平均を使う

### 🎯 損失関数
- `WeightedMSELoss`:  
  各出力に重みをかけた MSE を計算
  ```python
  loss = mean(weights * (y_pred - y_true)^2)
  ```

### 🧩 主な構成ステップ
1. 初期設定（loss_weights計算 + デバイス）
2. metadata.csv + Z-score 正規化 + 保存
3. Dataset定義（hue→cos/sin変換含む）
4. train/val 分割・DataLoader構築
5. CNNモデル定義（出力4ユニット）
6. 学習ループ（WeightedMSE + wandbログ）
7. 推論＋逆正規化＋画像上に数値表示
8. Lossカーブ描画（train / val）

---

## 🔄 ファイル連携の流れ

```
image2param_single.ipynb ─┬─> val_losses.json （記録）
                          │
image2params_v3.ipynb <───┘ （参照 + 重みに変換）
```

---

## 🔧 今後の拡張案

- ロスごとの個別ログ（dipだけの loss を wandb に記録）
- 評価指標の細分化（角度誤差、MAEなど）
- モデル出力の選択（条件付き生成への接続）