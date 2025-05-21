## 📘 Step 1: メタデータと画像前処理の準備（dipCount）

このステップでは、CSVファイルから `dipCount` ラベルを取り出して、画像ファイル名と紐づける準備を行います。  
PythonとPandasの基本的なデータ操作が使われています。

---

```python
import pandas as pd
from pathlib import Path
```

- `pandas` は表形式のデータ（CSVなど）を扱うための代表的ライブラリ。
- `Path` はファイル・ディレクトリパスを扱うモジュールで、OSを問わず安全にパス操作できる。

---

```python
DATA_DIR = Path('../../data/circle-stroke')
CSV_PATH = DATA_DIR / 'metadata.csv'
IMG_DIR = DATA_DIR / 'images'
```

- `Path(...)` でパスオブジェクトを作成しておくと、`/` 演算子で直感的にパスをつなげられる。
- これにより、`"../../data/circle-stroke/metadata.csv"` のようなパスを安全に記述できる。

---

```python
df = pd.read_csv(CSV_PATH)
```

- `read_csv` はCSVファイルを読み込み、`DataFrame` と呼ばれる表形式のオブジェクトに変換。
- これで `df` は、CSVの全行全列を持つPythonオブジェクトになる。

---

```python
df_targets = df[['id', 'dipCount']].copy()
```

- `[['id', 'dipCount']]` は、DataFrameから指定した2列だけを抽出している。
- `.copy()` をつけているのは、「元の `df` に影響を与えず、新しい独立した表を作る」ため。
  （Pandasでは参照のまま持つと、意図せず元データが変更されることがある）

---

```python
df_targets['filename'] = 'image_' + df_targets['id'] + '.png'
```

- `df_targets['id']` は文字列の列なので、Pythonの `+` で文字列連結できる。
- この処理で `"20250521_0001"` → `"image_20250521_0001.png"` に変換し、画像ファイル名として使えるようにしている。

---

```python
df_targets['filepath'] = df_targets['filename'].apply(lambda x: IMG_DIR / x)
```

- `.apply(lambda x: ...)` は「各行に対して関数を適用する」操作。
- ここでは `filename` を `IMG_DIR` と結合して、画像ファイルのフルパスを作っている。

---

```python
df_targets = df_targets[df_targets['filepath'].apply(lambda x: x.exists())].reset_index(drop=True)
```

- `x.exists()` は、指定したファイルパスが実在するかどうかを判定する `Path` オブジェクトのメソッド。
- `df_targets[...]` で **条件を満たす行だけに絞り込み**。
- `.reset_index(drop=True)` によって、行番号（インデックス）を振り直している。
  - `drop=True` を指定することで、古いインデックス列は新しい表に残らないようになる。

---

```python
print(f"使用可能なデータ数: {len(df_targets)}")
```

- `len(df_targets)` は行数（＝画像枚数）を返す。
- `f"文字列{値}"` は **f-string** と呼ばれ、値を文字列の中に埋め込んで表示できるPython 3.6以降の書き方。

---

### ✅ まとめ

- このステップで「**存在する画像ファイルにだけ対応したCSVデータ**」が得られます。
- 以降の学習ステップではこの `df_targets` を使って安全に学習を進められます。

## 📘 Step 3: データの分割と DataLoader の作成

このステップでは、モデルの訓練（training）と検証（validation）に必要なデータを分割し、  
PyTorchの `DataLoader` を用いて**ミニバッチ単位で自動的にデータを供給できるようにする**構造を整えます。

---

```python
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
```

- `train_test_split` は **scikit-learn** に含まれる関数で、  
  リストやDataFrameなどをランダムに訓練用と検証用に分けるのに使います。
- `DataLoader` は PyTorch の機能で、`Dataset` クラスからデータをまとめて取り出してくれるもの。
  - ミニバッチにまとめる（複数画像を1度に渡す）
  - データをランダムに並べ替える（シャッフル）
  - 自動的にループ処理できる（for文にそのまま使える）

---

```python
df_train, df_val = train_test_split(df_targets, test_size=0.2, random_state=42)
```

- `df_targets` からデータを **訓練用80%（df_train）** と **検証用20%（df_val）** に分割しています。
- `test_size=0.2`：全体の20%をvalidation側にするという意味。
- `random_state=42`：**乱数のシードを固定**するための指定。
  - これをつけると、実行のたびにランダム分割結果が変わることを防げます。
  - 再現性が大事な研究・実験には必須のテクニック！

---

```python
train_dataset = DipCountDataset(df_train, IMG_DIR)
val_dataset = DipCountDataset(df_val, IMG_DIR)
```

- **Step 2で作った `DipCountDataset` クラス**に、  
  分割後のDataFrame（df_train / df_val）を渡して、それぞれ **訓練用/検証用Dataset** を作っています。
- これにより、`Dataset` インスタンスから `image, dipCount` のペアを1つずつ取り出せるようになります。

---

```python
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

- `DataLoader` は **Datasetから複数データをまとめて取り出す仕組み**です。
- `batch_size=32`：
  - 一度に32個の画像とラベルを取り出す（=1バッチ）
  - 大きすぎるとメモリ不足になる、小さすぎると収束が遅いので、32〜64くらいが入門にはおすすめ。
- `shuffle=True`：
  - 毎エポックごとに訓練データの順番をシャッフルすることで、モデルが偏った順番に慣れるのを防ぐ。
- `shuffle=False`（検証データ）：
  - 検証では順番のランダム化は不要。再現性と安定性のため順序は固定しておく方がよい。

---

### ✅ このステップのまとめ

| 処理 | 内容 |
|------|------|
| データ分割 | `train_test_split()` で 80%:20% に分ける |
| Dataset作成 | `DipCountDataset(df, img_dir)` でデータセット化 |
| DataLoader作成 | `DataLoader(..., batch_size, shuffle)` で学習用構造に |

このあと、`for images, labels in train_loader:` という形でループすれば、32枚ずつデータを取り出して学習が行えます。

## 📘 Step 4: CNNモデルの定義（DipCountCNN）

このステップでは、PyTorchで畳み込みニューラルネットワーク（CNN）を定義して、  
画像から `dipCount`（くびれの数）を予測する回帰モデルを作成します。

PyTorchでは、ニューラルネットワークのモデルは `nn.Module` を継承してクラスとして定義します。

---

```python
import torch.nn as nn
import torch.nn.functional as F
```

- `nn` モジュールにはニューラルネットワークのレイヤー（線形層・畳み込み層など）が入っています。
- `F` は**関数的（functional）なAPI**で、ReLUやsoftmaxなどを関数として使いたいときに使います。

---

```python
class DipCountCNN(nn.Module):
```

- PyTorchでモデルを作るときの決まり文句。
- `nn.Module` を継承することで、学習・保存・推論などの機能を使えるようになります。
- モデルの構造（レイヤー）と処理の流れ（forward）をこの中で定義します。

---

```python
    def __init__(self):
        super().__init__()
```

- `__init__()` はクラスの初期化関数（Pythonの決まり）です。
- `super().__init__()` で `nn.Module` 側の初期化もきちんと呼び出しています（これ超重要！）。

---

```python
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
```

- `Conv2d(in_channels=3, out_channels=16, kernel_size=3)`：
  - 入力画像がRGBなのでチャネル数3（C=3）
  - 出力は16チャネル（＝16種類の特徴マップを作る）
  - 3x3のカーネルで畳み込みを行う
- `padding=1` にすることで、入力と同じサイズ（高さ・幅）を保てる（**境界をゼロで囲む**）

---

```python
        self.pool = nn.MaxPool2d(2, 2)
```

- 最大プーリング（MaxPooling）レイヤー。  
- 2×2の領域から最大値を取り出して、**画像サイズを半分に縮める**処理。  
  - 入力が `128×128` なら、これで `64×64` になる。

---

```python
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
```

- 1層目で得られた16チャネルの特徴マップを32チャネルに増やす。
- 同じく3x3のカーネルと1ピクセルのpaddingを使う。
- 画像サイズは `64×64 → 32×32` に。

---

```python
        self.fc1 = nn.Linear(32 * 32 * 32, 64)
```

- 全結合（Fully Connected）層。
- 前の畳み込みブロックでできた**最終特徴マップ**は `チャネル数 32 × 高さ 32 × 幅 32`
- それを1本のベクトルに flatten（平坦化）した後、64次元に圧縮。

---

```python
        self.fc2 = nn.Linear(64, 1)
```

- 最終出力：1個のスカラー値（`dipCount`）を出力します。
- 回帰タスクなので出力は **連続値（float）**、活性化関数は使いません。

---

```python
    def forward(self, x):
```

- モデルに画像を入力したときに「どのように処理を進めるか」を定義する関数。
- PyTorchでは `model(input)` と書くと `forward()` が呼ばれる仕組みです。

---

```python
        x = self.pool(F.relu(self.conv1(x)))
```

- 1層目：畳み込み → ReLU → プーリング
- `conv1(x)`：特徴マップを抽出
- `F.relu(...)`：ReLU（0以下を0に）で非線形性を導入
- `pool(...)`：サイズを半分に

---

```python
        x = self.pool(F.relu(self.conv2(x)))
```

- 同様に2層目：畳み込み → ReLU → プーリング

---

```python
        x = x.view(x.size(0), -1)
```

- `x.view(batch_size, -1)`：4次元の特徴マップを2次元に flatten
- `-1` は「自動的に残りの次元を計算せよ」という意味

---

```python
        x = F.relu(self.fc1(x))
```

- flattenされたベクトルを全結合層に通して、64次元に変換
- 活性化関数はReLU

---

```python
        x = self.fc2(x)
```

- 64次元 → 1次元（dipCountの予測値）

---

```python
        return x
```

- 推論結果（`dipCount`の予測値）を返します。

---

### ✅ このモデルのまとめ

| レイヤー種別 | 処理内容 |
|--------------|----------|
| Conv2d ×2    | 画像特徴の抽出 |
| MaxPool2d ×2 | 特徴の圧縮 |
| Linear ×2    | 高次元ベクトルを1値に変換 |
| ReLU         | 非線形性の導入 |

最終的に「画像 → dipCount（くびれの数）」を予測することができます。

## 📘 Step 5: 学習ループ（損失記録付き）

このステップでは、定義したCNNモデルに対して**実際に学習を行うループ処理**を構築します。

モデルの訓練には「訓練モードでの予測→損失計算→逆伝播→重みの更新」という流れを繰り返します。  
また、各エポックごとに**訓練損失・検証損失を記録**し、あとでロスカーブを描画できるようにします。

---

```python
import torch
import torch.optim as optim
```

- `torch` はPyTorch本体。テンソル計算・モデル実行の中心。
- `optim` はPyTorchの最適化関数（SGD, Adamなど）を提供。

---

```python
model = DipCountCNN()
```

- 先ほど定義した `DipCountCNN` のインスタンスを作成。
- このオブジェクトに画像を渡すことで、予測が行えるようになります。

---

```python
criterion = nn.MSELoss()
```

- **損失関数（Loss Function）**：ここでは `Mean Squared Error（MSE）` を使用。
- `MSELoss` は「予測値と正解値の差を2乗して平均をとる」回帰用の基本的な誤差関数です。

---

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

- モデルの学習に使う**最適化手法（optimizer）**の指定。
- `Adam` は自動で学習率を調整してくれる便利な最適化アルゴリズム。
- `model.parameters()` で、学習すべきすべての重みを渡しています。

---

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

- GPUが使える場合はGPU（cuda）、なければCPUを使う。
- `.to(device)` によって、モデルが使うテンソルを指定デバイスに転送します。
  - 学習データも同様に `.to(device)` する必要があります。

---

```python
num_epochs = 5
train_losses = []
val_losses = []
```

- `num_epochs`：エポック数（＝データ全体を何周するか）。少なめでテスト的にスタート。
- `train_losses` / `val_losses`：エポックごとの平均損失を格納するリスト。

---

### 🔁 学習ループ本体

```python
for epoch in range(num_epochs):
```

- エポック（訓練データを一通り使い切る単位）を繰り返すループ。

---

```python
    model.train()
```

- `train()`：PyTorchの訓練モードに切り替える関数。
- DropoutやBatchNormの挙動が訓練モード用になります。

---

```python
    total_loss = 0.0
```

- このエポックでの合計損失を保持しておく変数。

---

```python
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
```

- 訓練データを `train_loader` から1バッチずつ取り出します。
- GPUを使っている場合は `.to(device)` でデータもデバイスに移す必要があります。

---

```python
        optimizer.zero_grad()
```

- 以前の勾配情報をクリアします。
- PyTorchでは明示的に `.zero_grad()` しないと、前のエポックの勾配が残ってしまいます。

---

```python
        outputs = model(images)
```

- 画像をモデルに通して予測値を取得（`dipCount` の推定）。

---

```python
        loss = criterion(outputs, labels)
```

- 予測値と正解値を比較して、損失（誤差）を計算します。

---

```python
        loss.backward()
```

- 誤差をネットワーク全体に**逆伝播（backpropagation）**させて、各重みの勾配を計算。

---

```python
        optimizer.step()
```

- `optimizer` が勾配を使って**モデルの重みを更新**します。

---

```python
        total_loss += loss.item() * images.size(0)
```

- このバッチでの損失に、サンプル数（batch size）をかけて合計損失に加算。
- `.item()` は1要素のTensorからPythonのfloatを取り出す関数。

---

```python
    avg_train_loss = total_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)
```

- そのエポックでの平均損失を算出し、記録しておきます。

---

### 🔍 検証ループ

```python
    model.eval()
```

- 推論モードに切り替え（Dropoutなどが無効化される）。

---

```python
    val_loss = 0.0
    with torch.no_grad():
```

- `torch.no_grad()` ブロックは**勾配計算を無効化**するため、検証時にメモリと速度を節約できます。

---

```python
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
```

- 訓練ループと同様に、各バッチごとの損失を合計。

---

```python
    avg_val_loss = val_loss / len(val_loader.dataset)
    val_losses.append(avg_val_loss)
```

- 検証の平均損失を記録。

---

```python
    print(f"[{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
```

- エポックごとの進行と損失状況を表示します。

---

### ✅ このステップで得られること

| 内容 | 説明 |
|------|------|
| 学習処理の構築 | データ1周ごとに訓練・検証を行い、パラメータを更新 |
| 損失記録 | ロスカーブを描画するための記録が残る |
| デバイス対応 | GPUでもCPUでも動くように構成済み |

---

## 📘 Step 6: 複数画像の予測と可視化

このステップでは、学習したモデルがどれくらい `dipCount` を正しく予測できているかを、  
**複数枚の検証画像を使って目視確認する**処理を実装します。

機械学習では数値だけでなく、**実際に出力がどう見えているかを直感的に確認する**のがすごく大切！

---

```python
model.eval()
```

- 学習後のモデルを **推論モード（eval）** に切り替えます。
- `eval()` を呼ぶと Dropout や BatchNorm の挙動が「訓練用」から「推論用」に変わります。

---

```python
num_samples = 8
```

- 表示するサンプル画像の枚数を指定。
- 今回は例として8枚。縦横の比率に応じて調整できます。

---

```python
plt.figure(figsize=(16, 4))
```

- `matplotlib.pyplot.figure()` は図全体のキャンバスを用意します。
- `figsize=(横, 縦)` はインチ単位のサイズ。横長の表示にしたいので `(16, 4)` に設定。

---

```python
for i in range(num_samples):
```

- 検証データセットから先頭 `num_samples` 枚を順番に表示していきます。

---

```python
    sample_img, sample_label = val_dataset[i]
```

- `val_dataset`（検証用Dataset）から1枚ずつ画像とラベル（真のdipCount）を取り出します。
- `sample_img` は Tensor、`sample_label` は Tensor([dipCount])。

---

```python
    with torch.no_grad():
        pred = model(sample_img.unsqueeze(0).to(device)).item()
```

- `torch.no_grad()` で推論時の勾配計算をOFFにして高速化。
- `sample_img.unsqueeze(0)`：
  - `sample_img` は `[C, H, W]` なので、`[1, C, H, W]` に次元を追加して1バッチ化。
  - モデルは常にバッチ形式の入力を想定しているため必要な処理。
- `.to(device)`：画像をモデルと同じデバイス（GPU or CPU）に送る。
- `.item()`：出力Tensorから純粋なfloatに変換。

---

```python
    plt.subplot(1, num_samples, i+1)
```

- `subplot(rows, cols, index)`：複数の小さなグラフを並べるための関数。
- ここでは 1行 × `num_samples` 列で表示。
  - `i+1` としているのは、`subplot` のindexは1から始まる仕様だから。

---

```python
    plt.imshow(sample_img.permute(1, 2, 0))
```

- PyTorchの画像Tensorは `[C, H, W]` 形式だけど、matplotlibは `[H, W, C]` 形式が必要。
- `.permute(1, 2, 0)` でチャンネルを一番後ろに持ってきて表示可能な形に変換します。

---

```python
    plt.title(f"Pred: {pred:.1f}\nTrue: {sample_label.item():.1f}")
```

- 画像の上にタイトルをつけて、**予測値と正解値**を両方表示します。
- `f"..."` はPythonのフォーマット文字列。
  - `{pred:.1f}` → 小数点以下1桁で表示
  - `\n` は改行

---

```python
    plt.axis("off")
```

- 枠の目盛り（x軸・y軸）を非表示にして、画像だけをきれいに見せるための設定。

---

```python
plt.tight_layout()
plt.show()
```

- `tight_layout()`：画像同士の間の余白を自動調整して詰めてくれる。
- `show()`：このタイミングで実際に描画される。

---

### ✅ このステップでできること

| 内容 | 説明 |
|------|------|
| モデルの予測精度を目視確認 | 予測と正解がどれくらい近いかを直感的に把握できる |
| モデルの傾向を観察 | 「このタイプの画像は外してるな」とかの気づきを得る |
| 複数枚まとめて表示 | 1枚ずつよりも効率よく判断できる |

## 📘 Step 7: ロスカーブの可視化

このステップでは、Step 5で記録した「訓練損失（train_loss）」と「検証損失（val_loss）」の推移をグラフにして、  
モデルの学習の様子を**視覚的に確認できるように**します。

---

```python
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
```

- `train_losses` と `val_losses` は、それぞれエポックごとの損失値を保存したリスト。
- `plt.plot(...)` はそのリストを折れ線グラフとして表示します。
- `label=...` によって凡例（legend）用の名前を設定。

---

```python
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
```

- `xlabel`：X軸のラベル → 学習の回数（エポック数）
- `ylabel`：Y軸のラベル → 損失の大きさ（MSE）

---

```python
plt.title("Loss Curve")
```

- グラフ全体のタイトルを設定。ここでは「Loss Curve（損失曲線）」としています。

---

```python
plt.legend()
```

- `label="..."` で設定した名前を使って、凡例（どの線が何か）を表示します。

---

```python
plt.grid(True)
```

- 補助線（グリッド）を表示することで、値の変化が読み取りやすくなります。

---

```python
plt.show()
```

- 実際にグラフを表示。これを最後に書かないとグラフが出てきません。

---

### ✅ よくあるロスカーブの見方

| 状態 | 解釈 |
|------|------|
| 両方スムーズに下がっている | ✅ 学習順調！過学習なし！ |
| trainは下がってるけどvalが横ばい〜上昇 | ⚠️ 過学習（trainに合わせすぎ）かも |
| どちらも高止まり | ⚠️ モデルが学習できてない可能性（モデルが小さい or データ不足） |

---

### ✅ このステップで得られること

- 学習の進行状況が視覚的にわかる
- 過学習や学習不足の兆候を早期に察知できる
- モデルの改善（エポック数・構造・正則化）の方針を立てやすくなる