# 🔍 画像に対する人の好み分析：先行研究詳細まとめ

## 1. 顔の魅力度予測（SCUT-FBP 系）

**概要：**  
500〜5,500枚の顔画像に対し、1〜5段階の美的評価スコアを付与し、CNNなどで好みを予測する研究。主にアジア系女性画像でスタート。

**主な成果：**  
- SCUT‑FBP（500枚）：CNNモデルによる予測で Pearson 相関係数 0.8187（最高） [oai_citation:0‡arxiv.org](https://arxiv.org/abs/1511.02459?utm_source=chatgpt.com)。  
- SCUT‑FBP5500（5,500枚，男女・人種含む）：AlexNet・ResNet・ResNeXt を比較。ResNeXt などでは PC ≈0.93、MAE≅0.22 と極めて高精度 [oai_citation:1‡github.com](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release?utm_source=chatgpt.com)。  
- 損失関数工夫（ComboLoss など）：SEResNeXt50 + ComboLoss により、PC 約1%改善 [oai_citation:2‡arxiv.org](https://arxiv.org/abs/2010.10721?utm_source=chatgpt.com)。  
- 特徴領域解釈研究：目や鼻が美しさ予測に重要と判明（XRAI、Permutation Feature Importance など使用） [oai_citation:3‡link.springer.com](https://link.springer.com/article/10.1007/s44163-025-00226-8?utm_source=chatgpt.com)。

**用途・意義：**  
- 個人差や文化的背景の解析  
- 将来的な「髪型の好み」のような特定要素への応用が可能

---

## 2. テキスト→画像生成における好み収集（Pick-a-Pic / PickScore）

**概要：**  
ユーザーが生成画像2枚のうち好むものを選択するWebアプリ「Pick‑a‑Pic」で50万件以上の好みペアを収集し、CLIPを fine-tune した「PickScore」を開発。

**成果：**  
- PickScore：好み選択予測率 70.5%（人間68.0%、ランダム60.8%） [oai_citation:4‡arxiv.org](https://arxiv.org/html/2305.01569?utm_source=chatgpt.com)。  
- FIDよりも human ranking と高い相関（0.917 対 −0.900） [oai_citation:5‡arxiv.org](https://arxiv.org/html/2305.01569?utm_source=chatgpt.com)。  
- 生成モデル選定実験では、人は PickScore 上位画像を 71–85% の確率で好む [oai_citation:6‡arxiv.org](https://arxiv.org/pdf/2305.01569?utm_source=chatgpt.com)。  
- 報酬関数として組み込めば、拡散モデルの生成美的性向も向上する傾向あり。

---

## 3. Diffusion モデルの好み制御（DSPO, SPO）

**DSPO（ICLR 2025）:**  
- 拡散モデルを、Score-matching に基づいてユーザー好み分布に接近するよう fine-tune  
- 人の評価タスクで、従来の Preference Learning 比で優位な性能 [oai_citation:7‡openreview.net](https://openreview.net/forum?id=xyfb9HHvMe&utm_source=chatgpt.com)

**SPO（CVPR 2025）:**  
- 拡散過程の各ステップで選好ペアを生成 → 微細な美的調整を促す最適化  
- 従来手法より速く収束、テキスト・画像整合性を維持したまま美的向上

---

## 4. CLIP を用いた美的評価（IAA 課題）

**Aesthetic Post‑Training / CLIP ベース**  
- CLIPの埋め込み特徴＋AVEデータセットで単層線形回帰 → CNNより高精度  
- 少ないファインチューニングで ImageNet 型より効率良く学習可能

---

## 5. データ収集・評価手法の工夫

- **Likert評価**：Kaplan夫妻の景観心理学では、1–5評価で「水辺・開放感」など好みに影響  
- **ペアワイズ比較**：生成画像や医用画像で信頼性高く評価が得られる手法として多数採用  
- **視線追跡 / 行動評価**：商品画像では視線位置やクリック数を好み代理に使用  
- **EEG / fMRI**：UX設計において快感・注意など神経指標を評価に組み入れる研究進展中  

---

## ✅ 統合的観点と応用への示唆

- **モデル構築**：顔・プロダクト・アートなど得意ターゲットに応じ、データ・モデル選定を最適化  
- **評価設計**：Likert・ペアワイズ・視線・神経指標の併用設計で多角的な好み評価が可能  
- **生成器との連携**：GAN/VAE/SPOD 等に好みスコアを導入し、対話的・段階的生成が実現済  
- **潜在変数探索**：StyleGANなどでジェネラティブアート作品の好みバリエーションを直感的に生成可能

---

## 🛠 次のステップ提案

- ご自身のジェネラティブ作品に PickScore または SPO を組み込む  
- パラメータごとの好み分布を可視化し、ユーザークラスタ別の評価傾向分析に展開  
- EEG／視線など非言語データを追加し、より客観的な評価モデルを構築可能