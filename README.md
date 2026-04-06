# VAE-based Image Anomaly Detection

本專案使用 **Variational Autoencoder (VAE)** 進行影像異常檢測（anomaly detection），針對不同產品類別分別訓練模型，並以 **重建誤差（reconstruction error）** 作為判斷樣本是否為瑕疵品的依據。

Notebook 內容目前以 `vae.ipynb` 為主，流程包含資料前處理、VAE 建模、訓練、測試、門檻選擇，以及各類別與整體準確率的統計。

---

## 專案目標

資料集包含兩種影像：

- **正常商品（good）**
- **瑕疵商品（defective / error）**

模型會先只學習特定類別的正常樣本分布，接著在測試階段比較輸入影像與重建影像之間的差異。若重建誤差較高，代表該影像較可能為異常／瑕疵品。

評估指標以 **Accuracy** 為主。

---

## 方法概述

本專案採用每個類別各自訓練一個 VAE 模型的方式，流程如下：

1. 讀取指定類別的訓練集與驗證集。
2. 將影像 resize 為 `256 x 256`，並正規化到 `[-1, 1]`。
3. 訓練 VAE 以學習正常影像的潛在表示。
4. 對測試集中的 `good` 與 `error` 影像分別計算重建誤差。
5. 透過 **ROC curve** 找出最佳 threshold。
6. 根據 threshold 將影像分類為正常或瑕疵，並計算：
   - Good Accuracy
   - Defective Accuracy
   - Overall Accuracy

---

## 模型架構

### Encoder
- 輸入：`3 x 256 x 256`
- 多層卷積 + BatchNorm + LeakyReLU
- 最後壓縮為 latent vector
- `latent_dim = 256`

### Decoder
- 由 latent vector 還原為影像
- 使用多層 `ConvTranspose2d` 上採樣
- 最後使用 `Tanh()` 輸出到 `[-1, 1]`

### Loss Function
總 loss 由兩部分組成：

- **Reconstruction Loss**：MSE Loss
- **KL Divergence**：限制潛在空間分布

公式概念如下：

```python
Loss = Reconstruction Loss + beta * KL Divergence
```

在目前實作中：
- `beta = 1`
- reconstruction loss 使用 `mse_loss`
- scheduler 使用 `ReduceLROnPlateau`

---

## 資料夾結構

依照 notebook 中的路徑設定，資料集預期結構如下：

```bash
data/
├── train/
│   ├── bottle/
│   ├── cable/
│   ├── capsule/
│   ├── pill/
│   └── toothbrush/
├── val/
│   ├── bottle/
│   ├── cable/
│   ├── capsule/
│   ├── pill/
│   └── toothbrush/
└── test/
    ├── bottle/
    │   ├── good/
    │   └── error/
    ├── cable/
    │   ├── good/
    │   └── error/
    ├── capsule/
    │   ├── good/
    │   └── error/
    ├── pill/
    │   ├── good/
    │   └── error/
    └── toothbrush/
        ├── good/
        └── error/
```

> 程式目前會讀取 `*.png` 檔案，因此請確認資料格式與路徑一致。

---

## 使用套件

主要使用以下 Python 套件：

- `torch`
- `torchvision`
- `numpy`
- `matplotlib`
- `Pillow`
- `scikit-learn`
- `torchsummary`

可先安裝：

```bash
pip install torch torchvision numpy matplotlib pillow scikit-learn torchsummary
```

---

## 執行方式

### 1. 開啟 Notebook
```bash
jupyter notebook vae.ipynb
```

### 2. 依序執行所有 cells
Notebook 會依序完成：

- 載入資料
- 建立 VAE 模型
- 訓練各類別模型
- 繪製 loss curve
- 在測試集上評估每個類別的準確率
- 計算整體平均準確率

---

## 重要參數

目前 notebook 中的主要設定如下：

```python
image_size = 256
latent_dim = 256
batch_size = 64
epochs = 150
optimizer = Adam(lr=9e-5 ~ 2.5e-4, weight_decay=1e-5)
```

補充說明：

- 不同類別的 learning rate 有微調
- 使用固定 random seed (`12`) 以提升可重現性
- 若有 GPU，程式會優先使用 CUDA；否則會改用 MPS 或 CPU

---

## 評估結果

本次 notebook 中各類別的結果如下：

| Category | Good Accuracy | Defective Accuracy | Overall Accuracy |
|---|---:|---:|---:|
| bottle | 100.00% | 77.78% | 83.13% |
| cable | 84.48% | 63.04% | 71.33% |
| capsule | 26.09% | 97.25% | 84.85% |
| pill | 69.23% | 77.30% | 76.05% |
| toothbrush | 100.00% | 80.00% | 85.71% |

**Overall Average Accuracy: 80.21%**

---

## License

若此專案需公開釋出，建議依需求補上對應授權條款（例如 MIT License）。
