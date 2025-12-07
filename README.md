# 🌱 Soil Spectroscopy Prediction  
### Deep Learning · Machine Learning · Hyperspectral Analysis

This project predicts **five key soil properties** using **hyperspectral reflectance data** along with environmental tabular features.  
It integrates **PLS**, **LightGBM**, **1D-CNNs**, **Autoencoders**, and a **Stacked Ensemble** for high-accuracy multi-target regression.

---

## 📌 Project Goals

Predict the following soil attributes from spectral data:

- SOC (Soil Organic Carbon)  
- pH  
- Ca (Calcium)  
- P (Phosphorus)  
- Sand (%)

---

## 🚀 Methods Overview

### 1️⃣ Traditional Machine Learning
- **PLS Regression** (baseline chemometrics)
- **LightGBM** (trained on PCA-reduced spectra + aggregates)

---

### 2️⃣ Transfer Learning with 1D Autoencoder

**Encoder:**  
Conv1D + MaxPooling → compresses ~3500 spectral features into a 128-dim latent vector  

**Decoder:**  
Reconstructs spectra (used only during pretraining)

**Transfer Step:**  
Decoder removed → Encoder frozen/fine-tuned → Dense regression head attached

---

### 3️⃣ Hybrid Deep Learning Model (Multi-Input)

| Branch | Input | Architecture |
|--------|--------|--------------|
| **A — 1D-CNN** | Spectral data | 3×Conv1D → GlobalAveragePooling |
| **B — Dense Net** | Tabular data | Dense → Dropout |
| **Fusion** | concat(A, B) | Dense → Output(5 targets) |

---

### 4️⃣ Stacked Ensemble (Final Model)

**Base Models (Level-0):**
- PLS  
- LightGBM  
- Hybrid CNN  

**Meta-Model (Level-1):**
- **Ridge Regression**

➡️ Achieves the best performance in this project.

---

## ⚙️ Training Details

- Frameworks: TensorFlow/Keras, Scikit-learn, LightGBM  
- Hardware: Google Colab (T4 GPU)  
- Cross-Validation: 5-Fold  
- Optimizer: Adam  
- Loss: MSE  
- Batch Size: 32  
- Epochs:  
  - Autoencoder → 10  
  - Hybrid CNN → 80 (with EarlyStopping)

---

## 📊 Evaluation

### Metric Used: **MCRMSE**  
Mean Columnwise RMSE across all five soil properties.

### Performance Summary

| Model | Score / Notes |
|-------|----------------|
| Hybrid CNN | Loss ≈ 0.68 |
| **Stacked Ensemble** | **MCRMSE ≈ 0.438 (Best)** |
| Improvement | ~7% better than best individual model |

Predicted vs. Actual scatter plots show the ensemble aligns closest to the **y = x** line.

---

## 🧠 Key Insights

- Deep models need larger datasets; CNN alone underperforms with ~1157 samples.  
- Fusing **spectral + tabular features** boosts performance.  
- Autoencoder denoises spectra → stabilizes CNN training.  
- Stacking captures complementary strengths of all models.

---

## 🔮 Future Enhancements

- Spectral data augmentation (noise, shifting)  
- Attention-based CNNs to focus on key wavelengths  
- Hyperparameter tuning with Optuna  
- Use larger soil spectral libraries for pretraining  

---

## 👥 Contributors

- Gautam (102215039)  
- Navneet (102215082)  
- Urja (102215084)  
- Gaureesh (102215127)  
- Mehak (102215163)  

_Subgroup: 4NC6_

