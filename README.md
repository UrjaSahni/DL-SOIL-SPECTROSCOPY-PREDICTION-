🌱 Soil Spectroscopy Prediction
Deep Learning · Machine Learning · Hyperspectral Analysis

This project predicts five key soil properties using hyperspectral reflectance data combined with environmental tabular features.
It integrates chemometrics, ML models, 1D-CNNs, and a stacked ensemble for high-accuracy multi-target regression.

📌 Project Goals

Predict the following soil attributes using spectroscopy:

SOC (Soil Organic Carbon)

pH

Ca (Calcium)

P (Phosphorus)

Sand %

📂 Repository Structure
├── data/                       # Raw spectral + tabular data
├── notebooks/
│   └── DEEP_Learning_Project.ipynb
├── models/                     # Saved models (Autoencoder, CNN, Ensemble)
├── src/
│   ├── preprocessing.py        # SG filtering, scaling, PCA
│   ├── models.py               # All ML + DL architectures
│   ├── stacking.py             # Stacking ensemble
├── README.md
└── requirements.txt

🚀 Methods Overview
1️⃣ Traditional Machine Learning

PLS Regression — baseline chemometric model

LightGBM — trained on PCA-reduced spectra + statistical aggregates

2️⃣ Transfer Learning with 1D Autoencoder

Encoder:

Conv1D layers + MaxPooling

Compresses ~3500 spectral features → 128-dim latent space

Decoder:

Reconstructs spectra (used only during pretraining)

Transfer Step:

Decoder removed

Encoder frozen/fine-tuned + Dense regression head

3️⃣ Hybrid Deep Learning Model (Multi-Input)
Branch	Input	Architecture
A — 1D-CNN	Spectral data	3×Conv1D → GlobalAveragePooling
B — Dense Network	Tabular features	Dense → Dropout
Fusion	concat(A,B)	Dense → Output(5 targets)
4️⃣ Stacked Ensemble (Final Model)

Base (Level-0) Learners:

PLS

LightGBM

Hybrid CNN

Meta-Learner (Level-1):

Ridge Regression

➡️ Best-performing model in the project.

⚙️ Training Details

Frameworks: TensorFlow/Keras, Scikit-learn, LightGBM

Hardware: Google Colab (T4 GPU)

Cross-Validation: 5-Fold

Optimizer: Adam

Loss: MSE

Batch Size: 32

Epochs:

Autoencoder → 10

Hybrid CNN → 80 (Early Stopping)

📊 Evaluation
Primary Metric: MCRMSE

Mean Columnwise RMSE across all 5 targets.

Model Performance
Model	Score / Observation
Hybrid CNN	Loss ≈ 0.68
Stacked Ensemble	MCRMSE ≈ 0.438 (Best)
Improvement	~7% better than best single model

Visualization:
Predicted vs. Actual scatter plots show the ensemble gives the tightest fit around the y = x line.

🧠 Key Insights

Deep models need larger datasets — CNN alone underperforms with ~1157 samples.

Fusion of spectral + tabular data boosts accuracy.

Autoencoder reduces noise, stabilizes CNN training.

Stacking provides robust error correction across diverse models.

🔮 Future Enhancements

Spectral data augmentation (noise, shifts)

Attention layers for wavelength-level feature focus

Optuna hyperparameter tuning

Use larger public soil spectral libraries

👥 Contributors

Gautam (102215039)

Navneet (102215082)

Urja (102215084)

Gaureesh (102215127)

Mehak (102215163)

Subgroup: 4NC6
