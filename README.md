# Bubble Morphology Analysis & ML Prediction System

A computer vision and machine learning pipeline for characterizing bubbles in multiphase flow experiments. The system extracts morphological features from dual-view 2D images, predicts bubble Volume and Surface Area using a stacking ensemble model, and derives secondary fluid-mechanics parameters.

## Pipeline Overview

```
JPG images (images/)
    ↓
3D_reconstruction.ipynb  →  bubble_data.csv
    ↓
voting_ensemble.ipynb    →  models/*.pkl
    ↓
secondary_parameters.ipynb  →  results/
```

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Usage

Run the notebooks in order:

1. **`3D_reconstruction.ipynb`** — Loads raw images, extracts bubble contours, reconstructs 3D ellipsoids from dual orthogonal views, and exports features to `bubble_data.csv`.

2. **`voting_ensemble.ipynb`** — Trains a stacking ensemble (LinearRegression, RandomForest, GradientBoosting, SVR, XGBoost) to predict bubble Volume and Surface Area. Serializes trained models and preprocessing objects to `models/`.

3. **`secondary_parameters.ipynb`** — Loads trained models and predicts secondary physical parameters: Sauter mean diameter, rise velocity (Tomiyama correlation), and interfacial area concentration (IAC).

## Project Structure

```
├── utils.py                    # Core image processing and feature extraction functions
├── bubble_data.csv             # Training dataset (16 features + Volume/Surface Area targets)
├── images/                     # Input JPG images (dual-view pairs per frame)
├── models/                     # Serialized models, scalers, and feature selectors
├── results/                    # Output visualizations (HTML 3D plots, PNG charts)
├── requirements.txt
└── notebooks:
    ├── 3D_reconstruction.ipynb
    ├── voting_ensemble.ipynb
    └── secondary_parameters.ipynb
```

## Feature Set

Each bubble is described by 15 morphological and texture features extracted from its contour:

| Feature | Description |
|---------|-------------|
| `A` | Area (mm²) |
| `d_eq` | Equivalent diameter (mm) |
| `d_maj`, `d_min` | Major / minor axis lengths (mm) |
| `AR` | Aspect ratio |
| `C` | Circularity |
| `E` | Eccentricity |
| `S` | Solidity |
| `El` | Elongation |
| `FDR` | Feret diameter ratio |
| `Hu1` | First Hu moment |
| `IR` | Inertia ratio |
| `mu_I` | Mean grayscale intensity |
| `sigma_I2` | Intensity variance |
| `edge_mean` | Mean Sobel edge gradient |

Spatial calibration: **0.074 mm/pixel**.

## ML Model

- **Architecture:** `StackingRegressor` with 5 base estimators and a `LinearRegression` meta-learner
- **Preprocessing:** RFE (8 features per target) → `StandardScaler` → `PolynomialFeatures(degree=2, interaction_only=True)`
- **Targets:** Log-transformed Volume (mm³) and Surface Area (mm²)
- **Tuning:** `RandomizedSearchCV` (50 iterations, 5-fold CV)
