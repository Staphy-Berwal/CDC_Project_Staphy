# Multimodal House Price Prediction

This repository contains a reproducible multimodal machine-learning pipeline that predicts residential property prices by combining structured tabular features with satellite imagery embeddings. The project demonstrates data acquisition, feature engineering, model training, evaluation, and explainability (Grad-CAM).

---

## Repository structure

```
.
├── data_fetcher.ipynb
├── preprocessing.ipynb
├── model_training.ipynb
├── gradCAM.ipynb
└── README.md
```

---

## Quick overview

This project develops a multimodal regression framework to predict residential property prices by combining:
* Structured housing attributes (size, quality, location, etc.)
* Satellite imagery embeddings capturing neighborhood context such as greenery, road connectivity, and urban density

A pretrained Convolutional Neural Network (CNN) is used to extract high-dimensional visual features from satellite images, which are then fused with tabular data and used to train an XGBoost regression model.

The project also incorporates model explainability using Grad-CAM, enabling visual interpretation of which regions in satellite images influence price predictions.

---

## Setup

1. Clone the repository:

```bash
git clone <https://github.com/Staphy-Berwal/CDC_Project_Staphy.git>
cd CDC_Project_Staphy
```

2. Create a Python environment (recommended with conda):

```bash
conda create -n multimodal python=3.9 -y
conda activate multimodal
```

3. Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
pip install torch torchvision pillow opencv-python
```

4. API Configuration (Mapbox):

* Create a Mapbox account and get an access token.
* Add your token in `data_fetcher.ipynb`
```bash
MAPBOX_API_KEY = "your_api_key_here"
```

---

## Notebooks

### `data_fetcher.ipynb`
* Fetches satellite images using Mapbox Static Images API
* Uses latitude & longitude from the dataset
* Passes images directly through a pretrained CNN (ResNet)
* Extracts 512-dimensional image embeddings
* Outputs a CSV file:
  original tabular data + image embeddings


### `preprocessing.ipynb`

* Data cleaning and validation
* Missing value and duplicate checks
* Exploratory Data Analysis (EDA)
* Univariate & bivariate analysis
* Baseline model training using tabular data only
* Log transformation of target variable (`price`)

### `model_training.ipynb`

* Loads dataset containing tabular features + image embeddings
* Applies PCA to reduce image embeddings dimensionality
* Feature fusion (tabular + PCA-reduced image features)
* Trains XGBoost Regressor
* Evaluates performance using: RMSE, MAE, R² score
* Generates predictions on the test dataset
* Outputs final submission-ready CSV

### `gradCAM.ipynb`

* Implements Grad-CAM on the CNN image encoder
* Visualizes regions in satellite images influencing learned features
* Highlights:
  * Road networks
  * Green spaces
  * Housing density
* Ensures model explainability

---

## Recommended pipeline details

* **Image size:** 224×224 (or 256) for ResNet input.
* **CNN:** ResNet-18 pretrained on ImageNet; remove final fc layer and use global pooled features.
* **Embedding dim:** 512 (ResNet-18); apply PCA → 50–150 dims for tree models.
* **Regressor:** XGBoost with `n_estimators=500`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`.
* **Target:** use `log1p(price)`
---

## Explainability & Evaluation

* **Grad-CAM** for CNN attention maps; interpret which satellite regions drive embeddings.
* Compare metrics: RMSE, MAE, and R² for tabular-only vs multimodal.

---

## Notes & best practices

* Cache embeddings after extraction to avoid re-calling the API and incurring costs.
* Freeze CNN weights when extracting embeddings; do not train the CNN unless you have a large labeled dataset.
* Keep a clear train/validation/test split; do not fit PCA on the test set.
* Use Grad-CAM images and SHAP plots in the report for robust explainability.
