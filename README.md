---
title: WineScore
emoji: 🍷
colorFrom: red
colorTo: purple
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
license: mit
---

# 🍷 WineScore — Wine Quality Predictor

Predict the quality of a wine from its chemical properties using a GridSearchCV-tuned Random Forest model trained on the UCI Vinho Verde dataset.

## Features
- **Live quality prediction** (Poor / Average / Great) with confidence scores
- **SHAP explainability** — see exactly which chemical properties drove the prediction
- **Chemical profile radar** — visual overview of the wine's signature
- **Improvement suggestions** — which properties to adjust and in which direction

## Dataset
UCI Machine Learning Repository — [Wine Quality Data Set](https://archive.ics.uci.edu/ml/datasets/wine+quality)  
1,599 red wines + 4,898 white wines from Vinho Verde, Portugal.

## Model
- Algorithm: `RandomForestClassifier` wrapped in a `sklearn.Pipeline` with `StandardScaler`
- Tuning: 5-fold `GridSearchCV` on n_estimators, max_depth, max_features
- Target: 3-class quality label (Poor: 1–4, Average: 5–6, Great: 7–10)
- Explainability: `shap.TreeExplainer`
