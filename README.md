# Solar Energy Forecasting with Machine Learning
### A Hybrid CNN-BiLSTM and XGBoost Approach with SHAP Interpretability for Northern Nigeria

Final-year Electrical Engineering capstone project (Ahmadu Bello University, Zaria) forecasting hourly solar PV power output for three Northern Nigerian cities — Kano, Maiduguri, and Abuja — using 8 years (2016–2023) of NASA POWER data.

## What this project does

Solar forecasting models are typically evaluated purely on accuracy, with little scrutiny of *why* they produce the predictions they do. This project builds a hybrid deep learning + gradient boosting pipeline for the forecasting task itself, then uses SHAP (SHapley Additive exPlanations) to open up the model and identify which features are actually driving its predictions — treating interpretability as a first-class part of the pipeline rather than an afterthought.

**Architecture:** A CNN-BiLSTM network processes sequential weather/irradiance data (GHI, DHI, clearness index, temperature, humidity, wind speed) to produce an initial forecast, which is then passed as an additional feature into an XGBoost model that produces the final, corrected prediction.

## Key results

- The hybrid model was benchmarked against 5 baselines (standalone CNN, standalone LSTM, unidirectional CNN-LSTM, standalone XGBoost, CNN-BiLSTM) across all three cities using a daytime-masked metric suite (RMSE, MAE, MAPE, R², nRMSE, nMAE).
- SHAP TreeExplainer analysis on the XGBoost stage identified **GHI (Global Horizontal Irradiance)** and the **CNN-BiLSTM's own output** as the dominant drivers of the final prediction across all three cities — consistent with physical intuition, but confirmed empirically rather than assumed.
- Data was split strictly chronologically (2016–2021 train / 2022 validation / 2023 test) to avoid leakage and reflect realistic deployment conditions.

## Why this matters beyond solar forecasting

The core methodological question this project engages with — *does an explanation reflect the actual mechanism driving a model's output, or does it just sound plausible?* — is the same question at the heart of AI interpretability more broadly. Applying SHAP here meant treating feature attributions as testable claims about the model's behavior, not taking them at face value. That distinction, between a convincing explanation and a faithful one, is what drives my interest in interpretability and rigorous model evaluation as a research direction.
## Repo structure

```
notebooks/        Stage-by-stage Jupyter notebooks (data prep → feature engineering →
                   model training → baseline comparison → SHAP analysis)
plots/             SHAP summary plots and baseline comparison charts
requirements.txt   Python dependencies
```

## Tech stack

Python 3.13, TensorFlow, XGBoost, SHAP, scikit-learn, pandas, Jupyter Notebook

## Data source

[NASA POWER](https://power.larc.nasa.gov/) hourly meteorological and solar data, 2016–2023, for Kano (12.0°N, 8.5°E), Maiduguri (11.8°N, 13.2°E), and Abuja (9.1°N, 7.4°E).

## Status

Final-stage baseline comparison and reporting in progress as of August 2026.
