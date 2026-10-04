# Solar Energy Forecasting with Machine Learning
### A Hybrid CNN-BiLSTM and XGBoost Approach with SHAP Interpretability for Northern Nigeria

Final-year Electrical Engineering capstone project (Ahmadu Bello University, Zaria) forecasting hourly solar PV power output for three Northern Nigerian cities — Kano, Maiduguri, and Abuja — using 8 years (2016–2023) of NASA POWER data.

## What this project does

Solar forecasting models are typically evaluated purely on accuracy, with little scrutiny of *why* they produce the predictions they do. This project builds a hybrid deep learning + gradient boosting pipeline for the forecasting task, benchmarks it against several baselines, and then uses SHAP (SHapley Additive exPlanations) to identify which features actually drive its predictions.

**Architecture:** A CNN-BiLSTM network processes a 24-hour window of weather/irradiance data to forecast the next hour of PV power. An XGBoost model then learns the CNN-BiLSTM's residual errors from meta-features (the last-timestep inputs, the CNN-BiLSTM prediction and a latent representation), and the final prediction is the CNN-BiLSTM output plus the predicted correction. The final model (**Hybrid v3**) adds rolling mean/std features and is tuned with Optuna.

**Target:** Hourly PV power is *simulated* for a 1 MW farm from NASA POWER irradiance and temperature (GHI × area × efficiency × temperature factor). It is not measured plant output.

## Pipeline

| Stage | Notebook | What it does |
|---|---|---|
| 1 | `Stage1_data_exploration` | First look at the raw NASA POWER CSVs |
| 2 | `Stage2_Data_Loading_Exploration_Cleaning` | Loads yearly files, cleans the -999 sentinels, exploratory plots |
| 3 | `Stage3_Feature_Engineering_Normalisation` | PV power simulation, cyclical time features, lags (`PV_Power_Lag1`, `PV_Power_Lag24`, `GHI_Lag1`), chronological split, MinMax scaling (fit on train only), 24-hour sequences |
| 4 | `Stage4_CNN_biLSTM_Model` | Per-city CNN-BiLSTM training |
| 5 | `Stage5_XGBoost_Residual_Correction` | XGBoost residual correction on top of the CNN-BiLSTM |
| 6 | `Stage6_SHAP_Explainability_Analysis`, `Stage6_v2` | SHAP TreeExplainer analysis (bar, beeswarm, dependence, waterfall plots) |
| 7 | `Stage7` | Baselines (CNN, LSTM, CNN-LSTM, standalone XGBoost) with multi-run averaging |
| 7b | `Stage7_Optuna_Tuning` | Hybrid v2/v3 feature sets and Optuna tuning of the hybrid and standalone XGBoost |
| 8 | `Stage8_Final_Reporting` | Final 6-model comparison, composite ranking, scatter and RMSE charts |

Data was split strictly chronologically (2016–2021 train / 2022 validation / 2023 test) to avoid leakage. All metrics are computed on daytime hours only (PV power > 0).

## Results (2023 test set, daytime only)

RMSE and MAE are in kW for the simulated 1 MW farm (lower is better). Full tables, including the other metrics and run-to-run standard deviations, are in [`results/FINAL_reported_comparison.csv`](results/FINAL_reported_comparison.csv).

| City | Model | RMSE | MAE | MAPE (%) | R² |
|---|---|---|---|---|---|
| Kano | Standalone XGBoost | **29.81** | 17.08 | 11.30 | **0.984** |
| Kano | Hybrid v3 (Optuna) | 30.25 | **16.89** | **9.14** | 0.984 |
| Kano | CNN-BiLSTM (base) | 37.16 | 26.00 | 18.79 | 0.975 |
| Kano | LSTM (5-run avg) | 32.75 | 21.24 | 11.52 | 0.981 |
| Maiduguri | Standalone XGBoost | 33.72 | 19.63 | 17.93 | 0.979 |
| Maiduguri | Hybrid v3 (Optuna) | **33.33** | **18.80** | **13.87** | **0.980** |
| Maiduguri | CNN-BiLSTM (base) | 54.17 | 41.49 | 46.29 | 0.947 |
| Maiduguri | LSTM (5-run avg) | 40.34 | 27.16 | 19.93 | 0.970 |
| Abuja | Standalone XGBoost | 31.12 | **18.87** | 12.40 | 0.981 |
| Abuja | Hybrid v3 (Optuna) | **31.06** | 18.99 | **11.76** | 0.981 |
| Abuja | CNN-BiLSTM (base) | 47.26 | 35.69 | 31.08 | 0.956 |
| Abuja | LSTM (5-run avg) | 33.90 | 22.11 | 15.20 | 0.977 |

Standalone CNN and unidirectional CNN-LSTM were weaker still (see the CSV).

**Key findings**

- The hybrid substantially improves on its CNN-BiLSTM base model in every city (RMSE reduced by roughly 19–38%).
- Against a tuned standalone XGBoost the hybrid is essentially **on par**: RMSE is marginally worse in Kano and marginally better in Maiduguri and Abuja, while MAPE is better in all three cities. The differences in RMSE are small.
- The very high R² values should be read with care: the target is simulated from the same irradiance inputs, and lagged PV power is among the features, so the task is easier than forecasting measured plant output.
- SHAP analysis of the XGBoost stage identifies **GHI** and the **CNN-BiLSTM's own prediction** as the dominant drivers of the final prediction in all three cities, consistent with physical intuition (see `plots/shap_*` and `results/shap_importance_summary.csv`).

## Repo structure

```
data/             NASA POWER hourly CSVs per city and year (Abuja, Kano, Maiduguri; 2016–2023)
notebooks/        Stage-by-stage Jupyter notebooks (see Pipeline above)
models/           Trained CNN-BiLSTM (.keras) and XGBoost (.pkl) models
plots/            EDA, training, SHAP and model comparison figures
results/          Metric tables (CSV) and cleaned per-city data
requirements.txt  Python dependencies
```

`data/`, `models/`, the large intermediate files (`*.npy` sequences and predictions under `results/`) and Jupyter checkpoints are git-ignored, so they exist locally but are not in this repository. The models and intermediates are regenerated by running the notebooks in order.

## Reproducing

1. `pip install -r requirements.txt`
2. Download the NASA POWER hourly data for each city and year (2016–2023) from [NASA POWER](https://power.larc.nasa.gov/) and save it as `data/<City>/<City>_<year>.csv`.
3. Run the notebooks in stage order (Stage 1 → 8). Stages 3 onward write the sequences, scalers, models and metric tables used by later stages.

## Tech stack

Python 3.13, TensorFlow, XGBoost, SHAP, Optuna, scikit-learn, pandas, Jupyter Notebook

## Data source

[NASA POWER](https://power.larc.nasa.gov/) hourly meteorological and solar data, 2016–2023, for Kano (12.0°N, 8.5°E), Maiduguri (11.8°N, 13.2°E), and Abuja (9.1°N, 7.4°E).

## Status

All eight stages are complete, including Optuna tuning and final reporting.
