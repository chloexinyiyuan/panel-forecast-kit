# Retail Demand Forecasting — Store × Item Daily Sales

Forecasting daily sales at the store–item level to support inventory and
replenishment decisions. A full pipeline from EDA through feature engineering,
model comparison, time-series cross-validation, and error analysis.

**Data:** Kaggle *Store Item Demand Forecasting Challenge* (10 stores × 50 items,
daily sales, 2013–2017).

---

## Key Results

- **XGBoost (Poisson objective) achieves WAPE 11.95%** (MAE 6.32, RMSE 8.24),
  evaluated with **4-fold rolling time-series cross-validation**.
- **~25% lower WAPE than the best naive baseline** (7-day moving average, 15.93%;
  last-week-same-day, 16.60%).
- **Ensembling gave no stable gain** across CV folds, so a single interpretable
  XGBoost was chosen over stacked models.
- **Feature importance cross-validated by two independent methods** (ElasticNet
  and XGBoost gain) consistently shows lag/rolling features dominate.
- **Error analysis**: forecast accuracy is driven by sales volume — high-volume
  items are predicted reliably (WAPE ~9%), low-volume items are inherently
  harder (WAPE ~18%); store-level differences are minor.

---

## Project Structure
notebooks/
├── 01_Preview.ipynb # Data loading & overview
├── 02_EDA.ipynb # Exploratory analysis & business insights
├── 03_features.ipynb # Feature engineering (calendar, lag, rolling)
├── 04_models.ipynb # Model training & comparison
└── 05_benchmark.ipynb # Rolling-window CV & error analysis


---

## Approach & Key Findings

**EDA.** Sales show a clear upward trend (2013→2017), strong yearly seasonality
(summer peaks), and a weekend effect (Sunday ≈ +50% vs Monday). A month × weekday
heatmap revealed an interaction: summer amplifies the weekday pattern.

**Feature engineering.** Calendar features (day-of-week, month, is_weekend),
lag features (7/14/28/364 days), and rolling means (7/14/30 days). The 364-day
lag (= 52 weeks) aligns year-over-year on the same weekday.

**Model selection.** Linear models (Ridge/ElasticNet) underperformed the naive
baseline, confirming non-linear/interaction effects and motivating tree models.
XGBoost, LightGBM, and CatBoost performed comparably; XGBoost was chosen for
maturity and interpretability. A **Poisson objective** was used to match the
non-negative count nature of sales.

**Evaluation.** 4-fold rolling-origin time-series CV (train always precedes
validation). Accuracy improved as the training window grew (WAPE 13.9% → 11.0%),
indicating more history yields better forecasts.

**Error analysis → business value.** Prediction accuracy is governed by item
sales volume, not store. This suggests a differentiated inventory policy:
replenish high-volume items tightly to forecast, keep larger safety buffers for
low-volume long-tail items.

---

## How to Run

```bash
# 1. Create environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Download data from Kaggle and place train.csv in data/raw/
#    https://www.kaggle.com/c/demand-forecasting-kernels-only

# 3. Run notebooks in order (01 → 05)
```

---

## Limitations & Future Work

- **Forecast scenario.** The model relies on recent lag features (A-scenario:
  rolling daily forecasts). Long-horizon forecasting (e.g. a full quarter, as in
  the Kaggle 2018 test set) is a harder B-scenario where recent lags are
  unavailable and the model must lean on yearly seasonality (lag_364).
- **Kaggle submission** (2018 Q1, B-scenario) is planned as a follow-up ⟨add
  Kaggle link once submitted⟩.
- External signals (promotions, weather, holidays) could add information
  independent of sales history.

---

## Links
- GitHub: https://github.com/你的用户名/panel-forcast-kit
#- Kaggle submission: ⟨to be added⟩