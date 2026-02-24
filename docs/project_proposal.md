# Project Proposal: Forecasting Daily Passenger Throughput at JFK International Airport

## 1. Project Narrative & Motivation

JFK International Airport is the busiest international gateway in North America,
processing over 63 million passengers in 2024. Accurate daily passenger
forecasts are essential for TSA staffing, gate allocation, and operational
planning.

This project forecasts **daily departing passenger throughput** at JFK, measured
by TSA security checkpoint screening counts (TSA screens only departing
passengers). We combine three public data sources: TSA FOIA throughput records
(~2,300 daily observations), NOAA weather data, and BTS scheduled departure
counts. The dataset spans Dec 2018 – May 2025, covering the COVID-19 collapse,
recovery, and return to record volumes.

Our central contribution is a **head-to-head comparison of classical ML methods
against deep learning** on a real-world time series problem, providing empirical
evidence for which approach is better suited for airport operational forecasting
at this data scale.

## 2. Prior Work

Zachariah et al. (2024) compared neural network models (RNN, LSTM, GRU) against
ARIMA/SARIMA/SARIMAX for short-term forecasting of daily TSA checkpoint
passenger flows at five major U.S. airports during the pandemic, finding that
RNN outperformed SARIMA by 34% at Atlanta's airport.
([Journal of Air Transport Management, 2024](https://www.sciencedirect.com/science/article/abs/pii/S0969699723001680))

Yi & Guo (DHS/TSA research) applied polynomial regression to predict hourly and
daily TSA checkpoint throughput at LAX, demonstrating that ML-based scheduling
tools can improve TSA staffing decisions.
([EasyChair preprint](https://easychair.org/publications/paper/bk2D/download))

BTS (2024) used simple linear regression to estimate monthly passenger
enplanements from TSA screening counts, achieving R² > 0.99 on post-pandemic
data, confirming the strong link between TSA throughput and flight operations.
([BTS Technical Report](https://www.bts.gov/browse-statistical-products-and-data/preliminary-estimates/preliminary-estimates-enplanements-tsa))

Our project differs from prior work in three ways: (1) we incorporate three
external data sources (weather, flight schedules, holidays) rather than
throughput alone, (2) we compare six model families head-to-head including a
systematic ablation study, and (3) we provide cross-model feature importance
analysis to assess interpretability.

## 3. Research Questions

1. **How accurately can daily JFK passenger throughput be predicted using
   engineered time series features, weather data, and scheduled flight counts?**
   We hypothesize that combining autoregressive, calendar, weather, and flight
   features will achieve MAPE below 10% on post-COVID test data.

2. **Do tree-based ensemble methods (Random Forest, XGBoost) outperform
   classical statistical models (ARIMA/SARIMAX) and deep learning (LSTM) in
   terms of accuracy, robustness across metrics, and computational
   efficiency?** Given ~2,300 samples and rich engineered features, we expect
   gradient-boosted trees to deliver the best tradeoff. Models are compared
   across accuracy (MAE, RMSE, MAPE, R²), robustness (full vs post-recovery
   test set, and a temporal shift experiment training on post-recovery data
   only), and computational efficiency.

3. **What is the marginal predictive contribution of each feature group
   (calendar/holiday, weather, scheduled flights, COVID dummies) beyond
   autoregressive features alone?** We conduct a systematic ablation study, training the
   best-performing model under multiple feature configurations, to quantify how
   much each category of external information improves forecast accuracy and
   whether any feature group is redundant given the others.

4. **Do different model families (linear, tree-based, neural network) agree on
   which features are the most important predictors of daily throughput?** We
   compare feature importance rankings extracted from all models to assess
   whether the identified key drivers are consistent across architectures, or
   whether different models rely on fundamentally different signals.

## 4. Methods

### 4.1 Data

| Dataset | Source URL | Format | Records | Period | License |
|---------|-----------|--------|---------|--------|---------|
| TSA checkpoint throughput (JFK) | [mikelor/TsaThroughput (GitHub)](https://github.com/mikelor/TsaThroughput) | CSV | 2,337 daily | Dec 2018 – May 2025 | MIT |
| NOAA daily weather (JFK station) | [NOAA CDO (Station USW00094789)](https://www.ncdc.noaa.gov/cdo-web/datasets/GHCND/stations/GHCND:USW00094789/detail) | CSV | 3,319 daily | Jan 2017 – Feb 2026 | Public domain |
| BTS On-Time Performance (JFK) | [BTS TranStats](https://www.transtats.bts.gov/Tables.asp?DB_ID=120) | CSV (zipped) | 2,373 daily | Jan 2019 – Jun 2025 | Public domain |
| U.S. federal holidays | Python `holidays` library | generated | matching target period | MIT |

All datasets are freely downloadable with no login, application, or payment
required. After merging on date, the final modeling dataset contains **2,337
rows × 30 columns**.

### 4.2 Feature Engineering

| Category | Features |
|----------|----------|
| Calendar | Day of week, month, quarter, is_weekend (cyclical encoding) |
| Holiday  | is_holiday, is_long_weekend, holiday period flags (Thanksgiving, Christmas, July 4th, spring break) |
| Autoregressive | Lag 1, 7, 14, 28, 365 days |
| Rolling statistics | 7/14/30-day rolling mean, std, min, max |
| Differencing | Day-over-day change, week-over-week change, year-over-year change |
| Scheduled flights | scheduled_departures |
| Weather | Temperature (TAVG/TMAX/TMIN), precipitation, snowfall, wind speed, severe weather flags (fog, thunder, ice, etc.) |
| Event dummies | covid_acute (Mar–Jun 2020), covid_recovery (Jul 2020–Jun 2022); baseline = normal operations |

**Prediction framing:** We predict next-day throughput (day $t+1$) using
historical throughput up to day $t$ and day $t+1$'s scheduled departures and
weather forecast (both available in advance). Multi-step forecasts (7- or
30-day) can be produced recursively by feeding each prediction back as lag
input.

Detailed rationale for each feature engineering decision (e.g., departures-only,
COVID encoding, same-weekday-last-year vs lag_365, SARIMAX feature set
separation) is documented in
[`feature_engineering_decisions.md`](feature_engineering_decisions.md).

### 4.3 Models

| Category | Model | Rationale |
|----------|-------|-----------|
| Naive baseline | Seasonal Naive (shift-7) | Predict using same weekday last week; anchor for ML value |
| Statistical baseline | SARIMAX | Classical time series benchmark with seasonality |
| Classical ML | Ridge Regression | Linear baseline with regularization |
| Classical ML | Random Forest | Non-linear ensemble (bagging); robust, interpretable feature importance |
| Classical ML | XGBoost | Non-linear ensemble (boosting); often state-of-the-art on tabular data |
| Classical ML | SVR (RBF kernel) | Kernel-based non-linear regression |
| Deep Learning (optional) | LSTM | Recurrent neural network for sequence modeling |

### 4.4 Evaluation

- **Naive baseline:** A seasonal naive model (predict day $t+1$ using day
  $t+1-7$, i.e., same weekday last week) serves as the non-ML anchor. All ML
  models must demonstrably outperform this baseline to justify their complexity.
- **Split strategy:** Time-based train/test split (80/20); TimeSeriesSplit
  cross-validation for hyperparameter tuning.
- **Metrics:** MAPE (primary), MAE, RMSE, R².
- **COVID handling:** Models are trained on the full dataset (2,337 days) with
  three-level COVID dummies (covid_acute, covid_recovery, normal as baseline).
  Test-set metrics are reported for both the full test set and the
  post-recovery subset (Jul 2022 onward). We considered a continuous recovery
  curve but opted for dummies — simpler, no target leakage risk, and the
  continuous encoding does not improve post-recovery forecasting.
- **Temporal shift test (RQ2 robustness):** To assess whether COVID-era data
  helps or hurts forecasting performance, we compare two training
  configurations using the best-performing model: (a) default training on the
  full dataset, and (b) training on post-recovery data only (Jul 2022 – Jun
  2024), both tested on the same held-out period (Jul 2024 – May 2025). If
  the post-recovery-only model matches or exceeds the default, it suggests
  that COVID-era observations add noise rather than signal, with a practical
  implication that only recent data is needed for deployment.
- **Ablation study:** Train the best model under six configurations to answer
  RQ3: (1) autoregressive only, (2) + calendar/holiday, (3) + weather,
  (4) + scheduled_departures, (5) + COVID dummies, (6) all combined. Each
  group is added independently to the same baseline. Full design in
  [`rq3_ablation_design.md`](rq3_ablation_design.md).

### 4.5 Tools

Python (pandas, scikit-learn, XGBoost, statsmodels, pmdarima, PyTorch),
Jupyter Notebooks, Matplotlib/Seaborn. Environment managed with conda + uv.
Project follows the Cookiecutter Data Science template.

## 5. Preliminary Findings (from EDA)

See [`eda_findings.md`](eda_findings.md).
