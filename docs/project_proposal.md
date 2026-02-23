# Project Proposal: Forecasting Daily Passenger Throughput at JFK International Airport

## 1. Project Narrative & Motivation

John F. Kennedy International Airport is the busiest international gateway in
North America, processing over 63 million passengers in 2024. Efficient airport
operations — from staffing TSA security checkpoints to allocating gate resources
— depend critically on accurate short-term passenger volume forecasts. A
single-day staffing miscalculation can cascade into hour-long security lines,
missed flights, and millions of dollars in operational costs.

This project builds a machine learning pipeline to forecast **daily departing
passenger throughput** at JFK Airport, measured by TSA security checkpoint
screening counts. TSA checkpoints screen only departing passengers (arriving
passengers bypass security), so the target variable represents the total number
of outbound travelers processed across all terminals each day. We combine three
public data sources: TSA FOIA throughput records (2019–2025, ~2,300 daily
observations), NOAA weather observations recorded at the JFK weather station,
and BTS scheduled departure counts aggregated from per-flight on-time
performance records. The dataset spans a uniquely turbulent
period in aviation history — encompassing the COVID-19 pandemic collapse, the
subsequent recovery, and the return to record-breaking passenger volumes —
providing a rigorous test bed for time series forecasting methods.

Our central contribution is a **head-to-head comparison of classical machine
learning methods against deep learning** on a real-world time series problem.
While deep learning dominates recent literature, classical methods (Random
Forest, XGBoost, SVR) often outperform neural networks on small-to-medium
tabular datasets. This project provides empirical evidence for which approach
is better suited for airport operational forecasting.

## 2. Research Questions

1. **How accurately can daily JFK passenger throughput be predicted using
   engineered time series features, weather data, and scheduled flight counts?**
   We hypothesize that models incorporating lag features (e.g.,
   same-day-last-week, same-day-last-year), calendar features (day-of-week,
   holidays), weather conditions, and scheduled flight counts will achieve MAPE
   below 10% on post-COVID test data.

2. **Do tree-based ensemble methods (Random Forest, XGBoost) outperform
   classical statistical models (ARIMA/SARIMAX) and deep learning (LSTM) for
   this forecasting task?** Given the moderate dataset size (~2,300 samples) and
   the availability of rich engineered features, we expect gradient-boosted trees
   to deliver the best accuracy-to-complexity tradeoff.

3. **What is the marginal predictive contribution of each feature group
   (calendar/holiday, weather, scheduled flights) beyond autoregressive
   features alone?** We conduct a systematic ablation study, training the
   best-performing model under multiple feature configurations, to quantify how
   much each category of external information improves forecast accuracy and
   whether any feature group is redundant given the others.

4. **Do different model families (linear, tree-based, neural network) agree on
   which features are the most important predictors of daily throughput?** We
   compare Ridge regression coefficients, Random Forest/XGBoost feature
   importance rankings, and LSTM gradient-based attribution to assess whether
   the identified key drivers are consistent across model architectures, or
   whether different models rely on fundamentally different signals.

## 3. Methods

### 3.1 Data

| Dataset | Records | Period | Role |
|---------|---------|--------|------|
| TSA checkpoint throughput (JFK) | 2,337 daily | Dec 2018 – May 2025 | Target variable (departing passengers screened) |
| NOAA daily weather (JFK station) | 3,319 daily | Jan 2017 – Feb 2026 | Weather features |
| BTS On-Time Performance (JFK) | 2,373 daily | Jan 2019 – Jun 2025 | Scheduled departure counts |
| U.S. federal holidays | generated | matching target period | Calendar features |

After merging on date, the final modeling dataset contains **2,337 rows × 30
columns**.

### 3.2 Feature Engineering

| Category | Features |
|----------|----------|
| Calendar | Day of week, month, quarter, is_weekend (cyclical encoding) |
| Holiday  | is_holiday, is_long_weekend, holiday period flags (Thanksgiving, Christmas, July 4th, spring break) |
| Autoregressive | Lag 1, 7, 14, 28, 365 days |
| Rolling statistics | 7/14/30-day rolling mean, std, min, max |
| Differencing | Day-over-day change, week-over-week change, year-over-year change |
| Scheduled flights | scheduled_departures (see note below) |
| Weather | Temperature (TAVG/TMAX/TMIN), precipitation, snowfall, wind speed, severe weather flags (fog, thunder, ice, etc.) |
| Event dummies | covid_acute (Mar–Jun 2020), covid_recovery (Jul 2020–Jun 2022); baseline = normal operations |

Detailed rationale for each feature engineering decision (e.g., departures-only,
COVID encoding, same-weekday-last-year vs lag_365, SARIMAX feature set
separation) is documented in
[`docs/feature_engineering_decisions.md`](docs/feature_engineering_decisions.md).

**Note on flight feature selection:** The BTS dataset provides both scheduled
departures and scheduled arrivals. Since our target variable (TSA throughput)
counts only departing passengers, we use **scheduled_departures only**. Arriving
passengers do not pass through TSA screening. While some arriving passengers
connect to outbound flights, they are already captured in the departing flight's
screening count. Including both would introduce multicollinearity
(r ≈ 0.99 between departures and arrivals) with no additional information,
degrading the stability of linear models (Ridge, SARIMAX) for no predictive
gain.

### 3.3 Models

| Category | Model | Rationale |
|----------|-------|-----------|
| Statistical baseline | SARIMAX | Classical time series benchmark with seasonality |
| Classical ML | Ridge Regression | Linear baseline with regularization |
| Classical ML | Random Forest | Non-linear ensemble (bagging); robust, interpretable feature importance |
| Classical ML | XGBoost | Non-linear ensemble (boosting); often state-of-the-art on tabular data |
| Classical ML | SVR (RBF kernel) | Kernel-based non-linear regression |
| Deep Learning (optional) | LSTM | Recurrent neural network for sequence modeling |

### 3.4 Evaluation

- **Split strategy:** Time-based train/test split (80/20); TimeSeriesSplit
  cross-validation for hyperparameter tuning (no data leakage).
- **Metrics:** MAE, RMSE, MAPE, R².
- **COVID handling:** All models are trained on the full dataset (2,337 days)
  with three-level COVID dummy variables (covid_acute, covid_recovery, and
  normal as the baseline). To provide a fair assessment of real-world
  forecasting performance, we additionally report test-set metrics restricted
  to the post-recovery period (Jul 2022 onward, ~1,060 days of normal
  operations).
- **Design note on COVID encoding:** We considered a continuous recovery curve
  (e.g., a normalized index from 0 at lockdown trough to 1 at full recovery)
  to capture the gradual ramp-up. We opted against it: the continuous encoding
  adds model complexity without improving forecasting ability in the
  post-recovery period where predictions matter most, and computing it from
  historical throughput risks leaking target information into features.
  Three-level dummies are simpler, interpretable, and sufficient.
- **Ablation study:** Train the best model under five feature configurations to
  answer RQ3:
  (1) autoregressive only,
  (2) + calendar/holiday,
  (3) + weather,
  (4) + scheduled_departures,
  (5) all features combined.

### 3.5 Tools

Python (pandas, scikit-learn, XGBoost, statsmodels, pmdarima, PyTorch),
Jupyter Notebooks, Matplotlib/Seaborn. Environment managed with conda + uv.
Project follows the Cookiecutter Data Science template.

## 4. Preliminary Findings (from EDA)

- The merged dataset contains **2,337 daily observations** (Dec 2018 – May
  2025) with 30 columns after joining TSA throughput, NOAA weather, and BTS
  scheduled flights.
- Daily throughput ranges from **1,640** (COVID trough, April 2020) to
  **114,397** (peak summer 2024), with a mean of ~70,000 and a standard
  deviation of ~29,000.
- **Day-of-week seasonality:** Thursday and Sunday are consistently the
  busiest days (travel departure/return days); Tuesday and Wednesday are the
  lowest.
- **Monthly seasonality:** July–August peak; January–February trough.
- **Terminal closures:** Terminal 2 closed ~2023 and Terminal 7 closed ~2024,
  visible in the stacked area chart of terminal-level throughput.
- **Weather correlation:** Temperature shows moderate positive correlation with
  throughput (r ≈ 0.3); snowfall and precipitation are weakly negative.
- **Flight correlation:** Scheduled departure counts are strongly correlated
  with throughput — likely the strongest single predictor available.
- **Passengers per departing flight:** Mean of 111.8 (median 114.5). This ratio
  collapsed to as low as 5.1 during COVID (near-empty planes), then recovered
  to pre-pandemic levels by 2023.
- **COVID-19:** 2020 average throughput dropped to ~25K/day (vs ~90K
  pre-COVID), an 85% decline. Full recovery was reached by 2023–2024. The
  dataset breaks down as: pre-COVID ~430 days (18%), acute collapse ~120 days
  (5%), recovery ~730 days (31%), and post-recovery ~1,060 days (45%).
- **Missing data:** 8 missing dates in TSA data; 2 days without flight data;
  ~1% missing wind speed values — all manageable.
