# Feature Engineering Decisions Log

This document records non-obvious design decisions made during feature
engineering, with rationale for each. Referenced from the project proposal §3.2.

---

## 1. Scheduled Departures Only (not arrivals)

**Decision:** Use `scheduled_departures` as the sole flight feature. Drop
`scheduled_arrivals` and `total_scheduled_flights`.

**Rationale:**
- Our target variable (TSA throughput) counts only departing passengers.
  Arriving passengers do not pass through TSA screening.
- Connecting passengers (arrive then depart) are already counted in the
  departing flight's screening total.
- Departures and arrivals are nearly perfectly correlated (r ≈ 0.99).
  Including both introduces multicollinearity that degrades linear model
  (Ridge, SARIMAX) coefficient stability with no additional predictive signal.

---

## 2. Three-Level COVID Encoding (not continuous curve)

**Decision:** Encode COVID as two binary dummies — `covid_acute` (Mar–Jun 2020)
and `covid_recovery` (Jul 2020–Jun 2022) — with normal operations as the
baseline.

**Alternatives considered:**
- **Binary (is_covid True/False):** Too crude. April 2020 (1,640 pax/day) and
  January 2022 (60K pax/day) are both "True" but fundamentally different.
- **Continuous recovery index (0→1):** Would capture the gradual ramp-up more
  precisely, but adds complexity without improving forecasting in the
  post-recovery period where predictions matter most. Also risks target
  leakage if the index is computed from historical throughput.

**Rationale:** Three levels are simple, interpretable, and sufficient. Models
additionally receive `month`, `day_of_week`, and lag features that implicitly
capture recovery dynamics within each regime.

---

## 3. Same-Weekday-Last-Year (not lag_365)

**Decision:** Replace a raw `lag_365` feature with `same_weekday_last_year` —
the mean throughput of the same weekday in the same month of the prior year.

**Example:** For Wednesday March 12, 2025, this feature equals the average
throughput across all Wednesdays in March 2024.

**Alternatives considered:**
- **Raw lag_365:** Uses the value exactly 365 days ago. Fragile — if that
  specific day was anomalous (e.g., a snowstorm), the feature is misleading.
  Also doesn't account for day-of-week shift (365 days ago is a different
  weekday).
- **Drop lag_365 entirely and cap at lag_28:** Preserves more rows (~2,309 vs
  ~1,972) but loses the year-over-year seasonality signal.
- **Fill the first 365 NaN rows with a proxy (lag_28 or monthly mean):**
  Keeps all rows but introduces synthetic values that may mislead models.

**Rationale:** `same_weekday_last_year` is more robust (averaged over multiple
days, not a single observation) and correctly aligns with day-of-week patterns,
which EDA showed are the strongest seasonal signal. The ~365 row loss is
acceptable — 1,972 remaining observations is sufficient for all six models.

**Implementation note:** This feature still requires 12 months of prior data,
so the first year of observations (~365 rows) is lost. Shorter lag features
(lag_1, lag_7, lag_14, lag_28) only cost 28 rows, which is absorbed by this
larger loss.

---

## 4. SARIMAX Uses a Separate Feature Set

**Decision:** SARIMAX receives only exogenous features (calendar, holidays,
weather, scheduled_departures, COVID dummies). Lag, rolling, and differencing
features are excluded.

**Rationale:** SARIMAX has built-in autoregressive (AR), moving average (MA),
and differencing (I) components that serve the same purpose as our engineered
time series features. Including them would be redundant and could cause
multicollinearity or confuse the model's internal parameter estimation.

All other models (Ridge, RF, XGBoost, SVR, LSTM) require the full engineered
feature set because they treat each row independently and have no built-in
time series awareness.

**In practice, we maintain two feature sets:**
- **Set A (SARIMAX):** weather + calendar + holidays + COVID dummies +
  scheduled_departures
- **Set B (all other models):** Set A + lags + rolling stats + differencing +
  same_weekday_last_year

---

## 5. Cyclical Encoding for Calendar Features

**Decision:** Encode `day_of_week`, `month`, and `day_of_year` using sin/cos
transforms rather than one-hot encoding or raw integers.

**Rationale:**
- Raw integers (Monday=0, Sunday=6) imply an ordering that doesn't exist —
  Sunday is not "higher" than Monday.
- One-hot encoding works but creates 7+12 = 19 extra columns, increasing
  dimensionality.
- Sin/cos encoding (e.g., `sin(2π × day_of_week / 7)`) preserves the circular
  nature: Sunday and Monday are adjacent, December and January are adjacent.
  This is especially helpful for linear models and LSTM.

---

*This document is updated as new decisions arise during Notebooks 02–04.*
