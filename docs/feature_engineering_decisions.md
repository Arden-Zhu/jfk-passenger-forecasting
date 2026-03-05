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

**Decision:** Encode COVID as two binary dummies: `covid_acute` (Mar–Jun 2020)
and `covid_recovery` (Jul 2020–Jun 2022), with normal operations as the
baseline.

**Alternatives considered:**
- **Continuous recovery index (0→1):** Would capture the gradual ramp-up more
  precisely, but adds complexity without improving forecasting in the
  post-recovery period where predictions matter most. Also risks target
  leakage if the index is computed from historical throughput.

**Rationale:** Three levels are simple, interpretable, and sufficient. Models
additionally receive `month`, `day_of_week`, and lag features that implicitly
capture recovery dynamics within each regime.

---

## 3. Same-Weekday-Last-Year (not lag_365)

**Decision:** Replace a raw `lag_365` feature with `same_weekday_last_year`,
the mean throughput of the same weekday in the same month of the prior year.

**Example:** For Wednesday March 12, 2025, this feature equals the average
throughput across all Wednesdays in March 2024.

**Alternatives considered:**
- **Raw lag_365:** Uses the value exactly 365 days ago. Fragile: if that
  specific day was anomalous (e.g., a snowstorm), the feature is misleading.
  Also doesn't account for day-of-week shift (365 days ago is a different
  weekday).
- **Drop lag_365 entirely and cap at lag_28:** Preserves more rows (~2,309 vs
  ~1,972) but loses the year-over-year seasonality signal.

**Rationale:** `same_weekday_last_year` is more robust (averaged over multiple
days, not a single observation) and correctly aligns with day-of-week patterns,
which EDA showed are the strongest seasonal signal. The ~365 row loss is
acceptable; 1,972 remaining observations is sufficient for all six models.

**Implementation note:** This feature still requires 12 months of prior data,
so the first year of observations (~365 rows) is lost.

---

## 4. SARIMAX Uses a Separate Feature Set

**Decision:** SARIMAX receives only exogenous features (calendar, holidays,
weather, scheduled_departures, COVID dummies). Lag, rolling, and differencing
features are excluded.

**Rationale:** SARIMAX has built-in autoregressive (AR), moving average (MA),
and differencing (I) components that serve the same purpose as our engineered
time series features. Including them would be redundant and could cause
multicollinearity or confuse the model's internal parameter estimation.

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
- Raw integers (Monday=0, Sunday=6) imply an ordering that doesn't exist:
  Sunday is not "higher" than Monday.
- Sin/cos encoding (e.g., `sin(2π × day_of_week / 7)`) preserves the circular
  nature: Sunday and Monday are adjacent, December and January are adjacent.
  This is especially helpful for linear models and LSTM.

---

## 6. Drop Quarter (redundant with month encoding)

**Decision:** Drop `quarter` entirely rather than keeping it as a raw integer
or cyclical-encoding it.

**Alternatives considered:**
- **Keep as raw integer (1–4):** Implies Q4 > Q1 to linear models, which is
  meaningless. Tree-based models handle it fine, but it adds no signal as they
  can already get from month encoding.
- **Cyclical encode (sin/cos):** Correct but adds 2 columns that mostly
  duplicate the finer-grained `month_sin`/`month_cos`.

**Rationale:** `month_sin`/`month_cos` already capture seasonal patterns at
monthly granularity. Quarter is a strictly coarser version of the same signal.
Any split a tree makes on quarter, it can make on month. Dropping it
reduces dimensionality by 1 with no information loss.

---

## 7. Two-Scale Holiday Encoding (long weekends + travel periods)

**Decision:** Encode holidays at two temporal scales:
- `is_long_weekend`: 3-day window (Sat–Mon or Fri–Sun) for Monday/Friday
  federal holidays (MLK, Presidents', Memorial, Labor, Columbus Day).
- `period_*` flags: multi-day travel windows for major holidays:
  Thanksgiving (Tue–Sun, 6 days), Christmas/New Year (Dec 20–Jan 2, 14 days),
  July 4th (Jul 1–7, 7 days), Spring Break (Mar 15–Apr 15, 32 days).

**Rationale:** Airport demand does not spike on the holiday alone; it spreads
across a travel window whose width varies by holiday. Minor holidays create
clean 3-day bumps; major holidays produce week-long or multi-week surges.
A single `is_holiday` binary cannot capture this. The two scales are
complementary: `is_long_weekend` handles minor holidays;
`period_*` flags handle major ones.

---

## 8. Rolling Windows Use shift(1) to Prevent Leakage

**Decision:** Shift throughput by 1 row before computing rolling statistics,
so the window for row `t` covers days `t-w` through `t-1`.

**Rationale:** Without the shift, `rolling(7).mean()` at row `t` would include
day `t`'s own throughput, leaking the target into the features.

---

## 9. Forward-Fill Weather NaNs

**Decision:** Fill missing weather values (~29 rows for wind speed, <1% of
data) with forward-fill then backward-fill.

**Rationale:** Weather is temporally smooth; adjacent days share similar
conditions. Forward-fill is the standard imputation for sparse gaps in
time series sensor data.

---

## 10. No Imputation for Missing Scheduled Departures

**Decision:** The 2 missing `scheduled_departures` rows (Dec 30-31, 2018)
are not imputed. They fall in the first year, which is already dropped by
the `same_weekday_last_year` feature (requires 12 months of prior data).

---

## 11. Impute 8 Missing TSA Throughput Days (Notebook 01)

**Decision:** Reindex the daily TSA series to a gap-free date range and
impute 8 missing days using same-weekday-last-week (shift-7).

**Context:** The TSA data has 8 gaps: 2022-07-02 (1 day) and 2024-11-17
to 2024-11-23 (7 days). These are TSA FOIA PDF pipeline failures, not
airport closures. JFK operated normally on all 8 days.

**Rationale:** SARIMAX requires a gap-free time series. Lag and rolling
features also compute incorrectly across date gaps. Same-weekday-last-week
aligns with the strongest seasonal pattern in the data (day-of-week).

**Processed in:** `notebooks/01_data_preprocessing.ipynb`, before the
weather and flight merges.


