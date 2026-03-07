# Feature Data Dictionary

All features in `jfk_modeling_ready.csv`, grouped by ablation category.
Target variable: `daily_throughput` (daily TSA checkpoint screening count at JFK).

---

## Calendar & Holiday (13 features)

| Feature | Description | Source |
|---------|-------------|--------|
| `is_weekend` | 1 if Saturday or Sunday, 0 otherwise | Derived from date |
| `day_of_week_sin` | Sin component of day-of-week (period=7) | `sin(2pi * dayofweek / 7)` |
| `day_of_week_cos` | Cos component of day-of-week (period=7) | `cos(2pi * dayofweek / 7)` |
| `month_sin` | Sin component of month (period=12) | `sin(2pi * month / 12)` |
| `month_cos` | Cos component of month (period=12) | `cos(2pi * month / 12)` |
| `day_of_year_sin` | Sin component of day-of-year (period=365 or 366) | `sin(2pi * dayofyear / days_in_year)` |
| `day_of_year_cos` | Cos component of day-of-year (period=365 or 366) | `cos(2pi * dayofyear / days_in_year)` |
| `is_holiday` | 1 if US federal holiday (including observed), 0 otherwise | Python `holidays` library |
| `is_long_weekend` | 1 if part of a 3-day weekend from a Mon/Fri holiday | Derived from holiday dates |
| `period_thanksgiving` | 1 if Tue before through Sun after Thanksgiving (6 days) | Computed from Thanksgiving date |
| `period_christmas` | 1 if Dec 20 - Jan 2 (14 days) | Fixed calendar window |
| `period_july4` | 1 if Jul 1 - Jul 7 (7 days) | Fixed calendar window |
| `period_spring_break` | 1 if Mar 15 - Apr 15 (32 days) | Fixed calendar window |

---

## Weather (17 features)

| Feature | Description | Source |
|---------|-------------|--------|
| `TAVG` | Average daily temperature (F) | NOAA GHCND, station USW00094789 |
| `TMAX` | Maximum daily temperature (F) | NOAA GHCND |
| `TMIN` | Minimum daily temperature (F) | NOAA GHCND |
| `PRCP` | Daily precipitation (inches) | NOAA GHCND |
| `SNOW` | Daily snowfall (inches) | NOAA GHCND |
| `SNWD` | Snow depth on ground (inches) | NOAA GHCND |
| `AWND` | Average daily wind speed (mph) | NOAA GHCND |
| `WSF2` | Fastest 2-minute wind speed (mph) | NOAA GHCND |
| `WSF5` | Fastest 5-second wind speed (mph) | NOAA GHCND |
| `WT01` | 1 if fog, ice fog, or freezing fog observed | NOAA GHCND, binary |
| `WT02` | 1 if heavy fog or heavy freezing fog | NOAA GHCND, binary |
| `WT03` | 1 if thunder | NOAA GHCND, binary |
| `WT04` | 1 if ice pellets, sleet, or hail | NOAA GHCND, binary |
| `WT05` | 1 if hail | NOAA GHCND, binary |
| `WT06` | 1 if glaze or rime | NOAA GHCND, binary |
| `WT08` | 1 if smoke or haze | NOAA GHCND, binary |
| `WT09` | 1 if blowing snow | NOAA GHCND, binary |

---

## Flights (1 feature)

| Feature | Description | Source |
|---------|-------------|--------|
| `scheduled_departures` | Count of flights scheduled to depart JFK that day | BTS On-Time Performance, aggregated per day |

---

## COVID (2 features)

| Feature | Description | Source |
|---------|-------------|--------|
| `covid_acute` | 1 if Mar 15 - Jun 30, 2020 (lockdown period) | Fixed date window |
| `covid_recovery` | 1 if Jul 1, 2020 - Jun 30, 2022 (recovery period) | Fixed date window |

Both = 0 indicates normal operations (baseline).

---

## Autoregressive (19 features)

| Feature | Description | Source |
|---------|-------------|--------|
| `lag_1` | Throughput 1 day ago | `daily_throughput.shift(1)` |
| `lag_7` | Throughput 7 days ago (same weekday last week) | `daily_throughput.shift(7)` |
| `lag_14` | Throughput 14 days ago | `daily_throughput.shift(14)` |
| `lag_28` | Throughput 28 days ago | `daily_throughput.shift(28)` |
| `same_weekday_last_year` | Mean throughput of same weekday in same month, prior year | Grouped mean lookup |
| `roll_mean_7` | 7-day rolling mean (days t-7 to t-1) | `shift(1).rolling(7).mean()` |
| `roll_std_7` | 7-day rolling std | `shift(1).rolling(7).std()` |
| `roll_min_7` | 7-day rolling min | `shift(1).rolling(7).min()` |
| `roll_max_7` | 7-day rolling max | `shift(1).rolling(7).max()` |
| `roll_mean_14` | 14-day rolling mean (days t-14 to t-1) | `shift(1).rolling(14).mean()` |
| `roll_std_14` | 14-day rolling std | `shift(1).rolling(14).std()` |
| `roll_min_14` | 14-day rolling min | `shift(1).rolling(14).min()` |
| `roll_max_14` | 14-day rolling max | `shift(1).rolling(14).max()` |
| `roll_mean_30` | 30-day rolling mean (days t-30 to t-1) | `shift(1).rolling(30).mean()` |
| `roll_std_30` | 30-day rolling std | `shift(1).rolling(30).std()` |
| `roll_min_30` | 30-day rolling min | `shift(1).rolling(30).min()` |
| `roll_max_30` | 30-day rolling max | `shift(1).rolling(30).max()` |
| `diff_1` | Day-over-day throughput change | `daily_throughput.diff(1)` |
| `diff_7` | Week-over-week throughput change | `daily_throughput.diff(7)` |

All rolling features use `shift(1)` to exclude the current day (no leakage).

---

## Feature Set Summary

| Set | Used by | Features |
|-----|---------|----------|
| **A** | SARIMAX | Calendar/Holiday + Weather + Flights + COVID (33 features) |
| **B** | Ridge, RF, XGBoost, SVR, LSTM | Set A + Autoregressive (52 features) |
