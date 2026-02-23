# RQ3 Ablation Study — Experimental Design

This document details the experimental design for answering Research Question 3:

> **What is the marginal predictive contribution of each feature group
> (calendar/holiday, weather, scheduled flights, COVID dummies) beyond
> autoregressive features alone?**

---

## Approach: Independent Feature Group Ablation

Each external feature group is added **separately** to the same baseline to
isolate its marginal contribution.

## Configurations

| Config | Features included | What it measures |
|--------|-------------------|------------------|
| **(1) Baseline** | Lags (1, 7, 14, 28) + same_weekday_last_year + rolling stats (7/14/30-day mean, std) | Forecast quality from historical patterns alone — no external data |
| **(2) + Calendar** | Baseline + day_of_week, month, quarter, is_weekend (cyclical encoded) + is_holiday, is_long_weekend, holiday period flags | Marginal value of calendar & holiday information |
| **(3) + Weather** | Baseline + TAVG, TMAX, TMIN, PRCP, SNOW, SNWD, AWND, WSF2, WSF5, WT01–WT09 | Marginal value of weather conditions |
| **(4) + Departures** | Baseline + scheduled_departures | Marginal value of flight schedule information |
| **(5) + COVID** | Baseline + covid_acute, covid_recovery | Marginal value of COVID regime indicators |
| **(6) All combined** | Baseline + calendar + weather + departures + COVID dummies | Full model with all available information |

## What the Results Reveal

| Comparison | Question answered |
|------------|-------------------|
| (2) vs (1) | Does knowing the day/holiday help beyond historical patterns? |
| (3) vs (1) | Does weather add predictive value? |
| (4) vs (1) | Do flight schedules add predictive value? |
| (5) vs (1) | Does COVID encoding help the model handle regime changes? |
| (6) vs (1) | Total gain from all external data combined |

## Evaluation

- **Primary metric:** MAPE (aligns with RQ1 hypothesis of < 10%)
- **Supporting metrics:** MAE, RMSE, R²
- **Test set:** Time-based 80/20 split; additionally report post-recovery
  subset metrics.
- **Validation:** TimeSeriesSplit cross-validation (consistent across all RQs).

---

*Referenced from project proposal §3.4.*
