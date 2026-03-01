# Work Breakdown — Task List & Deliverables

All tasks needed to complete the project. Assign team members and dates after
reviewing.

---

## Phase 1: Data & Feature Engineering

| ID | Task | Owner | Deliverables | Depends on |
|----|------|-------|-------------|------------|
| 1.0 | Data acquisition | Hao | `data/raw/` — TSA throughput CSVs, NOAA weather CSV, BTS on-time ZIPs; `src/data/download_bts_flights.py`, `download_all.bat` | — |
| 1.1 | Data preprocessing & EDA | Hao | `notebooks/01_data_preprocessing.ipynb`, `data/processed/jfk_daily_merged.csv`, 10 EDA figures | 1.0 |
| 1.2 | Feature engineering | Hao | `notebooks/02_feature_engineering.ipynb`, `data/processed/jfk_modeling_ready.csv` | 1.1 |

**Status:** 1.0 and 1.1 are done. 1.2 is next.

---

## Phase 2: Modeling

| ID | Task | Owner | Deliverables | Depends on |
|----|------|-------|-------------|------------|
| 2.1 | Naive baseline (shift-7) | | Predictions, metrics; anchor for all model comparisons | 1.2 |
| 2.2 | SARIMAX | | Trained model, predictions, metrics, training time, standardized coefficients for RQ4 | 1.2 |
| 2.3 | Ridge Regression | | Trained model, predictions, metrics, training time, standardized coefficients for RQ4 | 1.2 |
| 2.4 | Random Forest | | Trained model, predictions, metrics, training time, `feature_importances_` for RQ4 | 1.2 |
| 2.5 | XGBoost | | Trained model, predictions, metrics, training time, `feature_importances_` for RQ4 | 1.2 |
| 2.6 | SVR (RBF kernel) | | Trained model, predictions, metrics, training time, permutation importance for RQ4 | 1.2 |
| 2.7 | LSTM (optional) | | Trained model, predictions, metrics, training time, permutation importance or gradient attribution for RQ4 | 1.2 |

**Notes:**
- Each model must use TimeSeriesSplit CV for hyperparameter tuning.
- Each model must save predictions as a CSV with columns:
  `date, y_true, y_pred, model_name` to `results/predictions/`. This allows
  Phase 3 to load all predictions from one folder and compare directly.
- SARIMAX uses feature set A (exogenous only). All others use feature set B
  (all features).

---

## Phase 3: Analysis & RQ Answers

| ID | Task | Owner | Deliverables | Depends on |
|----|------|-------|-------------|------------|
| 3.1 | RQ1 — Prediction accuracy | | Best MAPE reported, comparison against 10% hypothesis | All of Phase 2 |
| 3.2 | RQ2 — Model comparison | | Comparison table (6 models × 4 metrics × 2 test sets), training time comparison, winner identified | All of Phase 2 |
| 3.3 | RQ3 — Ablation study | | 6 configs × 4 metrics table, marginal contribution of each feature group | Best model from 3.2, plus 1.2 |
| 3.4 | RQ4 — Feature importance | | Cross-model ranking table, Spearman correlations, agreement analysis | Feature importances from 2.2–2.7 |

---

## Phase 4: Report & Presentation

| ID | Task | Owner | Deliverables | Depends on |
|----|------|-------|-------------|------------|
| 4.1 | Report — Introduction & Related Work | | Sections 1–2 of final report | Can start now |
| 4.2 | Report — Methods | | Section 3 (data, features, models, evaluation) | Can start now (from proposal) |
| 4.3 | Report — Results & Discussion | | Sections 4–5 (RQ answers, interpretation, limitations) | 3.1–3.4 |
| 4.4 | Report — Final assembly & polish | | Complete report, abstract, references, formatting | 4.1–4.3 |
| 4.5 | Presentation slides | | Slide deck for class presentation | 4.4 |
| 4.6 | Code cleanup & README | | Final README, clean notebooks, reproducibility check | All above |

---

## Summary: Critical Path

```
1.0 (done) → 1.1 (done) → 1.2 → [2.1–2.7 in parallel] → [3.1–3.4] → [4.3–4.6]
                                                            4.1–4.2 can start now
```

**Bottleneck:** 1.2 (feature engineering). Nothing in Phase 2 can start
without it.