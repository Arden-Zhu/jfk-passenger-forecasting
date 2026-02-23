# RQ4 Feature Importance Comparison — Experimental Design

This document details the experimental design for answering Research Question 4:

> **Do different model families (linear, tree-based, neural network) agree on
> which features are the most important predictors of daily throughput?**

---

## Approach

Train all 6 models on the same feature set (Config 6: all features combined),
then extract and compare feature importance rankings across models.

## Feature Importance Extraction

| Model | Method |
|-------|--------|
| Ridge | Absolute standardized coefficients |
| Random Forest | Built-in `feature_importances_` (mean decrease in impurity) |
| XGBoost | Built-in `feature_importances_` (gain-based) |
| SVR | Permutation importance |
| SARIMAX | Exogenous variable coefficients (feature set A only) |
| LSTM | Permutation importance or gradient-based attribution |

**Note:** SARIMAX only uses feature set A (no lags/rolling), so its rankings
are compared only within the exogenous features it receives.

## Comparison Method

1. Rank features 1st, 2nd, 3rd... within each model.
2. Build a cross-model ranking table (features × models).
3. Compute **Spearman rank correlation** between each pair of models to
   quantify agreement.

## What the Results Reveal

| Outcome | Interpretation |
|---------|----------------|
| High agreement (Spearman > 0.8) | Key drivers are model-agnostic; findings are robust |
| Moderate agreement (0.5–0.8) | Models share top features but diverge on mid-tier ones |
| Low agreement (< 0.5) | Different architectures rely on different signals — discuss why |

## Relationship to Other RQs

- **Independent of RQ3.** RQ3 varies features with one model; RQ4 varies
  models with all features. They can run in parallel.
- **Builds on RQ2.** RQ2 identifies which model wins; RQ4 examines whether the
  winner's explanation generalizes across architectures.

---

*Referenced from project proposal §2, RQ4.*
