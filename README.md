```markdown
# Mini Project 2 – Factor Model with PCA, OLS, Shrinkage & Global Minimum Variance

## Overview

This strategy uses a **multi‑factor model** trained offline on daily returns. The factors are extracted via **Principal Component Analysis (PCA)**. Asset‑specific factor loadings (betas) and alphas are estimated via **OLS regression**. A **Ledoit‑Wolf shrunk covariance matrix** provides a stable estimate of asset return covariances. At each rebalance (daily), the strategy computes current factor returns, then constructs a **global minimum variance (GMV) portfolio** projected to long‑only.

**Key statistical methods:**
- PCA (dimensionality reduction, 15 factors)
- Time‑series OLS regression (factor loadings per asset)
- Ledoit‑Wolf covariance shrinkage
- Global minimum variance optimisation

---

## Files

| File | Description |
|------|-------------|
| `train_mini2.py` | Training script (PCA, OLS, covariance) |
| `strategy.py` | Main strategy class (loads pre‑trained model) |
| `mini2_model.pkl` | Pre‑trained model (loadings, betas, alphas, covariance) |

---

## How to Run

### 1. Install dependencies
```bash
pip install pandas numpy scikit-learn
