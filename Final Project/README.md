```markdown
# Final Project – Adaptive Multi‑Factor Strategy with Bayesian Signal Weighting, Black‑Litterman & Risk Management

## Overview

This is the **complete, fully integrated** final project strategy. It combines:

- **Three trading signals** (momentum, mean‑reversion, volume‑confirmed trend)
- **Bayesian model averaging** (empirical Bayes) for dynamic signal weighting
- **Regime‑aware factor models** (2 regimes: low‑vol / high‑vol) trained via rolling PCA and k‑means
- **Black‑Litterman blending** – combines factor model prior with signal‑based views
- **Mean‑variance optimisation** (tangency portfolio)
- **Risk management** – CVaR heuristic + volatility targeting (kernel density estimation)

All components are **trained offline** on the large training set (500 MB, 180k periods) and run efficiently online.

---

## Files

| File | Description |
|------|-------------|
| `train_final.py` | Enhanced training script (rolling PCA, regime clustering, signal priors) |
| `strategy.py` | Main strategy class (loads pre‑trained model, orchestrates inference) |
| `signals.py` | Momentum, mean‑reversion, volume trend score computation |
| `bayesian.py` | Bayesian signal weighting (rolling correlation + softmax) |
| `black_litterman.py` | Black‑Litterman blending formula |
| `risk.py` | GMV, long‑only projection, CVaR heuristic, volatility targeting (KDE) |
| `final_model.pkl` | Pre‑trained model (~1.6 MB) |
| `main.py` | Backtest runner (modified to save results and plots) |
| `portfolio_returns.csv` | Output: 5‑min portfolio returns |
| `strategy_performance.png` | Performance plots (cumulative return, rolling Sharpe, drawdown) |
| `Final_Report.pdf` | Full project report (30% of grade) |

---

## How to Run

### 1. Install dependencies
```bash
pip install pandas numpy scikit-learn scipy matplotlib
