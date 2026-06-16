# S&P 500 Market-Neutral Factor Model

A point-in-time, market-neutral long/short equity model on the S&P 500 (top-500 by market cap).
A 5-seed LightGBM ensemble ranks names on 49 cross-sectional factors (multi-horizon volatility,
momentum, anti-lottery, fundamentals, path/shape risk, peer-relative), then builds a
**beta- and factor-neutralized** long-top-30 / short-bottom-30 book, rebalanced monthly on a
63-day signal.

## Results (honest)

Validated on a **sealed 2024-2026 holdout** the model never saw during development:

| | in-sample (2016-23) | **holdout (2024-26)** |
|---|---|---|
| net annual | +9.2% | **+3.7%** |
| Sharpe | 0.98 | **0.39** |

The in-sample Sharpe inflated ~3x over the holdout - a reminder that only out-of-sample results count.
The genuine edge is **small but real**: ~+3-4% net, Sharpe ~0.35-0.4, market-neutral (survives crashes),
shortable, and scalable. Costs charged at 5 bps/side. No leverage.

## Layout

```
newcycle/featlab.py            point-in-time price/volume substrate (cached matrices)
newcycle/s2_build_features.py  feature helpers (rolling extremes, trend slope/R^2)
rebuild/harness.py             universe, cost model, reserved holdout, net-book evaluator
rebuild/best_model.py          the 49-feature GBM (importable feature set + config)
rebuild/sp500_model.py         S&P-500 universe + market-neutral long/short book
rebuild/sp500_holdout.py       the one-shot sealed-holdout evaluation
rebuild/sp500_predict.py       live longs/shorts + SHAP-based reasons -> reports/sp500_predictions.json
rebuild/sp500_history.py       monthly return series for the site -> reports/sp500_history.json
web/index.html                 the model site: performance history, drawdowns, current book
web/sp500-positions.html       positions dashboard (current picks + why)
web/quant-factor-platform.html research showcase with a live factor-model bench
```

## Run

```bash
pip install -r requirements.txt
# from the repo root:
python rebuild/sp500_holdout.py     # train ensemble, evaluate DEV vs sealed holdout
python rebuild/sp500_predict.py     # latest longs/shorts with model reasons -> reports/sp500_predictions.json
```

**Data.** The model reads a cached PIT substrate (`data/newcycle/*.npz`) and a leak-free fundamentals
panel (`data/newcycle/fund_panel_pit.parquet`). These are ~8 GB, regenerated from Alpha Vantage (prices)
and SEC EDGAR (fundamentals), and are **not** tracked here. Open `web/sp500-positions.html` directly to
view the model's most recent output without any data.

## Disclaimer

Research model - **not investment advice.** Prices as of the stated date; results are net of the stated
costs. Past and simulated performance do not guarantee future results.
