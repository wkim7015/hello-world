# Quantitative Strategy Backtester for KOSPI & KOSDAQ

This repository contains a quantitative long-only strategy designed to operate on KOSPI and KOSDAQ stocks.

## Goal
To develop a rigorously validated, bias-free quantitative strategy.

## Elimination of Data Leakage
Based on strict quantitative research standards, this framework eliminates standard backtesting biases:
1. **Survivorship Bias:** The universe of stocks is sampled point-in-time. The framework fetches 10 years of historical market capitalization and builds a universe of all stocks that were *ever* in the KOSPI 150 or KOSDAQ 30 at that exact historical moment, including those that failed or were delisted.
2. **Look-Ahead Bias:** Regime filters (like Volatility and Moving Averages) use strictly lagging (`.rolling()`) data.
3. **In-Sample Curve Fitting:** The optimization utilizes a Walk-Forward Validation split. The models search for parameters exclusively on the first 60% of the timeline (In-Sample Training) and validate performance strictly out-of-sample on the remaining 40%.

## How to Use
Everything is contained in a single unified script: `quant_strategy.py`.

1. **Download Data:**
   Run `python quant_strategy.py --download`
   This handles the authentication with KRX, builds the historical point-in-time universe, and fetches 10-year OHLCV, Fundamental, and Institutional volume data.

2. **Generate and Validate Strategy:**
   Run `python quant_strategy.py --optimize`
   This computes an extreme concentration factor model (combining Momentum, Volatility, and Institutional Buying) coupled with a dynamic trailing stop-loss overlay to aggressively minimize drawdowns and evaluates it on an out-of-sample walk-forward framework.

Outputs generated:
- `trades.csv`: List of all buy and sell trades.
- `portfolio_values.csv`: The portfolio value over time.
- `best_weights.csv`: The daily target weights of the portfolio.
