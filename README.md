# XAUUSD AI Signal Dashboard

A local Streamlit dashboard that downloads gold market data, calculates technical indicators, trains a Random Forest classifier, and displays BUY, NEUTRAL, SELL probabilities.

## Install

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Then open the local address shown by Streamlit.

## Notes

- The default symbol `GC=F` is a gold futures proxy, not broker-specific spot XAUUSD.
- For an actual trading workflow, replace the data-loading function with your authorized broker or market-data feed.
- Model probabilities are estimates for research purposes, not guaranteed signals.
- Validate spread, slippage, execution delay, and out-of-sample performance before any live use.
