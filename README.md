# Simple XAUUSD Confluence Dashboard

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment
Upload `app.py` and `requirements.txt` to the root of your GitHub repository, then deploy `app.py` on Streamlit Community Cloud.

## Important
The default `GC=F` is a futures proxy. Confirm the pip definition and price feed with your broker. Backtested win rate is historical and excludes trading costs.
