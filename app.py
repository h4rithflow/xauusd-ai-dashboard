import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit

st.set_page_config(page_title="XAUUSD AI Signal Dashboard", page_icon="🟡", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {background:#111827; border:1px solid #293548; padding:14px; border-radius:12px;}
.signal-buy {color:#16c784; font-size:2.3rem; font-weight:800;}
.signal-sell {color:#ea3943; font-size:2.3rem; font-weight:800;}
.signal-wait {color:#f5b942; font-size:2.3rem; font-weight:800;}
.small-note {color:#94a3b8; font-size:.85rem;}
</style>
""", unsafe_allow_html=True)

LABELS = {-1:"SELL", 0:"NEUTRAL", 1:"BUY"}
FEATURES = ["ret1","ret2","ret5","ema9_20","ema20_50","rsi","macd_pct","macd_hist_pct",
            "bb_pos","bb_width","atr_pct","stoch_k","stoch_d","adx","di_diff","vol10",
            "vol20","body","range_pos"]

@st.cache_data(ttl=300, show_spinner=False)
def load_data(symbol, period, interval):
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False,
                     progress=False, multi_level_index=False)
    if df.empty:
        raise ValueError("No market data returned by the selected feed.")
    return df.sort_index().dropna(subset=["Open","High","Low","Close"])

def rsi(s, n=14):
    d=s.diff(); gain=d.clip(lower=0); loss=-d.clip(upper=0)
    rs=gain.ewm(alpha=1/n, adjust=False, min_periods=n).mean()/loss.ewm(alpha=1/n, adjust=False, min_periods=n).mean().replace(0,np.nan)
    return (100-100/(1+rs)).clip(0,100)

def atr(df, n=14):
    pc=df.Close.shift(); tr=pd.concat([(df.High-df.Low),(df.High-pc).abs(),(df.Low-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()

def adx(df, n=14):
    up=df.High.diff(); down=-df.Low.diff()
    plus=pd.Series(np.where((up>down)&(up>0),up,0.0),index=df.index)
    minus=pd.Series(np.where((down>up)&(down>0),down,0.0),index=df.index)
    a=atr(df,n).replace(0,np.nan)
    pdi=100*plus.ewm(alpha=1/n,adjust=False).mean()/a
    mdi=100*minus.ewm(alpha=1/n,adjust=False).mean()/a
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return dx.ewm(alpha=1/n,adjust=False,min_periods=n).mean(),pdi,mdi

def engineer(df):
    x=df.copy(); c=x.Close
    x["ret1"]=c.pct_change(); x["ret2"]=c.pct_change(2); x["ret5"]=c.pct_change(5)
    for n in (9,20,50,200): x[f"ema{n}"]=c.ewm(span=n,adjust=False).mean()
    x["ema9_20"]=x.ema9/x.ema20-1; x["ema20_50"]=x.ema20/x.ema50-1
    x["rsi"]=rsi(c)
    e12=c.ewm(span=12,adjust=False).mean(); e26=c.ewm(span=26,adjust=False).mean()
    x["macd"]=e12-e26; x["macd_signal"]=x.macd.ewm(span=9,adjust=False).mean()
    x["macd_hist"]=x.macd-x.macd_signal; x["macd_pct"]=x.macd/c; x["macd_hist_pct"]=x.macd_hist/c
    mid=c.rolling(20).mean(); sd=c.rolling(20).std(); x["bb_upper"]=mid+2*sd; x["bb_lower"]=mid-2*sd
    width=(x.bb_upper-x.bb_lower).replace(0,np.nan); x["bb_pos"]=(c-x.bb_lower)/width; x["bb_width"]=width/mid
    x["atr"]=atr(x); x["atr_pct"]=x.atr/c
    lo=x.Low.rolling(14).min(); hi=x.High.rolling(14).max(); rng=(hi-lo).replace(0,np.nan)
    x["stoch_k"]=100*(c-lo)/rng; x["stoch_d"]=x.stoch_k.rolling(3).mean()
    x["adx"],pdi,mdi=adx(x); x["di_diff"]=pdi-mdi
    x["vol10"]=x.ret1.rolling(10).std(); x["vol20"]=x.ret1.rolling(20).std()
    cr=(x.High-x.Low).replace(0,np.nan); x["body"]=(x.Close-x.Open)/cr
    h20=x.High.rolling(20).max(); l20=x.Low.rolling(20).min(); x["range_pos"]=(c-l20)/(h20-l20).replace(0,np.nan)
    return x.replace([np.inf,-np.inf],np.nan)

def build_model(data, forecast, threshold):
    d=data.copy(); future=d.Close.shift(-forecast); fr=future/d.Close-1
    d["target"]=np.where(fr>threshold,1,np.where(fr<-threshold,-1,0)).astype(float)
    d.loc[future.isna(),"target"]=np.nan
    train=d.dropna(subset=FEATURES+["target"])
    if len(train)<220: raise ValueError("Not enough clean candles. Select a longer period or larger interval.")
    X=train[FEATURES]; y=train.target.astype(int)
    split=max(int(len(train)*.8),1); Xtr,Xte=X.iloc[:split],X.iloc[split:]; ytr,yte=y.iloc[:split],y.iloc[split:]
    model=RandomForestClassifier(n_estimators=400,max_depth=7,min_samples_leaf=8,max_features="sqrt",class_weight="balanced_subsample",random_state=42,n_jobs=-1)
    model.fit(Xtr,ytr); accuracy=accuracy_score(yte,model.predict(Xte)) if len(Xte) else np.nan
    model.fit(X,y)
    latest=data.dropna(subset=FEATURES).iloc[[-1]]
    raw=model.predict_proba(latest[FEATURES])[0]; probs={LABELS[int(k)]:float(v) for k,v in zip(model.classes_,raw)}
    probs={k:probs.get(k,0.0) for k in ["BUY","NEUTRAL","SELL"]}
    ordered=sorted(probs.items(), key=lambda z:z[1], reverse=True)
    signal=ordered[0][0] if ordered[0][1]>=.45 and ordered[0][1]-ordered[1][1]>=.08 else "WAIT"
    importance=pd.DataFrame({"Feature":FEATURES,"Importance":model.feature_importances_}).sort_values("Importance",ascending=False)
    return latest.iloc[0],probs,signal,accuracy,importance,train

st.title("🟡 XAUUSD AI Signal Dashboard")
st.caption("Technical indicators + Random Forest probabilities. Research use only, not financial advice.")

with st.sidebar:
    st.header("Settings")
    symbol=st.text_input("Market symbol", "GC=F", help="GC=F is used as a gold futures proxy. Enter your feed's XAUUSD symbol if supported.")
    interval=st.selectbox("Candle interval", ["1d","1h","30m","15m","5m"], index=0)
    periods={"1d":["1y","2y","5y"],"1h":["1mo","3mo","6mo","1y","2y"],"30m":["1mo","60d"],"15m":["1mo","60d"],"5m":["5d","1mo","60d"]}
    period=st.selectbox("History", periods[interval], index=min(1,len(periods[interval])-1))
    forecast=st.slider("Forecast candles",1,12,1)
    threshold_pct=st.slider("Neutral band (%)",0.05,1.00,0.20,0.05)
    threshold=threshold_pct/100
    st.caption("A prediction inside ± neutral band is labelled NEUTRAL.")
    if st.button("Refresh data", use_container_width=True): st.cache_data.clear()

try:
    with st.spinner("Loading data and training model..."):
        raw=load_data(symbol,period,interval); data=engineer(raw)
        latest,probs,signal,accuracy,importance,train=build_model(data,forecast,threshold)

    color={"BUY":"#16c784","SELL":"#ea3943","NEUTRAL":"#94a3b8","WAIT":"#f5b942"}[signal]
    prev=float(raw.Close.iloc[-2]); price=float(latest.Close); delta=(price/prev-1)*100
    a,b,c,d,e=st.columns(5)
    a.metric("Latest price",f"{price:,.2f}",f"{delta:+.2f}%",border=True)
    b.metric("AI signal",signal,border=True)
    c.metric("BUY probability",f"{probs['BUY']:.1%}",border=True)
    d.metric("SELL probability",f"{probs['SELL']:.1%}",border=True)
    e.metric("Holdout accuracy",f"{accuracy:.1%}",border=True,help="Simple final 20% chronological holdout. Not expected future performance.")

    st.markdown(f"<div style='margin:12px 0'>Current classification: <span style='color:{color};font-size:2rem;font-weight:800'>{signal}</span></div>",unsafe_allow_html=True)

    left,right=st.columns([2.2,1])
    with left:
        view=data.dropna(subset=["ema20","ema50"]).tail(180)
        fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[.72,.28],vertical_spacing=.04)
        fig.add_trace(go.Candlestick(x=view.index,open=view.Open,high=view.High,low=view.Low,close=view.Close,name="Price"),row=1,col=1)
        fig.add_trace(go.Scatter(x=view.index,y=view.ema20,name="EMA 20",line=dict(color="#f5b942",width=1.5)),row=1,col=1)
        fig.add_trace(go.Scatter(x=view.index,y=view.ema50,name="EMA 50",line=dict(color="#4c8bf5",width=1.5)),row=1,col=1)
        fig.add_trace(go.Scatter(x=view.index,y=view.bb_upper,name="BB Upper",line=dict(color="#64748b",width=1,dash="dot")),row=1,col=1)
        fig.add_trace(go.Scatter(x=view.index,y=view.bb_lower,name="BB Lower",line=dict(color="#64748b",width=1,dash="dot"),fill="tonexty",fillcolor="rgba(100,116,139,.08)"),row=1,col=1)
        fig.add_trace(go.Scatter(x=view.index,y=view.rsi,name="RSI 14",line=dict(color="#a855f7")),row=2,col=1)
        fig.add_hline(y=70,line_dash="dash",line_color="#ea3943",row=2,col=1); fig.add_hline(y=30,line_dash="dash",line_color="#16c784",row=2,col=1)
        fig.update_layout(height=650,xaxis_rangeslider_visible=False,template="plotly_dark",margin=dict(l=10,r=10,t=35,b=10),legend_orientation="h")
        st.plotly_chart(fig,width="stretch")

    with right:
        gauge=go.Figure(go.Bar(x=[probs["BUY"],probs["NEUTRAL"],probs["SELL"]],y=["BUY","NEUTRAL","SELL"],orientation="h",marker_color=["#16c784","#94a3b8","#ea3943"],text=[f"{v:.1%}" for v in [probs['BUY'],probs['NEUTRAL'],probs['SELL']]],textposition="auto"))
        gauge.update_layout(title="AI probabilities",xaxis=dict(range=[0,1],tickformat=".0%"),height=270,template="plotly_dark",margin=dict(l=10,r=10,t=45,b=10))
        st.plotly_chart(gauge,width="stretch")
        st.subheader("Indicators")
        ind=pd.DataFrame({"Indicator":["RSI 14","ADX 14","ATR","MACD Histogram","EMA 20","EMA 50"],"Value":[latest.rsi,latest.adx,latest.atr,latest.macd_hist,latest.ema20,latest.ema50]})
        ind["Value"]=ind.Value.map(lambda v:f"{v:,.4f}")
        st.dataframe(ind,hide_index=True,width="stretch")

    tab1,tab2,tab3=st.tabs(["Feature importance","Model information","Download"])
    with tab1:
        fi=importance.head(12).sort_values("Importance")
        ffig=go.Figure(go.Bar(x=fi.Importance,y=fi.Feature,orientation="h",marker_color="#f5b942"))
        ffig.update_layout(height=430,template="plotly_dark",margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(ffig,width="stretch")
    with tab2:
        st.write(f"Training candles: **{len(train):,}**")
        st.write(f"Forecast horizon: **{forecast} candle(s)**")
        st.write(f"Neutral threshold: **±{threshold_pct:.2f}%**")
        st.info("The displayed probabilities are model estimates from historical patterns. Holdout accuracy excludes trading costs and does not guarantee live performance.")
    with tab3:
        output=pd.DataFrame([{"timestamp":latest.name,"symbol":symbol,"price":price,"signal":signal,"buy_probability":probs['BUY'],"neutral_probability":probs['NEUTRAL'],"sell_probability":probs['SELL'],"rsi_14":latest.rsi,"adx_14":latest.adx,"atr_14":latest.atr}])
        st.download_button("Download latest signal CSV",output.to_csv(index=False),"xauusd_latest_signal.csv","text/csv",use_container_width=True)
        history=data.reset_index().to_csv(index=False)
        st.download_button("Download indicator history CSV",history,"xauusd_indicator_history.csv","text/csv",use_container_width=True)

except Exception as exc:
    st.error(str(exc))
    st.info("Try a longer history, a larger candle interval, or a valid symbol supported by the selected data feed.")
