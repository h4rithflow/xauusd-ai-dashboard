import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Simple XAUUSD Signal", page_icon="🟡", layout="wide")

st.markdown("""
<style>
.block-container{max-width:1180px;padding-top:1.2rem}.card{background:#111827;border:1px solid #293548;border-radius:14px;padding:16px;margin-bottom:12px}
.buy{color:#16c784;font-weight:800}.sell{color:#ea3943;font-weight:800}.wait{color:#f5b942;font-weight:800}
[data-testid="stMetric"]{background:#111827;border:1px solid #293548;padding:12px;border-radius:12px}
</style>
""", unsafe_allow_html=True)

LABELS={-1:"SELL",0:"WAIT",1:"BUY"}
FEATURES=["ret1","ret3","ema9_20","ema20_50","rsi","macdh","bbpos","atrpct","adx","didiff","stoch","vol"]

@st.cache_data(ttl=300,show_spinner=False)
def get_raw(symbol):
    # One download supports M5/M15. Higher frames are resampled locally.
    d=yf.download(symbol,period="60d",interval="5m",auto_adjust=False,progress=False,multi_level_index=False)
    if d.empty: raise ValueError("No data returned. Try GC=F or another supported gold symbol.")
    return d[["Open","High","Low","Close","Volume"]].dropna().sort_index()

def resample(df,rule):
    if rule=="5min": return df.copy()
    return df.resample(rule).agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()

def rsi(s,n=14):
    d=s.diff(); g=d.clip(lower=0); l=-d.clip(upper=0)
    rs=g.ewm(alpha=1/n,adjust=False).mean()/l.ewm(alpha=1/n,adjust=False).mean().replace(0,np.nan)
    return 100-100/(1+rs)

def atr(df,n=14):
    pc=df.Close.shift(); tr=pd.concat([df.High-df.Low,(df.High-pc).abs(),(df.Low-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()

def indicators(df):
    x=df.copy(); c=x.Close
    x["ret1"]=c.pct_change(); x["ret3"]=c.pct_change(3)
    e9=c.ewm(span=9,adjust=False).mean(); e20=c.ewm(span=20,adjust=False).mean(); e50=c.ewm(span=50,adjust=False).mean()
    x["ema20"]=e20; x["ema50"]=e50; x["ema9_20"]=e9/e20-1; x["ema20_50"]=e20/e50-1
    x["rsi"]=rsi(c); m=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean(); ms=m.ewm(span=9,adjust=False).mean(); x["macdh"]=(m-ms)/c
    mid=c.rolling(20).mean(); sd=c.rolling(20).std(); up=mid+2*sd; low=mid-2*sd; x["bbpos"]=(c-low)/(up-low).replace(0,np.nan)
    x["atr"]=atr(x); x["atrpct"]=x.atr/c
    upm=x.High.diff(); dnm=-x.Low.diff(); pdm=pd.Series(np.where((upm>dnm)&(upm>0),upm,0),index=x.index); mdm=pd.Series(np.where((dnm>upm)&(dnm>0),dnm,0),index=x.index)
    a=x.atr.replace(0,np.nan); pdi=100*pdm.ewm(alpha=1/14,adjust=False).mean()/a; mdi=100*mdm.ewm(alpha=1/14,adjust=False).mean()/a
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan); x["adx"]=dx.ewm(alpha=1/14,adjust=False).mean(); x["didiff"]=pdi-mdi
    lo=x.Low.rolling(14).min(); hi=x.High.rolling(14).max(); x["stoch"]=100*(c-lo)/(hi-lo).replace(0,np.nan)
    x["vol"]=x.ret1.rolling(20).std()
    return x.replace([np.inf,-np.inf],np.nan)

def model_signal(df,horizon,threshold):
    x=indicators(df); future=x.Close.shift(-horizon); ret=future/x.Close-1
    x["target"]=np.where(ret>threshold,1,np.where(ret<-threshold,-1,0)).astype(float); x.loc[future.isna(),"target"]=np.nan
    train=x.dropna(subset=FEATURES+["target"])
    if len(train)<180: return None
    X=train[FEATURES]; y=train.target.astype(int); cut=int(len(train)*.75)
    model=RandomForestClassifier(n_estimators=250,max_depth=6,min_samples_leaf=8,class_weight="balanced_subsample",random_state=42,n_jobs=-1)
    model.fit(X.iloc[:cut],y.iloc[:cut]); pred=model.predict(X.iloc[cut:]); prob=model.predict_proba(X.iloc[cut:])
    valid=y.iloc[cut:].to_numpy(); actionable=pred!=0
    released=int(actionable.sum()); wins=int((pred[actionable]==valid[actionable]).sum()) if released else 0
    winrate=wins/released if released else np.nan
    model.fit(X,y); latest=x.dropna(subset=FEATURES).iloc[-1]; raw=model.predict_proba(latest[FEATURES].to_frame().T)[0]
    probs={LABELS[int(k)]:float(v) for k,v in zip(model.classes_,raw)}; probs={k:probs.get(k,0) for k in ["BUY","WAIT","SELL"]}
    order=sorted(probs.items(),key=lambda p:p[1],reverse=True)
    signal=order[0][0] if order[0][1]>=.45 and order[0][1]-order[1][1]>=.08 else "WAIT"
    return {"signal":signal,"confidence":max(probs.values()),"probs":probs,"price":float(latest.Close),"atr":float(latest.atr),"rsi":float(latest.rsi),"adx":float(latest.adx),"winrate":winrate,"released":released,"wins":wins,"data":x}

def trade_plan(signal,price,a,pip,zone_pips,atr_mult,rr):
    half=zone_pips*pip/2; zone_low=price-half; zone_high=price+half; risk=max(a*atr_mult,zone_pips*pip)
    if signal=="BUY": sl=price-risk; tp=price+risk*rr
    elif signal=="SELL": sl=price+risk; tp=price-risk*rr
    else: sl=tp=np.nan
    return zone_low,zone_high,sl,tp,risk

st.title("🟡 Simple XAUUSD Signal")
st.caption("Multi-timeframe confluence, entry zone, risk-based targets and historical released-signal win rate")

with st.sidebar:
    st.header("Settings")
    symbol=st.text_input("Gold symbol","GC=F")
    pip=st.number_input("Pip size",min_value=0.001,value=0.01,step=0.001,format="%.3f",help="Default: 1 pip = 0.01 price units. Confirm with your broker.")
    zone_pips=st.number_input("Entry-zone width (pips)",10,500,50,10)
    rr=st.selectbox("Risk : reward",[1.5,2.0],format_func=lambda v:f"1:{v:g}")
    atr_mult=st.slider("Stop-loss ATR multiplier",0.5,2.5,1.0,0.1)
    if st.button("Refresh",use_container_width=True): st.cache_data.clear()

frames={"M5":("5min",3,.0007),"M15":("15min",2,.0010),"H1":("1h",2,.0015),"H4":("4h",1,.0020),"D1":("1D",1,.0030)}

try:
    raw=get_raw(symbol); results={}
    with st.spinner("Calculating confluence..."):
        for name,(rule,h,t) in frames.items(): results[name]=model_signal(resample(raw,rule),h,t)
    results={k:v for k,v in results.items() if v is not None}
    if not results: raise ValueError("Insufficient candles for all selected timeframes.")

    buy=sum(v["signal"]=="BUY" for v in results.values()); sell=sum(v["signal"]=="SELL" for v in results.values()); total=len(results)
    if buy>=3: overall="BUY"
    elif sell>=3: overall="SELL"
    else: overall="WAIT"
    color={"BUY":"buy","SELL":"sell","WAIT":"wait"}[overall]
    anchor=results.get("M15",next(iter(results.values())))
    zl,zh,sl,tp,risk=trade_plan(overall,anchor["price"],anchor["atr"],pip,zone_pips,atr_mult,rr)

    c1,c2,c3,c4=st.columns(4)
    c1.metric("Confluence signal",overall)
    c2.metric("Current price",f"{anchor['price']:,.2f}")
    c3.metric("Agreement",f"{max(buy,sell)}/{total} timeframes")
    avgwr=np.nanmean([v["winrate"] for v in results.values()])
    c4.metric("Historical win rate",f"{avgwr:.1%}" if np.isfinite(avgwr) else "N/A",help="Correct-direction rate among actionable BUY/SELL predictions in each timeframe's final 25% chronological holdout.")

    st.subheader("Timeframe confluence")
    cols=st.columns(len(results))
    for col,(name,v) in zip(cols,results.items()):
        with col:
            st.markdown(f"### {name}")
            cls={"BUY":"buy","SELL":"sell","WAIT":"wait"}[v['signal']]
            st.markdown(f"<div class='{cls}' style='font-size:1.6rem'>{v['signal']}</div>",unsafe_allow_html=True)
            st.caption(f"Confidence {v['confidence']:.0%}")
            st.caption(f"Win rate {v['winrate']:.1%}" if np.isfinite(v['winrate']) else "Win rate N/A")
            st.caption(f"Released {v['released']} | Won {v['wins']}")

    st.subheader("Trade plan")
    if overall=="WAIT":
        st.warning("No entry released. At least 3 timeframes must agree on BUY or SELL.")
    else:
        a,b,c,d=st.columns(4)
        a.metric("Entry zone",f"{zl:,.2f} – {zh:,.2f}")
        b.metric("Stop loss",f"{sl:,.2f}")
        c.metric(f"Take profit 1:{rr:g}",f"{tp:,.2f}")
        d.metric("Risk distance",f"{risk:,.2f}")
        st.caption(f"The entry band is {zone_pips} pips wide. SL uses the greater of {atr_mult:.1f}× ATR or the zone width. Confirm pip size with your broker.")

    st.subheader("Released-signal validation")
    rows=[]
    for name,v in results.items(): rows.append({"Timeframe":name,"Signal":v['signal'],"Confidence":v['confidence'],"Released signals":v['released'],"Winning signals":v['wins'],"Win rate":v['winrate']})
    table=pd.DataFrame(rows); table["Confidence"]=(table.Confidence*100).round(1).astype(str)+"%"; table["Win rate"]=table["Win rate"].map(lambda x:f"{x:.1%}" if pd.notna(x) else "N/A")
    st.dataframe(table,hide_index=True,use_container_width=True)

    st.subheader("Price overview")
    chart=anchor["data"].dropna(subset=["ema20","ema50"]).tail(200)
    fig=go.Figure(); fig.add_trace(go.Candlestick(x=chart.index,open=chart.Open,high=chart.High,low=chart.Low,close=chart.Close,name="Price")); fig.add_trace(go.Scatter(x=chart.index,y=chart.ema20,name="EMA20",line=dict(color="#f5b942"))); fig.add_trace(go.Scatter(x=chart.index,y=chart.ema50,name="EMA50",line=dict(color="#4c8bf5")))
    if overall!="WAIT": fig.add_hrect(y0=zl,y1=zh,fillcolor="#16c784" if overall=="BUY" else "#ea3943",opacity=.15,line_width=0); fig.add_hline(y=sl,line_dash="dash",line_color="#ea3943"); fig.add_hline(y=tp,line_dash="dash",line_color="#16c784")
    fig.update_layout(height=470,template="plotly_dark",xaxis_rangeslider_visible=False,margin=dict(l=10,r=10,t=25,b=10))
    st.plotly_chart(fig,use_container_width=True)

    st.info("Win rate is a backward-looking model test, not a guaranteed live result. It measures direction classification only and excludes spread, slippage, commissions and news gaps. This dashboard is for research, not personalized financial advice.")
except Exception as e:
    st.error(str(e))
