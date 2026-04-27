# pip install streamlit yfinance requests plotly scipy pandas
import streamlit as st
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Options Flow Dashboard", page_icon="📈")

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; background-color: #0a0c10; color: #e0e4ef; }
.stApp { background-color: #0a0c10; }
h1,h2,h3 { font-family: 'IBM Plex Mono', monospace; }
.signal-box { background:#111318; border-left:3px solid #7b8cde; border-radius:4px; padding:12px 16px; margin:6px 0; font-family:'IBM Plex Mono',monospace; font-size:13px; }
.signal-buy  { border-left-color:#00e5a0; }
.signal-sell { border-left-color:#ff4d6d; }
.signal-warn { border-left-color:#f4b942; }
.signal-info { border-left-color:#7b8cde; }
.tab-header { font-family:'IBM Plex Mono',monospace; font-size:11px; color:#6b7694; text-transform:uppercase; letter-spacing:2px; padding:8px 0 16px 0; border-bottom:1px solid #1e2230; margin-bottom:20px; }
.section-title { font-family:'IBM Plex Mono',monospace; font-size:13px; color:#7b8cde; text-transform:uppercase; letter-spacing:1.5px; margin:24px 0 12px 0; border-bottom:1px solid #1e2230; padding-bottom:8px; }
div[data-testid="stMetric"] { background:#111318; border:1px solid #1e2230; border-radius:8px; padding:14px 18px; }
div[data-testid="stMetric"] label { color:#6b7694 !important; font-size:11px; text-transform:uppercase; letter-spacing:1px; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color:#e0e4ef; font-family:'IBM Plex Mono',monospace; }
.stTabs [data-baseweb="tab-list"] { background:#0a0c10; border-bottom:1px solid #1e2230; }
.stTabs [data-baseweb="tab"] { font-family:'IBM Plex Mono',monospace; font-size:12px; color:#6b7694; text-transform:uppercase; letter-spacing:1px; background:transparent; border:none; border-bottom:2px solid transparent; }
.stTabs [aria-selected="true"] { color:#e0e4ef !important; border-bottom:2px solid #7b8cde !important; background:transparent !important; }
</style>
""", unsafe_allow_html=True)

# ── Plot theme ─────────────────────────────────────────────────────────────────
PT = dict(paper_bgcolor="#0a0c10", plot_bgcolor="#0a0c10",
          font=dict(family="IBM Plex Mono", color="#e0e4ef", size=11),
          xaxis=dict(gridcolor="#1e2230", zerolinecolor="#1e2230"),
          yaxis=dict(gridcolor="#1e2230", zerolinecolor="#1e2230"),
          margin=dict(l=40,r=20,t=40,b=40))

# ── Constants ──────────────────────────────────────────────────────────────────
SECTOR_ETFS = {"Technology":"XLK","Financials":"XLF","Healthcare":"XLV","Energy":"XLE",
               "Consumer Disc.":"XLY","Consumer Staples":"XLP","Industrials":"XLI",
               "Materials":"XLB","Utilities":"XLU","Real Estate":"XLRE","Communication":"XLC"}

SP500_SAMPLE = ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","JPM","V","LLY",
                "UNH","XOM","MA","AVGO","HD","CVX","MRK","ABBV","PEP","KO","COST",
                "ADBE","WMT","CRM","TMO","ACN","MCD","NFLX","AMD","LIN","DHR","TXN",
                "NEE","PM","ORCL","RTX","HON","AMGN","QCOM","LOW","IBM","GE","CAT",
                "BA","GS","SBUX","INTU","ISRG","SPGI","BLK","NOW","DE","AMAT","LRCX",
                "ADI","PANW","GILD","MU","AXP","PLD","CI","SLB","EOG","WFC","MS","C",
                "BAC","USB","PNC","TFC","COF","CB","MMC","ZTS","REGN","VRTX","BMY",
                "MRNA","BIIB","EW","SYK","BSX","MDT","ABT","JNJ","PFE","HCA","CVS"]

# ── Session (fixes Yahoo 429 rate limits) ─────────────────────────────────────
@st.cache_resource
def get_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/124.0.0.0 Safari/537.36"),
        "Accept-Language": "en-US,en;q=0.9",
    })
    try:
        s.get("https://finance.yahoo.com", timeout=8)
    except Exception:
        pass
    return s

# ── Data functions ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=900)
def get_hist(sym, period="1y"):
    sess = get_session()
    for attempt in range(3):
        try:
            t = yf.Ticker(sym, session=sess)
            df = t.history(period=period, auto_adjust=True)
            if df is not None and not df.empty:
                return df
        except Exception:
            if attempt < 2:
                time.sleep(4 * (attempt + 1))
    return pd.DataFrame()

@st.cache_data(ttl=900)
def batch_download(tickers, period="1y"):
    sess = get_session()
    tickers = list(tickers)
    for attempt in range(3):
        try:
            df = yf.download(tickers, period=period, auto_adjust=True,
                             group_by="ticker", progress=False,
                             threads=False, session=sess)
            if not df.empty:
                return df
        except Exception:
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
    return pd.DataFrame()

def extract_close(batch_df, sym):
    try:
        if isinstance(batch_df.columns, pd.MultiIndex):
            return batch_df["Close"][sym].dropna()
        return batch_df["Close"].dropna()
    except Exception:
        return pd.Series(dtype=float)

# ── Math helpers ───────────────────────────────────────────────────────────────
def bs_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return max(S-K, 0)
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def bs_greeks(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return {}
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return dict(
        delta=norm.cdf(d1),
        gamma=norm.pdf(d1)/(S*sigma*np.sqrt(T)),
        theta=(-(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2))/365,
        vega=S*norm.pdf(d1)*np.sqrt(T)/100
    )

def time_to_expiry(exp):
    return max((datetime.strptime(exp, "%Y-%m-%d") - datetime.today()).days/365, 0.001)

def compute_hv(hist, window=21):
    ret = np.log(hist["Close"]/hist["Close"].shift(1)).dropna()
    return ret.rolling(window).std() * np.sqrt(252)

def compute_rv_stats(hist):
    ret = np.log(hist["Close"]/hist["Close"].shift(1)).dropna()
    return tuple(ret.rolling(w).std().iloc[-1]*np.sqrt(252) for w in [10,21,63,126])

def atm_iv(calls, puts, spot):
    c = calls.iloc[(calls["strike"]-spot).abs().argsort()[:1]]["impliedVolatility"]
    p = puts.iloc[(puts["strike"]-spot).abs().argsort()[:1]]["impliedVolatility"]
    v = pd.concat([c,p]).dropna()
    return v.mean() if not v.empty else np.nan

def call_skew(calls, spot):
    if len(calls) < 5: return np.nan, np.nan
    atm_iv_val = calls.iloc[(calls["strike"]-spot).abs().argsort()[:1]]["impliedVolatility"].values[0]
    otm = calls[calls["strike"] > spot*1.05]
    if otm.empty: return np.nan, np.nan
    raw = otm.iloc[0]["impliedVolatility"] - atm_iv_val
    std = calls["impliedVolatility"].std()
    return round(raw,4), round(raw/std, 2) if std > 0 else np.nan

def mom_score(hist):
    c = hist["Close"]
    return {p: (c.iloc[-1]/c.iloc[-p]-1)*100 if len(c)>=p else np.nan for p in [21,63,126,252]}

def w52(hist):
    c = hist["Close"].tail(252)
    cur = hist["Close"].iloc[-1]
    return (cur/c.max()-1)*100, (cur/c.min()-1)*100, c.max(), c.min()

def signal(text, t="info"):
    return f'<div class="signal-box signal-{t}">{text}</div>'

def color_pct(val):
    if pd.isna(val): return ""
    if val > 5:  return "color: #00e5a0"
    if val < -5: return "color: #ff4d6d"
    return "color: #f4b942"

# ── Screeners ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=900)
def screen_momentum(tickers, benchmark="SPY"):
    all_syms = list(set(list(tickers) + [benchmark]))
    batch = batch_download(all_syms, "1y")
    if batch.empty: return pd.DataFrame()

    bc = extract_close(batch, benchmark)
    if bc.empty or len(bc) < 63: return pd.DataFrame()
    b63  = (bc.iloc[-1]/bc.iloc[-63]-1)*100
    b252 = (bc.iloc[-1]/bc.iloc[-252]-1)*100 if len(bc)>=252 else np.nan

    results = []
    for sym in tickers:
        try:
            c = extract_close(batch, sym)
            if c.empty or len(c) < 63: continue
            cur = c.iloc[-1]
            def m(p): return (c.iloc[-1]/c.iloc[-p]-1)*100 if len(c)>=p else np.nan
            m21,m63,m126,m252 = m(21),m(63),m(126),m(252)
            h52 = c.tail(252).max(); l52 = c.tail(252).min()
            results.append({"Ticker":sym,"Price":round(cur,2),
                "1M %":round(m21,1),"3M %":round(m63,1),"6M %":round(m126,1),"12M %":round(m252,1),
                "vs SPY 3M":round(m63-b63,1) if not np.isnan(m63) else np.nan,
                "vs SPY 12M":round(m252-b252,1) if (not np.isnan(m252) and not np.isnan(b252)) else np.nan,
                "52W High%":round((cur/h52-1)*100,1),"52W Low%":round((cur/l52-1)*100,1)})
        except Exception: continue

    df = pd.DataFrame(results)
    if df.empty: return df
    for col in ["1M %","3M %","6M %","12M %"]:
        df[col+"_r"] = df[col].rank(pct=True)
    df["MomScore"] = df[["1M %_r","3M %_r","6M %_r","12M %_r"]].mean(axis=1).round(2)
    df = df.drop(columns=[c for c in df.columns if c.endswith("_r")])
    return df.sort_values("MomScore", ascending=False)

@st.cache_data(ttl=900)
def screen_sectors():
    syms = list(SECTOR_ETFS.values())
    batch = batch_download(syms, "1y")
    if batch.empty: return pd.DataFrame()
    results = []
    for name, sym in SECTOR_ETFS.items():
        try:
            c = extract_close(batch, sym)
            if c.empty or len(c) < 63: continue
            cur = c.iloc[-1]
            def m(p): return (c.iloc[-1]/c.iloc[-p]-1)*100 if len(c)>=p else np.nan
            results.append({"Sector":name,"ETF":sym,"Price":round(cur,2),
                "1M %":round(m(21),1),"3M %":round(m(63),1),
                "6M %":round(m(126),1),"12M %":round(m(252),1),
                "vs 52W High":round((cur/c.tail(252).max()-1)*100,1)})
        except Exception: continue
    df = pd.DataFrame(results)
    if df.empty: return df
    for col in ["1M %","3M %","6M %","12M %"]:
        df[col+"_r"] = df[col].rank(pct=True)
    df["MomScore"] = df[["1M %_r","3M %_r","6M %_r","12M %_r"]].mean(axis=1).round(2)
    df = df.drop(columns=[c for c in df.columns if c.endswith("_r")])
    return df.sort_values("MomScore", ascending=False)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:11px;color:#6b7694;text-transform:uppercase;letter-spacing:2px;margin-bottom:16px;">Parameters</div>', unsafe_allow_html=True)
    ticker_input = st.text_input("Ticker", "AAPL").upper().strip()
    r = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 4.5)/100
    st.markdown("---")
    st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:11px;color:#6b7694;text-transform:uppercase;letter-spacing:2px;margin-bottom:12px;">Screener</div>', unsafe_allow_html=True)
    screen_universe = st.selectbox("Universe", ["S&P 500 Sample", "Custom Watchlist"])
    if screen_universe == "Custom Watchlist":
        custom = st.text_area("Tickers (comma-separated)", "AAPL,MSFT,NVDA,TSLA")
        screen_tickers = [t.strip().upper() for t in custom.split(",") if t.strip()]
    else:
        screen_tickers = SP500_SAMPLE
    benchmark_input = st.text_input("Benchmark", "SPY")
    relative_to     = st.text_input("Relative vs.", "QQQ")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:4px;">
  <span style="font-family:'IBM Plex Mono',monospace;font-size:22px;font-weight:700;color:#e0e4ef;">OPTIONS FLOW</span>
  <span style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#6b7694;letter-spacing:3px;">Dashboard V5</span>
</div>
<div style="font-family:'IBM Plex Sans',sans-serif;font-size:13px;color:#6b7694;margin-bottom:24px;">
  Vertical Spread Analysis · Volatility Regime · Momentum Screening
</div>""", unsafe_allow_html=True)

# ── Load main ticker ───────────────────────────────────────────────────────────
sess = get_session()
try:
    hist_1y = get_hist(ticker_input, "1y")
    if hist_1y.empty:
        st.error(f"⚠️ No data for **{ticker_input}**. Check the ticker or wait 30s and refresh.")
        st.stop()
    spot = hist_1y["Close"].iloc[-1]
    ticker_obj = yf.Ticker(ticker_input, session=sess)
    expirations = []
    for _a in range(3):
        try:
            expirations = ticker_obj.options or []
            if expirations: break
        except Exception:
            time.sleep(3)
    if not expirations:
        st.error(f"⚠️ No options data for **{ticker_input}**. Yahoo may be rate-limiting — wait 30s and refresh.")
        st.stop()
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

rv10, rv21, rv63, rv126 = compute_rv_stats(hist_1y)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tabs = st.tabs(["📊  Overview","📉  Volatility","🎯  Chain","⚡  Spreads","📈  Momentum","🌐  Sectors"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown(f'<div class="tab-header">Market Overview · {ticker_input}</div>', unsafe_allow_html=True)
    d1 = (hist_1y["Close"].iloc[-1]/hist_1y["Close"].iloc[-2]-1)*100
    d21= (hist_1y["Close"].iloc[-1]/hist_1y["Close"].iloc[-21]-1)*100
    d63= (hist_1y["Close"].iloc[-1]/hist_1y["Close"].iloc[-63]-1)*100
    ph, pl, h52, l52 = w52(hist_1y)

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Spot",     f"${spot:.2f}",   f"{d1:+.2f}% today")
    c2.metric("1M Ret",   f"{d21:+.1f}%")
    c3.metric("3M Ret",   f"{d63:+.1f}%")
    c4.metric("52W High", f"${h52:.2f}", f"{ph:.1f}% away")
    c5.metric("52W Low",  f"${l52:.2f}", f"+{pl:.1f}% away")

    fig = make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.75,0.25],vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=hist_1y.index,open=hist_1y["Open"],high=hist_1y["High"],
        low=hist_1y["Low"],close=hist_1y["Close"],
        increasing_line_color="#00e5a0",decreasing_line_color="#ff4d6d",name="Price"),row=1,col=1)
    fig.add_trace(go.Scatter(x=hist_1y.index,y=hist_1y["Close"].rolling(20).mean(),
        line=dict(color="#7b8cde",width=1),name="MA20"),row=1,col=1)
    fig.add_trace(go.Scatter(x=hist_1y.index,y=hist_1y["Close"].rolling(50).mean(),
        line=dict(color="#f4b942",width=1,dash="dot"),name="MA50"),row=1,col=1)
    colors=["#00e5a0" if c>=o else "#ff4d6d" for c,o in zip(hist_1y["Close"],hist_1y["Open"])]
    fig.add_trace(go.Bar(x=hist_1y.index,y=hist_1y["Volume"],marker_color=colors,opacity=0.6,name="Vol"),row=2,col=1)
    fig.update_layout(**PT,height=480,showlegend=True,legend=dict(orientation="h",y=1.02,x=0))
    fig.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Volatility Snapshot</div>', unsafe_allow_html=True)
    vc1,vc2,vc3,vc4 = st.columns(4)
    vc1.metric("10D RV", f"{rv10:.1%}")
    vc2.metric("21D RV", f"{rv21:.1%}")
    vc3.metric("63D RV", f"{rv63:.1%}")
    vc4.metric("126D RV",f"{rv126:.1%}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — VOLATILITY
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown(f'<div class="tab-header">Volatility Analysis · {ticker_input}</div>', unsafe_allow_html=True)

    sel_exp_vol = st.selectbox("Reference Expiration", expirations, key="vol_exp")
    T_vol = time_to_expiry(sel_exp_vol)
    try:
        chain_vol = ticker_obj.option_chain(sel_exp_vol)
        calls_v = chain_vol.calls.dropna(subset=["impliedVolatility"])
        puts_v  = chain_vol.puts.dropna(subset=["impliedVolatility"])
        iv_atm  = atm_iv(calls_v, puts_v, spot)
    except Exception:
        calls_v, puts_v, iv_atm = pd.DataFrame(), pd.DataFrame(), np.nan

    vrp = iv_atm - rv21

    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("ATM IV",          f"{iv_atm:.1%}" if not np.isnan(iv_atm) else "—")
    m2.metric("21D Realized",    f"{rv21:.1%}")
    m3.metric("Vol Risk Premium",f"{vrp:+.1%}"   if not np.isnan(vrp)   else "—")
    m4.metric("10D RV",          f"{rv10:.1%}")
    m5.metric("63D RV",          f"{rv63:.1%}")

    if not np.isnan(vrp):
        if vrp > 0.05:
            st.markdown(signal(f"📉 VRP = {vrp:+.1%} → IV elevated vs RV. Credit spreads / short premium favored.", "sell"), unsafe_allow_html=True)
        elif vrp < -0.05:
            st.markdown(signal(f"📈 VRP = {vrp:+.1%} → IV cheap vs RV. Debit spreads / long vol favored.", "buy"), unsafe_allow_html=True)
        else:
            st.markdown(signal(f"⚖️ VRP = {vrp:+.1%} → IV near fair value.", "info"), unsafe_allow_html=True)

    st.markdown("")

    # ── HV vs IV Chart with lookback selector ─────────────────────────────────
    st.markdown('<div class="section-title">Historical vs Implied Volatility</div>', unsafe_allow_html=True)

    hv_lookback = st.select_slider(
        "Lookback period",
        options=["6M","1Y","2Y","3Y","5Y"],
        value="1Y"
    )
    period_map = {"6M":"6mo","1Y":"1y","2Y":"2y","3Y":"3y","5Y":"5y"}
    hv_period = period_map[hv_lookback]

    hist_hv = get_hist(ticker_input, hv_period)

    if not hist_hv.empty:
        hv10  = compute_hv(hist_hv, 10)
        hv21  = compute_hv(hist_hv, 21)
        hv63  = compute_hv(hist_hv, 63)
        hv126 = compute_hv(hist_hv, 126)

        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(x=hv10.index,  y=hv10,  name="10D HV",  line=dict(color="#ff4d6d",width=1.5)))
        fig_vol.add_trace(go.Scatter(x=hv21.index,  y=hv21,  name="21D HV",  line=dict(color="#f4b942",width=1.5)))
        fig_vol.add_trace(go.Scatter(x=hv63.index,  y=hv63,  name="63D HV",  line=dict(color="#7b8cde",width=1.5)))
        fig_vol.add_trace(go.Scatter(x=hv126.index, y=hv126, name="126D HV", line=dict(color="#a78bfa",width=1,dash="dot")))
        if not np.isnan(iv_atm):
            fig_vol.add_hline(y=iv_atm, line_color="#00e5a0", line_dash="dash",
                              annotation_text=f"ATM IV {iv_atm:.1%}", annotation_position="top right")
        fig_vol.update_layout(**PT, height=380, title=f"Realized Vol Cones vs ATM IV ({hv_lookback})",
                              yaxis_tickformat=".0%")
        st.plotly_chart(fig_vol, use_container_width=True)

        # Vol percentile context
        hv21_series = hv21.dropna()
        if len(hv21_series) > 20 and not np.isnan(iv_atm):
            iv_rank_pct = (iv_atm - hv21_series.min()) / (hv21_series.max() - hv21_series.min()) * 100
            vrp_series  = iv_atm - hv21_series
            st.markdown('<div class="section-title">Vol Regime Context</div>', unsafe_allow_html=True)
            rc1,rc2,rc3,rc4 = st.columns(4)
            rc1.metric(f"IV Rank ({hv_lookback})",    f"{iv_rank_pct:.0f}%")
            rc2.metric("HV21 Min",  f"{hv21_series.min():.1%}")
            rc3.metric("HV21 Max",  f"{hv21_series.max():.1%}")
            rc4.metric("HV21 Avg",  f"{hv21_series.mean():.1%}")
    else:
        st.info(f"Could not load {hv_lookback} history.")

    # IV vs RV table
    rows = [("10D RV",rv10,iv_atm-rv10),("21D RV",rv21,iv_atm-rv21),
            ("63D RV",rv63,iv_atm-rv63),("126D RV",rv126,iv_atm-rv126)]
    df_ivrv = pd.DataFrame(rows, columns=["Tenor","Realized Vol","VRP (IV - RV)"])
    df_ivrv["Realized Vol"] = df_ivrv["Realized Vol"].map("{:.1%}".format)
    df_ivrv["VRP (IV - RV)"] = df_ivrv["VRP (IV - RV)"].map(lambda x: f"{x:+.1%}")
    st.dataframe(df_ivrv, use_container_width=True, hide_index=True)

    # ── Call Skew ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Call Skew Analysis</div>', unsafe_allow_html=True)
    if not calls_v.empty:
        sk_raw, sk_z = call_skew(calls_v, spot)
        sk1,sk2 = st.columns(2)
        sk1.metric("Call Skew (OTM5% - ATM)", f"{sk_raw:+.3f}" if sk_raw and not np.isnan(sk_raw) else "—")
        sk2.metric("Skew Z-Score",             f"{sk_z:+.2f}σ"  if sk_z  and not np.isnan(sk_z)  else "—")
        if sk_z and not np.isnan(sk_z):
            if sk_z > 1.5:
                st.markdown(signal(f"🔺 Call skew elevated ({sk_z:+.2f}σ) — upside bid. Sell OTM call spreads.", "sell"), unsafe_allow_html=True)
            elif sk_z < -1.5:
                st.markdown(signal(f"🔻 Call skew depressed ({sk_z:+.2f}σ) — upside cheap. Buy debit call spreads.", "buy"), unsafe_allow_html=True)
            else:
                st.markdown(signal(f"⚖️ Call skew normal ({sk_z:+.2f}σ).", "info"), unsafe_allow_html=True)

        fig_smile = go.Figure()
        fig_smile.add_trace(go.Scatter(x=calls_v["strike"],y=calls_v["impliedVolatility"],
            mode="lines+markers",name="Call IV",line=dict(color="#00e5a0",width=2),marker=dict(size=5)))
        fig_smile.add_trace(go.Scatter(x=puts_v["strike"],y=puts_v["impliedVolatility"],
            mode="lines+markers",name="Put IV",line=dict(color="#ff4d6d",width=2),marker=dict(size=5)))
        fig_smile.add_vline(x=spot,line_color="#f4b942",line_dash="dash",annotation_text="Spot")
        fig_smile.update_layout(**PT,height=320,title="IV Smile",yaxis_tickformat=".0%")
        st.plotly_chart(fig_smile, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — OPTIONS CHAIN
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown(f'<div class="tab-header">Options Chain · {ticker_input}</div>', unsafe_allow_html=True)
    sel_exp_chain = st.selectbox("Expiration", expirations, key="chain_exp")
    T_chain = time_to_expiry(sel_exp_chain)
    try:
        chain = ticker_obj.option_chain(sel_exp_chain)
        calls = chain.calls.dropna(subset=["impliedVolatility"]).copy()
        puts  = chain.puts.dropna(subset=["impliedVolatility"]).copy()
    except Exception as e:
        st.error(f"Error loading chain: {e}"); st.stop()

    iv_atm_chain = atm_iv(calls, puts, spot)
    calls["bs"] = calls.apply(lambda x: bs_call(spot,x["strike"],T_chain,r,x["impliedVolatility"]),axis=1)
    calls["mispricing"] = calls["lastPrice"] - calls["bs"]
    greeks = calls.apply(lambda x: bs_greeks(spot,x["strike"],T_chain,r,x["impliedVolatility"]),axis=1)
    for g in ["delta","gamma","theta","vega"]:
        calls[g] = greeks.apply(lambda x: x.get(g, np.nan))

    s1,s2,s3,s4 = st.columns(4)
    s1.metric("DTE",       f"{int(T_chain*365)}d")
    s2.metric("ATM IV",    f"{iv_atm_chain:.1%}" if not np.isnan(iv_atm_chain) else "—")
    s3.metric("Exp. Move", f"±${spot*iv_atm_chain*np.sqrt(T_chain):.2f}" if not np.isnan(iv_atm_chain) else "—")
    s4.metric("Strikes",   f"{len(calls)}")

    dcols = {c:c for c in ["strike","lastPrice","bs","mispricing","impliedVolatility","delta","gamma","theta","vega","volume","openInterest"] if c in calls.columns}
    rename = {"strike":"Strike","lastPrice":"Mkt","bs":"BS","mispricing":"Mispricing",
              "impliedVolatility":"IV","delta":"Δ","gamma":"Γ","theta":"Θ","vega":"ν",
              "volume":"Vol","openInterest":"OI"}
    st.dataframe(calls[list(dcols.keys())].rename(columns=rename).round(4),
                 use_container_width=True, height=400)

    fig_mis = go.Figure()
    fig_mis.add_trace(go.Bar(x=calls["strike"],y=calls["mispricing"],
        marker_color=["#00e5a0" if v<0 else "#ff4d6d" for v in calls["mispricing"]]))
    fig_mis.add_hline(y=0,line_color="#6b7694")
    fig_mis.add_vline(x=spot,line_color="#f4b942",line_dash="dash",annotation_text="Spot")
    fig_mis.update_layout(**PT,height=260,title="Mispricing by Strike (Mkt - BS)")
    st.plotly_chart(fig_mis, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SPREAD BUILDER
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown(f'<div class="tab-header">Vertical Spread Builder · {ticker_input}</div>', unsafe_allow_html=True)
    sel_exp_sb = st.selectbox("Expiration", expirations, key="sb_exp")
    T_sb = time_to_expiry(sel_exp_sb)
    try:
        chain_sb = ticker_obj.option_chain(sel_exp_sb)
        calls_sb = chain_sb.calls.dropna(subset=["impliedVolatility"]).copy()
        puts_sb  = chain_sb.puts.dropna(subset=["impliedVolatility"]).copy()
    except Exception as e:
        st.error(f"Error: {e}"); st.stop()

    iv_atm_sb = atm_iv(calls_sb, puts_sb, spot)
    vrp_sb = iv_atm_sb - rv21

    col_l, col_r = st.columns([1,2])
    with col_l:
        strategy = st.selectbox("Strategy",["Bull Call Spread","Bear Call Spread","Bull Put Spread","Bear Put Spread"])
        use_calls = "Call" in strategy
        chain_use = calls_sb if use_calls else puts_sb
        strikes = sorted(chain_use["strike"].unique())
        mid = len(strikes)//2
        k1 = st.selectbox("Long Strike",  strikes, index=min(mid,   len(strikes)-1), key="k1")
        k2 = st.selectbox("Short Strike", strikes, index=min(mid+2, len(strikes)-1), key="k2")
        def get_prem(k): 
            row = chain_use[chain_use["strike"]==k]
            return float(row["lastPrice"].values[0]) if not row.empty else 1.0
        p1 = st.number_input("Long Premium",  value=round(get_prem(k1),2), step=0.05)
        p2 = st.number_input("Short Premium", value=round(get_prem(k2),2), step=0.05)
        fees      = st.number_input("Fees + Slippage", value=0.20, step=0.05)
        contracts = st.number_input("Contracts (x100)", value=1, min_value=1)

    def spread_pnl(S, strat, k1, k2, p1, p2, fees):
        if   strat=="Bull Call Spread": return np.maximum(S-k1,0)-np.maximum(S-k2,0)-(p1-p2)-fees
        elif strat=="Bear Call Spread": return np.maximum(S-k2,0)-np.maximum(S-k1,0)+(p2-p1)-fees
        elif strat=="Bull Put Spread":  return -(np.maximum(k1-S,0)-np.maximum(k2-S,0))+(p2-p1)-fees
        elif strat=="Bear Put Spread":  return np.maximum(k1-S,0)-np.maximum(k2-S,0)-(p1-p2)-fees
        return np.zeros_like(S)

    if "Bull" in strategy:
        net = p1-p2; max_p=(abs(k2-k1)-net-fees)*100*contracts; max_l=(net+fees)*100*contracts
        be = k1+net+fees if use_calls else k2-net-fees
    else:
        net=p2-p1; max_p=(net-fees)*100*contracts; max_l=(abs(k2-k1)-net+fees)*100*contracts
        be = k2+net-fees if use_calls else k1-net+fees

    Z  = np.random.normal(size=3000)
    ST = spot*np.exp((-0.5*iv_atm_sb**2)*T_sb + iv_atm_sb*np.sqrt(T_sb)*Z)
    mc = spread_pnl(ST,strategy,k1,k2,p1,p2,fees)*100*contracts
    ev_mc = np.mean(mc); pop = np.mean(mc>0)

    with col_r:
        pr = np.linspace(spot*0.7, spot*1.3, 300)
        pnl_c = spread_pnl(pr,strategy,k1,k2,p1,p2,fees)*100*contracts
        fig_sp = go.Figure()
        fig_sp.add_trace(go.Scatter(x=pr,y=pnl_c,fill="tozeroy",line=dict(color="#7b8cde",width=2),name="P&L"))
        fig_sp.add_hline(y=0,line_color="#6b7694")
        fig_sp.add_vline(x=spot,line_color="#f4b942",line_dash="dash",annotation_text="Spot")
        fig_sp.add_vline(x=be,  line_color="#00e5a0",line_dash="dot", annotation_text=f"B/E ${be:.2f}")
        fig_sp.update_layout(**PT,height=300,title=f"{strategy} Payoff",yaxis_title="P&L ($)")
        st.plotly_chart(fig_sp, use_container_width=True)

        fig_mc = go.Figure()
        fig_mc.add_trace(go.Histogram(x=mc,nbinsx=60,marker_color="#7b8cde",opacity=0.7))
        fig_mc.add_vline(x=0,    line_color="#6b7694",line_dash="dash")
        fig_mc.add_vline(x=ev_mc,line_color="#00e5a0",line_dash="dot",annotation_text=f"EV ${ev_mc:.0f}")
        fig_mc.update_layout(**PT,height=220,title="Monte Carlo P&L (n=3000)")
        st.plotly_chart(fig_mc, use_container_width=True)

    st.markdown('<div class="section-title">Trade Scorecard</div>', unsafe_allow_html=True)
    sc1,sc2,sc3,sc4,sc5,sc6 = st.columns(6)
    sc1.metric("Max Profit",  f"${max_p:,.0f}")
    sc2.metric("Max Loss",    f"${max_l:,.0f}")
    sc3.metric("R/R",         f"{max_p/max_l:.2f}x" if max_l else "—")
    sc4.metric("Breakeven",   f"${be:.2f}")
    sc5.metric("MC EV",       f"${ev_mc:,.0f}")
    sc6.metric("Prob Profit", f"{pop:.0%}")

    score=0; sigs=[]
    if ev_mc>0:  score+=1; sigs.append(("buy",  f"✅ Positive EV: ${ev_mc:,.0f}"))
    else:                  sigs.append(("sell", f"⚠️ Negative EV: ${ev_mc:,.0f}"))
    if vrp_sb>0.03 and "Bear" in strategy: score+=1; sigs.append(("buy","✅ Elevated VRP supports credit strategy"))
    elif vrp_sb<-0.03 and "Bull" in strategy: score+=1; sigs.append(("buy","✅ Cheap IV supports debit strategy"))
    else: sigs.append(("info",f"ℹ️ VRP = {vrp_sb:+.1%}"))
    if pop>0.55: score+=1; sigs.append(("buy", f"✅ PoP: {pop:.0%}"))
    else:                  sigs.append(("warn",f"⚠️ PoP: {pop:.0%}"))
    if max_l>0 and max_p/max_l>=0.5: score+=1; sigs.append(("buy", f"✅ R/R: {max_p/max_l:.2f}x"))
    else:                              sigs.append(("sell",f"⚠️ R/R: {max_p/max_l:.2f}x"))
    st.metric("Trade Score", f"{score} / 4")
    for t,txt in sigs:
        st.markdown(signal(txt,t), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — MOMENTUM
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="tab-header">Momentum Screener</div>', unsafe_allow_html=True)
    col_btn, col_info = st.columns([1,3])
    with col_btn:
        run_screen = st.button("🔍 Run Screener", type="primary")
    with col_info:
        st.markdown(f'<div style="color:#6b7694;font-size:12px;font-family:IBM Plex Mono;padding-top:8px;">{len(screen_tickers)} tickers · benchmark: {benchmark_input}</div>', unsafe_allow_html=True)

    if run_screen:
        df_screen = screen_momentum(screen_tickers, benchmark_input)
        if df_screen.empty:
            st.warning("No data returned.")
        else:
            fc1,fc2,fc3 = st.columns(3)
            min_mom  = fc1.slider("Min 3M Mom (%)", -50, 50, 0)
            min_rs   = fc2.slider("Min vs SPY 3M (%)", -30, 30, 0)
            near_hi  = fc3.checkbox("Within 10% of 52W High")
            df_f = df_screen[df_screen["3M %"]>=min_mom]
            df_f = df_f[df_f["vs SPY 3M"]>=min_rs]
            if near_hi: df_f = df_f[df_f["52W High%"]>=-10]

            styled = df_f.style\
                .applymap(color_pct, subset=["1M %","3M %","6M %","12M %","vs SPY 3M","vs SPY 12M","52W High%"])\
                .format({"Price":"${:.2f}","1M %":"{:+.1f}%","3M %":"{:+.1f}%","6M %":"{:+.1f}%",
                         "12M %":"{:+.1f}%","vs SPY 3M":"{:+.1f}%","vs SPY 12M":"{:+.1f}%",
                         "52W High%":"{:+.1f}%","52W Low%":"{:+.1f}%","MomScore":"{:.2f}"})
            st.dataframe(styled, use_container_width=True, height=450)

            # Time-series chart
            st.markdown('<div class="section-title">Time-Series Momentum — Top 10</div>', unsafe_allow_html=True)
            top10 = df_f.head(10)["Ticker"].tolist()
            palette = ["#00e5a0","#7b8cde","#f4b942","#ff4d6d","#a78bfa","#38bdf8","#fb923c","#4ade80","#f472b6","#94a3b8"]
            batch_ts = batch_download(top10, "1y")
            fig_ts = go.Figure()
            for i,sym in enumerate(top10):
                try:
                    c = extract_close(batch_ts, sym)
                    if c.empty: continue
                    nc = c/c.iloc[0]*100
                    fig_ts.add_trace(go.Scatter(x=nc.index,y=nc,name=sym,line=dict(color=palette[i%len(palette)],width=1.5)))
                except Exception: continue
            fig_ts.add_hline(y=100,line_color="#6b7694",line_dash="dot")
            fig_ts.update_layout(**PT,height=380,title="Normalized 1Y Price (Base=100)")
            st.plotly_chart(fig_ts, use_container_width=True)

            # Cross-sectional bar
            st.markdown('<div class="section-title">Cross-Sectional 3M Momentum — Top 20</div>', unsafe_allow_html=True)
            top20 = df_f.head(20)
            fig_cs = go.Figure(go.Bar(x=top20["Ticker"],y=top20["3M %"],
                marker_color=["#00e5a0" if v>0 else "#ff4d6d" for v in top20["3M %"]],
                text=top20["3M %"].map("{:+.1f}%".format),textposition="outside"))
            fig_cs.update_layout(**PT,height=300)
            st.plotly_chart(fig_cs, use_container_width=True)

            # Relative momentum
            st.markdown(f'<div class="section-title">Relative Momentum vs {relative_to}</div>', unsafe_allow_html=True)
            try:
                rel_h = get_hist(relative_to, "1y")
                if not rel_h.empty and len(rel_h)>=63:
                    rel_ret = (rel_h["Close"].iloc[-1]/rel_h["Close"].iloc[-63]-1)*100
                    rel20 = df_f.head(20).copy()
                    rel20["vs "+relative_to] = rel20["3M %"] - rel_ret
                    fig_rel = go.Figure(go.Bar(x=rel20["Ticker"],y=rel20["vs "+relative_to],
                        marker_color=["#00e5a0" if v>0 else "#ff4d6d" for v in rel20["vs "+relative_to]],
                        text=rel20["vs "+relative_to].map("{:+.1f}%".format),textposition="outside"))
                    fig_rel.add_hline(y=0,line_color="#6b7694")
                    fig_rel.update_layout(**PT,height=300,title=f"3M Alpha vs {relative_to}")
                    st.plotly_chart(fig_rel, use_container_width=True)
            except Exception:
                st.info(f"Could not load {relative_to}.")
    else:
        st.markdown('<div style="color:#6b7694;font-family:IBM Plex Mono;font-size:13px;padding:40px 0;text-align:center;">Press Run Screener to load data.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — SECTORS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="tab-header">Sector Rotation</div>', unsafe_allow_html=True)
    run_sect = st.button("📊 Load Sector Data", type="primary")
    if run_sect:
        df_sect = screen_sectors()
        if df_sect.empty:
            st.warning("No sector data.")
        else:
            styled_s = df_sect.style\
                .applymap(color_pct, subset=["1M %","3M %","6M %","12M %"])\
                .format({"Price":"${:.2f}","1M %":"{:+.1f}%","3M %":"{:+.1f}%",
                         "6M %":"{:+.1f}%","12M %":"{:+.1f}%","vs 52W High":"{:+.1f}%","MomScore":"{:.2f}"})
            st.dataframe(styled_s, use_container_width=True, hide_index=True)

            fig_sect = go.Figure()
            for period,color in [("1M %","#ff4d6d"),("3M %","#f4b942"),("6M %","#7b8cde"),("12M %","#00e5a0")]:
                fig_sect.add_trace(go.Bar(name=period,x=df_sect["Sector"],y=df_sect[period],marker_color=color))
            fig_sect.update_layout(**PT,height=380,barmode="group",title="Sector Returns by Period")
            st.plotly_chart(fig_sect, use_container_width=True)

            sect_batch = batch_download(list(SECTOR_ETFS.values()), "1y")
            palette2 = ["#00e5a0","#7b8cde","#f4b942","#ff4d6d","#a78bfa","#38bdf8","#fb923c","#4ade80","#f472b6","#94a3b8","#cbd5e1"]
            fig_sn = go.Figure()
            for i,(_,row) in enumerate(df_sect.iterrows()):
                try:
                    c = extract_close(sect_batch, row["ETF"])
                    if c.empty: continue
                    nc = c/c.iloc[0]*100
                    fig_sn.add_trace(go.Scatter(x=nc.index,y=nc,name=row["Sector"],line=dict(color=palette2[i%len(palette2)],width=2)))
                except Exception: continue
            fig_sn.add_hline(y=100,line_color="#6b7694",line_dash="dot")
            fig_sn.update_layout(**PT,height=420,title="Sector ETFs — Normalized 1Y (Base=100)")
            st.plotly_chart(fig_sn, use_container_width=True)

            fig_52 = go.Figure(go.Bar(x=df_sect["Sector"],y=df_sect["vs 52W High"],
                marker_color=["#00e5a0" if v>-5 else "#f4b942" if v>-15 else "#ff4d6d" for v in df_sect["vs 52W High"]],
                text=df_sect["vs 52W High"].map("{:+.1f}%".format),textposition="outside"))
            fig_52.add_hline(y=0,line_color="#6b7694")
            fig_52.update_layout(**PT,height=300,title="Distance from 52W High (%)")
            st.plotly_chart(fig_52, use_container_width=True)
    else:
        st.markdown('<div style="color:#6b7694;font-family:IBM Plex Mono;font-size:13px;padding:40px 0;text-align:center;">Press Load Sector Data.</div>', unsafe_allow_html=True)
