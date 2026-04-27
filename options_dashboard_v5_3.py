import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    layout="wide",
    page_title="Options Flow Dashboard",
    page_icon="📈"
)

# ─────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0a0c10;
    color: #e0e4ef;
}

.stApp { background-color: #0a0c10; }

h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; letter-spacing: -0.5px; }

.metric-card {
    background: #111318;
    border: 1px solid #1e2230;
    border-radius: 8px;
    padding: 16px 20px;
    font-family: 'IBM Plex Mono', monospace;
}

.metric-label { font-size: 11px; color: #6b7694; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
.metric-value { font-size: 26px; font-weight: 700; color: #e0e4ef; }
.metric-delta { font-size: 12px; margin-top: 4px; }

.positive { color: #00e5a0; }
.negative { color: #ff4d6d; }
.neutral  { color: #7b8cde; }
.warning  { color: #f4b942; }

.signal-box {
    background: #111318;
    border-left: 3px solid #7b8cde;
    border-radius: 4px;
    padding: 12px 16px;
    margin: 6px 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
}

.signal-buy   { border-left-color: #00e5a0; }
.signal-sell  { border-left-color: #ff4d6d; }
.signal-warn  { border-left-color: #f4b942; }
.signal-info  { border-left-color: #7b8cde; }

.tab-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #6b7694;
    text-transform: uppercase;
    letter-spacing: 2px;
    padding: 8px 0 16px 0;
    border-bottom: 1px solid #1e2230;
    margin-bottom: 20px;
}

.section-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    color: #7b8cde;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin: 24px 0 12px 0;
    border-bottom: 1px solid #1e2230;
    padding-bottom: 8px;
}

div[data-testid="stMetric"] {
    background: #111318;
    border: 1px solid #1e2230;
    border-radius: 8px;
    padding: 14px 18px;
}
div[data-testid="stMetric"] label { color: #6b7694 !important; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #e0e4ef; font-family: 'IBM Plex Mono', monospace; }

.stSelectbox > div, .stTextInput > div { background: #111318 !important; border-color: #1e2230 !important; }
.stDataFrame { border: 1px solid #1e2230; border-radius: 8px; }

.stTabs [data-baseweb="tab-list"] { background: #0a0c10; border-bottom: 1px solid #1e2230; gap: 0; }
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: #6b7694;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 10px 20px;
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    color: #e0e4ef !important;
    border-bottom: 2px solid #7b8cde !important;
    background: transparent !important;
}

.momentum-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid #1e2230;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
}

.ticker-badge {
    background: #1e2230;
    border-radius: 4px;
    padding: 2px 8px;
    font-weight: 700;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
PLOT_THEME = dict(
    paper_bgcolor="#0a0c10",
    plot_bgcolor="#0a0c10",
    font=dict(family="IBM Plex Mono", color="#e0e4ef", size=11),
    xaxis=dict(gridcolor="#1e2230", zerolinecolor="#1e2230"),
    yaxis=dict(gridcolor="#1e2230", zerolinecolor="#1e2230"),
    margin=dict(l=40, r=20, t=40, b=40),
)

SECTOR_ETFS = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Energy": "XLE",
    "Consumer Disc.": "XLY",
    "Consumer Staples": "XLP",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication": "XLC",
}

SP500_SAMPLE = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","UNH","LLY",
    "JPM","V","XOM","MA","AVGO","HD","CVX","MRK","ABBV","PEP","KO","COST",
    "ADBE","WMT","CRM","TMO","ACN","MCD","NFLX","AMD","LIN","DHR","TXN",
    "NEE","PM","ORCL","RTX","HON","AMGN","QCOM","LOW","IBM","GE","CAT",
    "BA","GS","SBUX","INTU","ISRG","SPGI","BLK","NOW","DE","AMAT","LRCX",
    "ADI","PANW","GILD","MU","AXP","PLD","CI","SLB","EOG","PSX","VLO",
    "WFC","MS","C","BAC","USB","PNC","TFC","COF","CB","MMC","AON","ALL",
    "ZTS","REGN","VRTX","BMY","MRNA","BIIB","IDXX","EW","SYK","BSX","MDT",
    "ABT","JNJ","PFE","DXCM","ILMN","HCA","DGX","LH","CNC","HUM","CVS",
]

# ─────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────
def bs_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S/K) + (r + sigma**2 / 2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def bs_greeks(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return {}
    d1 = (np.log(S/K) + (r + sigma**2 / 2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r*T) * norm.cdf(d2)) / 365
    vega  = S * norm.pdf(d1) * np.sqrt(T) / 100
    return dict(delta=delta, gamma=gamma, theta=theta, vega=vega)

def time_to_expiry(exp):
    return max((datetime.strptime(exp, "%Y-%m-%d") - datetime.today()).days / 365, 0.001)

def _retry(fn, retries=3, wait=4):
    """Call fn(), retrying on rate-limit errors with exponential backoff."""
    for attempt in range(retries):
        try:
            result = fn()
            if result is not None and (not isinstance(result, pd.DataFrame) or not result.empty):
                return result
        except Exception as e:
            if "Too Many Requests" in str(e) or "429" in str(e) or "rate" in str(e).lower():
                if attempt < retries - 1:
                    time.sleep(wait * (2 ** attempt))
                    continue
            raise
    return pd.DataFrame()

@st.cache_data(ttl=900)   # 15 min cache — reduces re-fetches significantly
def get_hist(ticker_sym, period="1y"):
    def _fetch():
        t = yf.Ticker(ticker_sym)
        h = t.history(period=period, auto_adjust=True)
        return h
    return _retry(_fetch)

@st.cache_data(ttl=900)
def get_ticker_info(ticker_sym):
    t = yf.Ticker(ticker_sym)
    return t

@st.cache_data(ttl=900)
def batch_download(tickers, period="1y"):
    """Download all tickers in one yf.download() call — far fewer HTTP requests."""
    sess = _make_yf_session()
    tickers = list(tickers) if not isinstance(tickers, list) else tickers
    for attempt in range(3):
        try:
            raw = yf.download(
                tickers,
                period=period,
                auto_adjust=True,
                group_by="ticker",
                progress=False,
                threads=False,
                session=sess,
            )
            if not raw.empty:
                return raw
        except Exception:
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
    return pd.DataFrame()

def extract_close(batch_df, sym):
    """Pull the Close series for one ticker out of a batch download result."""
    try:
        if isinstance(batch_df.columns, pd.MultiIndex):
            return batch_df["Close"][sym].dropna()
        return batch_df["Close"].dropna()
    except Exception:
        return pd.Series(dtype=float)

def compute_hv(hist, window=21):
    ret = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
    return ret.rolling(window).std() * np.sqrt(252)

def compute_rv_stats(hist):
    ret = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
    rv_10  = ret.rolling(10).std().iloc[-1]  * np.sqrt(252)
    rv_21  = ret.rolling(21).std().iloc[-1]  * np.sqrt(252)
    rv_63  = ret.rolling(63).std().iloc[-1]  * np.sqrt(252)
    rv_126 = ret.rolling(126).std().iloc[-1] * np.sqrt(252)
    return rv_10, rv_21, rv_63, rv_126

def compute_atm_iv(calls, puts, spot):
    atm_c = calls.iloc[(calls["strike"] - spot).abs().argsort()[:1]]["impliedVolatility"]
    atm_p = puts.iloc[(puts["strike"]  - spot).abs().argsort()[:1]]["impliedVolatility"]
    vals = pd.concat([atm_c, atm_p]).dropna()
    return vals.mean() if not vals.empty else np.nan

def vol_risk_premium(iv, rv_21):
    return iv - rv_21

def iv_rank(iv_series, current_iv, window=252):
    s = iv_series.dropna().tail(window)
    if len(s) < 10:
        return np.nan
    return (current_iv - s.min()) / (s.max() - s.min()) * 100

def compute_call_skew(calls, spot):
    """25-delta call skew: OTM calls vs ATM IV, expressed in std devs vs historical"""
    if len(calls) < 5:
        return np.nan, np.nan
    atm_row = calls.iloc[(calls["strike"] - spot).abs().argsort()[:1]]
    atm_iv = atm_row["impliedVolatility"].values[0]
    otm_calls = calls[calls["strike"] > spot * 1.05]
    if otm_calls.empty:
        return np.nan, np.nan
    otm_iv = otm_calls.iloc[0]["impliedVolatility"]
    skew_raw = otm_iv - atm_iv
    # approximate historical skew distribution (using all strikes)
    iv_std = calls["impliedVolatility"].std()
    skew_z = skew_raw / iv_std if iv_std > 0 else np.nan
    return round(skew_raw, 4), round(skew_z, 2)

def momentum_score(hist, periods=[21, 63, 126, 252]):
    close = hist["Close"]
    scores = {}
    for p in periods:
        if len(close) >= p + 1:
            scores[p] = (close.iloc[-1] / close.iloc[-p] - 1) * 100
        else:
            scores[p] = np.nan
    return scores

def week52_proximity(hist):
    high = hist["Close"].tail(252).max()
    low  = hist["Close"].tail(252).min()
    cur  = hist["Close"].iloc[-1]
    pct_from_high = (cur / high - 1) * 100
    pct_from_low  = (cur / low  - 1) * 100
    return pct_from_high, pct_from_low, high, low

@st.cache_data(ttl=900)
def screen_momentum(tickers, benchmark="SPY"):
    all_syms = list(set(tickers + [benchmark]))

    with st.spinner(f"Downloading {len(all_syms)} tickers in batch..."):
        batch = batch_download(all_syms, period="1y")

    if batch.empty:
        return pd.DataFrame()

    def get_close(sym):
        return extract_close(batch, sym)

    bench_close = get_close(benchmark)
    if bench_close.empty or len(bench_close) < 63:
        return pd.DataFrame()

    bench_ret_63  = (bench_close.iloc[-1] / bench_close.iloc[-63]  - 1) * 100
    bench_ret_252 = (bench_close.iloc[-1] / bench_close.iloc[-252] - 1) * 100 if len(bench_close) >= 252 else np.nan

    results = []
    for sym in tickers:
        try:
            close = get_close(sym)
            if close.empty or len(close) < 63:
                continue
            cur = close.iloc[-1]

            def mom(p):
                return (close.iloc[-1] / close.iloc[-p] - 1) * 100 if len(close) >= p else np.nan

            m21, m63, m126, m252 = mom(21), mom(63), mom(126), mom(252)

            high52 = close.tail(252).max()
            low52  = close.tail(252).min()
            pct_high = (cur / high52 - 1) * 100
            pct_low  = (cur / low52  - 1) * 100

            results.append({
                "Ticker":     sym,
                "Price":      round(cur, 2),
                "1M %":       round(m21,  1) if not np.isnan(m21)  else np.nan,
                "3M %":       round(m63,  1) if not np.isnan(m63)  else np.nan,
                "6M %":       round(m126, 1) if not np.isnan(m126) else np.nan,
                "12M %":      round(m252, 1) if not np.isnan(m252) else np.nan,
                "vs SPY 3M":  round(m63  - bench_ret_63,  1) if not np.isnan(m63)  else np.nan,
                "vs SPY 12M": round(m252 - bench_ret_252, 1) if not np.isnan(m252) else np.nan,
                "52W High%":  round(pct_high, 1),
                "52W Low%":   round(pct_low,  1),
            })
        except Exception:
            continue

    df = pd.DataFrame(results)
    if df.empty:
        return df
    for col in ["1M %","3M %","6M %","12M %"]:
        df[f"{col}_rank"] = df[col].rank(pct=True)
    df["MomScore"] = df[["1M %_rank","3M %_rank","6M %_rank","12M %_rank"]].mean(axis=1).round(2)
    df = df.drop(columns=[c for c in df.columns if c.endswith("_rank")])
    return df.sort_values("MomScore", ascending=False)

@st.cache_data(ttl=900)
def screen_sector_momentum():
    syms = list(SECTOR_ETFS.values())
    names = list(SECTOR_ETFS.keys())

    batch = batch_download(syms, period="1y")
    if batch.empty:
        return pd.DataFrame()

    results = []
    for name, sym in zip(names, syms):
        try:
            close = extract_close(batch, sym)
            if close.empty or len(close) < 63:
                continue
            cur = close.iloc[-1]

            def mom(p):
                return (close.iloc[-1] / close.iloc[-p] - 1) * 100 if len(close) >= p else np.nan

            high52 = close.tail(252).max()
            pct_high = (cur / high52 - 1) * 100

            results.append({
                "Sector": name, "ETF": sym,
                "Price":  round(cur, 2),
                "1M %":   round(mom(21),  1),
                "3M %":   round(mom(63),  1),
                "6M %":   round(mom(126), 1),
                "12M %":  round(mom(252), 1),
                "vs 52W High": round(pct_high, 1),
            })
        except Exception:
            continue

    df = pd.DataFrame(results)
    if df.empty:
        return df
    for col in ["1M %","3M %","6M %","12M %"]:
        df[f"{col}_rank"] = df[col].rank(pct=True)
    df["MomScore"] = df[["1M %_rank","3M %_rank","6M %_rank","12M %_rank"]].mean(axis=1).round(2)
    df = df.drop(columns=[c for c in df.columns if c.endswith("_rank")])
    return df.sort_values("MomScore", ascending=False)

def signal_html(text, stype="info"):
    return f'<div class="signal-box signal-{stype}">{text}</div>'

def color_val(v, good_positive=True):
    if pd.isna(v): return "—"
    if v > 0:
        c = "positive" if good_positive else "negative"
    else:
        c = "negative" if good_positive else "positive"
    return f'<span class="{c}">{v:+.1f}%</span>'

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="display:flex; align-items:baseline; gap:12px; margin-bottom:4px;">
  <span style="font-family:'IBM Plex Mono',monospace; font-size:22px; font-weight:700; color:#e0e4ef;">OPTIONS FLOW</span>
  <span style="font-family:'IBM Plex Mono',monospace; font-size:11px; color:#6b7694; letter-spacing:3px; text-transform:uppercase;">Dashboard V5</span>
</div>
<div style="font-family:'IBM Plex Sans',sans-serif; font-size:13px; color:#6b7694; margin-bottom:24px;">
  Vertical Spread Analysis · Volatility Regime · Momentum Screening
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-family:IBM Plex Mono,monospace; font-size:11px; color:#6b7694; text-transform:uppercase; letter-spacing:2px; margin-bottom:16px;">Parameters</div>', unsafe_allow_html=True)
    ticker_input = st.text_input("Ticker", "AAPL").upper().strip()
    r = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 4.5) / 100
    st.markdown("---")
    st.markdown('<div style="font-family:IBM Plex Mono,monospace; font-size:11px; color:#6b7694; text-transform:uppercase; letter-spacing:2px; margin-bottom:12px;">Screener Settings</div>', unsafe_allow_html=True)
    screen_universe = st.selectbox("Universe", ["S&P 500 Sample (100)", "Custom Watchlist"])
    if screen_universe == "Custom Watchlist":
        custom_tickers = st.text_area("Tickers (comma-separated)", "AAPL,MSFT,NVDA,TSLA,AMZN")
        screen_tickers = [t.strip().upper() for t in custom_tickers.split(",") if t.strip()]
    else:
        screen_tickers = SP500_SAMPLE
    benchmark_input = st.text_input("Benchmark", "SPY")
    relative_to = st.text_input("Relative Momentum vs.", "QQQ")

# ─────────────────────────────────────────────
# LOAD TICKER DATA
# ─────────────────────────────────────────────
try:
    ticker = yf.Ticker(ticker_input)
    hist_1y = _retry(lambda: get_hist(ticker_input, "1y"))

    if hist_1y.empty:
        st.error(f"No price data for {ticker_input} — Yahoo Finance may be rate limiting. Wait 30s and refresh.")
        st.stop()

    spot = hist_1y["Close"].iloc[-1]

    opts = _retry(lambda: ticker.options)
    expirations = opts if opts else []

    if not expirations:
        st.error(f"No options data for {ticker_input}")
        st.stop()

except Exception as e:
    st.error(f"Error loading {ticker_input}: {e}. If this is a rate limit error, wait 30–60 seconds and refresh.")
    st.stop()

# Compute vol stats
rv_10, rv_21, rv_63, rv_126 = compute_rv_stats(hist_1y)
hv_series = compute_hv(hist_1y, 21)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tabs = st.tabs([
    "📊  Overview",
    "📉  Volatility",
    "🎯  Options Chain",
    "⚡  Spread Builder",
    "📈  Momentum",
    "🌐  Sector Rotation",
])

# ══════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="tab-header">Market Overview · ' + ticker_input + '</div>', unsafe_allow_html=True)

    # Quick metrics
    pct_chg_1d = (hist_1y["Close"].iloc[-1] / hist_1y["Close"].iloc[-2] - 1) * 100
    pct_chg_1m = (hist_1y["Close"].iloc[-1] / hist_1y["Close"].iloc[-21] - 1) * 100
    pct_chg_3m = (hist_1y["Close"].iloc[-1] / hist_1y["Close"].iloc[-63] - 1) * 100
    pct_high, pct_low, h52, l52 = week52_proximity(hist_1y)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Spot Price", f"${spot:.2f}", f"{pct_chg_1d:+.2f}% today")
    c2.metric("1M Return",  f"{pct_chg_1m:+.1f}%")
    c3.metric("3M Return",  f"{pct_chg_3m:+.1f}%")
    c4.metric("52W High",   f"${h52:.2f}", f"{pct_high:.1f}% away")
    c5.metric("52W Low",    f"${l52:.2f}", f"+{pct_low:.1f}% away")

    st.markdown("")

    # Price chart with volume
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(
        x=hist_1y.index,
        open=hist_1y["Open"], high=hist_1y["High"],
        low=hist_1y["Low"],   close=hist_1y["Close"],
        increasing_line_color="#00e5a0", decreasing_line_color="#ff4d6d",
        name="Price"
    ), row=1, col=1)
    # 20 & 50 MA
    fig.add_trace(go.Scatter(
        x=hist_1y.index, y=hist_1y["Close"].rolling(20).mean(),
        line=dict(color="#7b8cde", width=1), name="MA20"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=hist_1y.index, y=hist_1y["Close"].rolling(50).mean(),
        line=dict(color="#f4b942", width=1, dash="dot"), name="MA50"
    ), row=1, col=1)
    # Volume
    colors = ["#00e5a0" if c >= o else "#ff4d6d" for c, o in zip(hist_1y["Close"], hist_1y["Open"])]
    fig.add_trace(go.Bar(
        x=hist_1y.index, y=hist_1y["Volume"],
        marker_color=colors, opacity=0.6, name="Volume"
    ), row=2, col=1)

    fig.update_layout(**PLOT_THEME, height=480, showlegend=True,
                      legend=dict(orientation="h", y=1.02, x=0))
    fig.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # Vol snapshot at bottom
    st.markdown('<div class="section-title">Volatility Snapshot</div>', unsafe_allow_html=True)
    vc1, vc2, vc3, vc4 = st.columns(4)
    vc1.metric("10D Realized Vol", f"{rv_10:.1%}")
    vc2.metric("21D Realized Vol", f"{rv_21:.1%}")
    vc3.metric("63D Realized Vol", f"{rv_63:.1%}")
    vc4.metric("126D Realized Vol", f"{rv_126:.1%}")

# ══════════════════════════════════════════════
# TAB 2 — VOLATILITY
# ══════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="tab-header">Volatility Analysis · ' + ticker_input + '</div>', unsafe_allow_html=True)

    # Need ATM IV — get from nearest expiry
    sel_exp_vol = st.selectbox("Reference Expiration", expirations, key="vol_exp")
    T_vol = time_to_expiry(sel_exp_vol)

    try:
        chain_vol = ticker.option_chain(sel_exp_vol)
        calls_vol = chain_vol.calls.dropna(subset=["impliedVolatility"])
        puts_vol  = chain_vol.puts.dropna(subset=["impliedVolatility"])
        atm_iv = compute_atm_iv(calls_vol, puts_vol, spot)
    except Exception:
        atm_iv = np.nan

    vrp = vol_risk_premium(atm_iv, rv_21)

    # ── Key Metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("ATM IV (Selected Exp)", f"{atm_iv:.1%}" if not np.isnan(atm_iv) else "—")
    m2.metric("21D Realized Vol",      f"{rv_21:.1%}")
    m3.metric("Vol Risk Premium",      f"{vrp:+.1%}" if not np.isnan(vrp) else "—",
              help="IV minus 21D RV. Positive = IV elevated vs realized (short vol favored)")
    m4.metric("10D RV",  f"{rv_10:.1%}")
    m5.metric("63D RV",  f"{rv_63:.1%}")

    # ── VRP Signal
    st.markdown("")
    if not np.isnan(vrp):
        if vrp > 0.05:
            st.markdown(signal_html(f"📉 VOL RISK PREMIUM = {vrp:+.1%} → IV elevated vs RV. Selling premium is historically advantaged (credit spreads, iron condors).", "sell"), unsafe_allow_html=True)
        elif vrp < -0.05:
            st.markdown(signal_html(f"📈 VOL RISK PREMIUM = {vrp:+.1%} → IV cheap vs RV. Long gamma / long vol setups favored (debit spreads, long straddles).", "buy"), unsafe_allow_html=True)
        else:
            st.markdown(signal_html(f"⚖️ VOL RISK PREMIUM = {vrp:+.1%} → IV near fair value. Neutral regime.", "info"), unsafe_allow_html=True)

    st.markdown("")

    # ── HV vs IV Chart
    st.markdown('<div class="section-title">Historical vs Implied Volatility</div>', unsafe_allow_html=True)

    hv_10_series  = compute_hv(hist_1y, 10)
    hv_21_series  = compute_hv(hist_1y, 21)
    hv_63_series  = compute_hv(hist_1y, 63)

    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=hv_10_series.index, y=hv_10_series,
                                  name="10D HV",  line=dict(color="#ff4d6d", width=1.5)))
    fig_vol.add_trace(go.Scatter(x=hv_21_series.index, y=hv_21_series,
                                  name="21D HV",  line=dict(color="#f4b942", width=1.5)))
    fig_vol.add_trace(go.Scatter(x=hv_63_series.index, y=hv_63_series,
                                  name="63D HV",  line=dict(color="#7b8cde", width=1.5)))
    if not np.isnan(atm_iv):
        fig_vol.add_hline(y=atm_iv, line_color="#00e5a0", line_dash="dash",
                           annotation_text=f"ATM IV {atm_iv:.1%}", annotation_position="top right")
    fig_vol.update_layout(**PLOT_THEME, height=320, title="Realized Volatility Cones vs ATM IV",
                           yaxis_tickformat=".0%")
    st.plotly_chart(fig_vol, use_container_width=True)

    # ── IV vs RV Table
    rows = [
        ("10D RV",  rv_10,  atm_iv - rv_10),
        ("21D RV",  rv_21,  atm_iv - rv_21),
        ("63D RV",  rv_63,  atm_iv - rv_63),
        ("126D RV", rv_126, atm_iv - rv_126),
    ]
    df_ivrv = pd.DataFrame(rows, columns=["Tenor", "Realized Vol", "IV - RV (VRP)"])
    df_ivrv["Realized Vol"] = df_ivrv["Realized Vol"].map("{:.1%}".format)
    df_ivrv["IV - RV (VRP)"] = df_ivrv["IV - RV (VRP)"].map(lambda x: f"{x:+.1%}")
    st.dataframe(df_ivrv, use_container_width=True, hide_index=True)

    st.markdown("")

    # ── Call Skew
    st.markdown('<div class="section-title">Call Skew Analysis</div>', unsafe_allow_html=True)

    if not calls_vol.empty:
        skew_raw, skew_z = compute_call_skew(calls_vol, spot)

        sk1, sk2 = st.columns(2)
        sk1.metric("Call Skew (OTM 5% - ATM IV)", f"{skew_raw:+.3f}" if skew_raw else "—",
                   help="Positive = OTM calls more expensive than ATM (upside bid)")
        sk2.metric("Skew Z-Score (std devs vs chain)", f"{skew_z:+.2f}" if skew_z else "—",
                   help="Z-score of skew vs the cross-sectional IV dispersion of the chain")

        if skew_z:
            if skew_z > 1.5:
                st.markdown(signal_html(f"🔺 Call skew elevated ({skew_z:+.2f}σ) — upside is bid. Market pricing in tail-up risk. Consider selling OTM call spreads.", "sell"), unsafe_allow_html=True)
            elif skew_z < -1.5:
                st.markdown(signal_html(f"🔻 Call skew depressed ({skew_z:+.2f}σ) — upside cheap vs ATM. Consider buying debit call spreads.", "buy"), unsafe_allow_html=True)
            else:
                st.markdown(signal_html(f"⚖️ Call skew normal ({skew_z:+.2f}σ) — no structural edge from skew.", "info"), unsafe_allow_html=True)

        # IV Smile
        fig_smile = go.Figure()
        fig_smile.add_trace(go.Scatter(
            x=calls_vol["strike"], y=calls_vol["impliedVolatility"],
            mode="lines+markers", name="Call IV",
            line=dict(color="#00e5a0", width=2),
            marker=dict(size=5)
        ))
        fig_smile.add_trace(go.Scatter(
            x=puts_vol["strike"], y=puts_vol["impliedVolatility"],
            mode="lines+markers", name="Put IV",
            line=dict(color="#ff4d6d", width=2),
            marker=dict(size=5)
        ))
        fig_smile.add_vline(x=spot, line_color="#f4b942", line_dash="dash",
                            annotation_text="Spot", annotation_position="top right")
        fig_smile.update_layout(**PLOT_THEME, height=320, title="IV Smile — Calls & Puts",
                                yaxis_tickformat=".0%")
        st.plotly_chart(fig_smile, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3 — OPTIONS CHAIN
# ══════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="tab-header">Options Chain · ' + ticker_input + '</div>', unsafe_allow_html=True)

    sel_exp_chain = st.selectbox("Expiration", expirations, key="chain_exp")
    T_chain = time_to_expiry(sel_exp_chain)

    try:
        chain = ticker.option_chain(sel_exp_chain)
        calls = chain.calls.dropna(subset=["impliedVolatility"]).copy()
        puts  = chain.puts.dropna(subset=["impliedVolatility"]).copy()
    except Exception as e:
        st.error(f"Error loading chain: {e}")
        st.stop()

    atm_iv_chain = compute_atm_iv(calls, puts, spot)

    # Compute BS + Greeks on calls
    def enrich_calls(df, S, T, r):
        df = df.copy()
        df["bs"]        = df.apply(lambda x: bs_call(S, x["strike"], T, r, x["impliedVolatility"]), axis=1)
        df["mispricing"]= df["lastPrice"] - df["bs"]
        greeks = df.apply(lambda x: bs_greeks(S, x["strike"], T, r, x["impliedVolatility"]), axis=1)
        df["delta"] = greeks.apply(lambda g: g.get("delta", np.nan))
        df["gamma"] = greeks.apply(lambda g: g.get("gamma", np.nan))
        df["theta"] = greeks.apply(lambda g: g.get("theta", np.nan))
        df["vega"]  = greeks.apply(lambda g: g.get("vega",  np.nan))
        return df

    calls = enrich_calls(calls, spot, T_chain, r)

    # Summary metrics
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("DTE", f"{int(T_chain * 365)}d")
    s2.metric("ATM IV", f"{atm_iv_chain:.1%}" if not np.isnan(atm_iv_chain) else "—")
    s3.metric("Expected Move (1σ)", f"±${spot * atm_iv_chain * np.sqrt(T_chain):.2f}" if not np.isnan(atm_iv_chain) else "—")
    s4.metric("Calls / Puts", f"{len(calls)} / {len(puts)}")

    st.markdown("")

    # Display table
    display_cols = {
        "strike": "Strike", "lastPrice": "Mkt Price", "bs": "BS Value",
        "mispricing": "Mispricing", "impliedVolatility": "IV",
        "delta": "Δ Delta", "gamma": "Γ Gamma", "theta": "Θ Theta", "vega": "ν Vega",
        "volume": "Volume", "openInterest": "OI"
    }
    available = [c for c in display_cols.keys() if c in calls.columns]
    show_df = calls[available].rename(columns=display_cols).round(4)

    # Highlight ATM
    atm_strike = calls.iloc[(calls["strike"] - spot).abs().argsort()[:1]]["strike"].values[0]

    st.dataframe(
        show_df.style.apply(
            lambda row: ["background-color: #1a2040" if row["Strike"] == atm_strike else "" for _ in row],
            axis=1
        ),
        use_container_width=True,
        height=400
    )

    # Mispricing chart
    st.markdown('<div class="section-title">Market Price vs Black-Scholes</div>', unsafe_allow_html=True)
    fig_mis = go.Figure()
    fig_mis.add_trace(go.Bar(
        x=calls["strike"], y=calls["mispricing"],
        marker_color=["#00e5a0" if v < 0 else "#ff4d6d" for v in calls["mispricing"]],
        name="Mispricing (Mkt - BS)"
    ))
    fig_mis.add_hline(y=0, line_color="#6b7694")
    fig_mis.add_vline(x=spot, line_color="#f4b942", line_dash="dash", annotation_text="Spot")
    fig_mis.update_layout(**PLOT_THEME, height=260, title="Call Mispricing by Strike")
    st.plotly_chart(fig_mis, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 4 — SPREAD BUILDER
# ══════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="tab-header">Vertical Spread Builder · ' + ticker_input + '</div>', unsafe_allow_html=True)

    sel_exp_sb = st.selectbox("Expiration", expirations, key="sb_exp")
    T_sb = time_to_expiry(sel_exp_sb)

    try:
        chain_sb = ticker.option_chain(sel_exp_sb)
        calls_sb = chain_sb.calls.dropna(subset=["impliedVolatility"]).copy()
        puts_sb  = chain_sb.puts.dropna(subset=["impliedVolatility"]).copy()
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

    atm_iv_sb = compute_atm_iv(calls_sb, puts_sb, spot)
    vrp_sb = vol_risk_premium(atm_iv_sb, rv_21)

    col_l, col_r = st.columns([1, 2])

    with col_l:
        strategy = st.selectbox("Strategy", [
            "Bull Call Spread", "Bear Call Spread",
            "Bull Put Spread",  "Bear Put Spread",
        ])
        trade_type = "calls" if "Call" in strategy else "puts"
        chain_use = calls_sb if trade_type == "calls" else puts_sb
        strikes_avail = sorted(chain_use["strike"].unique())

        k1 = st.selectbox("Long Strike", strikes_avail, index=min(len(strikes_avail)//2, len(strikes_avail)-1), key="k1")
        k2 = st.selectbox("Short Strike", strikes_avail, index=min(len(strikes_avail)//2+2, len(strikes_avail)-1), key="k2")

        p1_row = chain_use[chain_use["strike"] == k1]
        p2_row = chain_use[chain_use["strike"] == k2]
        p1_default = float(p1_row["lastPrice"].values[0]) if not p1_row.empty else 2.0
        p2_default = float(p2_row["lastPrice"].values[0]) if not p2_row.empty else 1.0

        p1 = st.number_input("Long Leg Premium", value=round(p1_default, 2), step=0.05)
        p2 = st.number_input("Short Leg Premium", value=round(p2_default, 2), step=0.05)
        fees = st.number_input("Fees + Slippage", value=0.20, step=0.05)
        contracts = st.number_input("Contracts (x100 shares)", value=1, min_value=1)

    # Spread math
    if "Bull Call" in strategy or "Bull Put" in strategy:
        net_debit  = p1 - p2  # paying for spread
        max_profit = (abs(k2 - k1) - net_debit - fees) * 100 * contracts
        max_loss   = (net_debit + fees) * 100 * contracts
        breakeven  = k1 + net_debit + fees if "Call" in strategy else k2 - net_debit - fees
    else:
        net_credit = p2 - p1
        max_profit = (net_credit - fees) * 100 * contracts
        max_loss   = (abs(k2 - k1) - net_credit + fees) * 100 * contracts
        breakeven  = k2 + net_credit - fees if "Call" in strategy else k1 - net_credit + fees

    # Monte Carlo
    num_sims = 3000
    Z  = np.random.normal(size=num_sims)
    ST = spot * np.exp((-0.5 * atm_iv_sb**2) * T_sb + atm_iv_sb * np.sqrt(T_sb) * Z)

    def spread_payoff(S_arr, strategy, k1, k2, p1, p2, fees):
        if "Bull Call" in strategy:
            return np.maximum(S_arr-k1,0) - np.maximum(S_arr-k2,0) - (p1-p2) - fees
        elif "Bear Call" in strategy:
            return np.maximum(S_arr-k2,0) - np.maximum(S_arr-k1,0) + (p2-p1) - fees
        elif "Bull Put" in strategy:
            return -(np.maximum(k1-S_arr,0) - np.maximum(k2-S_arr,0)) + (p2-p1) - fees
        elif "Bear Put" in strategy:
            return np.maximum(k1-S_arr,0) - np.maximum(k2-S_arr,0) - (p1-p2) - fees
        return np.zeros_like(S_arr)

    mc_pnl = spread_payoff(ST, strategy, k1, k2, p1, p2, fees) * 100 * contracts
    ev_mc  = np.mean(mc_pnl)
    pop    = np.mean(mc_pnl > 0)

    with col_r:
        # Payoff chart
        price_range = np.linspace(spot * 0.7, spot * 1.3, 300)
        pnl_curve   = spread_payoff(price_range, strategy, k1, k2, p1, p2, fees) * 100 * contracts

        fig_sp = go.Figure()
        fill_colors = ["#00e5a0" if v > 0 else "#ff4d6d" for v in pnl_curve]
        fig_sp.add_trace(go.Scatter(
            x=price_range, y=pnl_curve,
            fill="tozeroy",
            line=dict(color="#7b8cde", width=2),
            name="P&L"
        ))
        fig_sp.add_hline(y=0, line_color="#6b7694")
        fig_sp.add_vline(x=spot,      line_color="#f4b942", line_dash="dash", annotation_text="Spot")
        fig_sp.add_vline(x=breakeven, line_color="#00e5a0", line_dash="dot",  annotation_text=f"B/E ${breakeven:.2f}")
        fig_sp.add_vline(x=k1, line_color="#7b8cde", line_width=1)
        fig_sp.add_vline(x=k2, line_color="#7b8cde", line_width=1)
        fig_sp.update_layout(**PLOT_THEME, height=300, title=f"{strategy} Payoff at Expiration",
                              yaxis_title="P&L ($)")
        st.plotly_chart(fig_sp, use_container_width=True)

        # MC distribution
        fig_mc = go.Figure()
        fig_mc.add_trace(go.Histogram(
            x=mc_pnl, nbinsx=60,
            marker_color="#7b8cde", opacity=0.7,
            name="Simulated P&L"
        ))
        fig_mc.add_vline(x=0,     line_color="#6b7694", line_dash="dash")
        fig_mc.add_vline(x=ev_mc, line_color="#00e5a0", line_dash="dot",
                          annotation_text=f"EV ${ev_mc:.0f}")
        fig_mc.update_layout(**PLOT_THEME, height=220, title=f"Monte Carlo P&L Distribution (n={num_sims})")
        st.plotly_chart(fig_mc, use_container_width=True)

    # ── Trade Scorecard
    st.markdown('<div class="section-title">Trade Scorecard</div>', unsafe_allow_html=True)
    sc1, sc2, sc3, sc4, sc5, sc6 = st.columns(6)
    sc1.metric("Max Profit",  f"${max_profit:,.0f}")
    sc2.metric("Max Loss",    f"${max_loss:,.0f}")
    sc3.metric("R/R Ratio",   f"{max_profit/max_loss:.2f}x" if max_loss else "—")
    sc4.metric("Breakeven",   f"${breakeven:.2f}")
    sc5.metric("MC EV",       f"${ev_mc:,.0f}")
    sc6.metric("Prob Profit", f"{pop:.0%}")

    # ── Signals
    st.markdown("")
    score = 0
    sigs  = []

    if ev_mc > 0:
        score += 1
        sigs.append(("buy",  f"✅ Positive EV: ${ev_mc:,.0f}"))
    else:
        sigs.append(("sell", f"⚠️ Negative EV: ${ev_mc:,.0f}"))

    if vrp_sb > 0.03 and ("Bear" in strategy or "short" in strategy.lower()):
        score += 1
        sigs.append(("buy", f"✅ VRP = {vrp_sb:+.1%} — elevated IV supports credit strategy"))
    elif vrp_sb < -0.03 and ("Bull" in strategy):
        score += 1
        sigs.append(("buy", f"✅ VRP = {vrp_sb:+.1%} — cheap IV supports debit strategy"))
    else:
        sigs.append(("info", f"ℹ️ VRP = {vrp_sb:+.1%}"))

    if pop > 0.55:
        score += 1
        sigs.append(("buy", f"✅ Prob of Profit: {pop:.0%}"))
    else:
        sigs.append(("warn", f"⚠️ Prob of Profit: {pop:.0%}"))

    if max_loss > 0 and max_profit / max_loss >= 0.5:
        score += 1
        sigs.append(("buy", f"✅ R/R Ratio: {max_profit/max_loss:.2f}x (acceptable)"))
    else:
        sigs.append(("sell", f"⚠️ R/R Ratio: {max_profit/max_loss:.2f}x (poor)"))

    st.metric("📊 Trade Score", f"{score} / 4")
    for stype, txt in sigs:
        st.markdown(signal_html(txt, stype), unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 5 — MOMENTUM SCREENER
# ══════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="tab-header">Momentum Screener · S&P 500 Large Caps</div>', unsafe_allow_html=True)

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_screen = st.button("🔍 Run Screener", type="primary")
    with col_info:
        st.markdown(f'<div style="color:#6b7694; font-size:12px; font-family: IBM Plex Mono; padding-top:8px;">Universe: {len(screen_tickers)} tickers · Benchmark: {benchmark_input}</div>', unsafe_allow_html=True)

    if run_screen:
        with st.spinner("Screening universe..."):
            df_screen = screen_momentum(screen_tickers, benchmark_input)

        if df_screen.empty:
            st.warning("No data returned.")
        else:
            # ── Filters
            fc1, fc2, fc3 = st.columns(3)
            min_mom = fc1.slider("Min 3M Momentum (%)", -50, 50, 0)
            min_rs  = fc2.slider("Min vs SPY 3M (%)", -30, 30, 0)
            near_high = fc3.checkbox("Near 52W High (within 10%)", value=False)

            df_f = df_screen[df_screen["3M %"] >= min_mom]
            df_f = df_f[df_f["vs SPY 3M"] >= min_rs]
            if near_high:
                df_f = df_f[df_f["52W High%"] >= -10]

            st.markdown(f'<div style="color:#6b7694; font-size:11px; font-family:IBM Plex Mono; margin-bottom:8px;">{len(df_f)} tickers match filters</div>', unsafe_allow_html=True)

            # Color-code the dataframe
            def color_pct(val):
                if pd.isna(val): return ""
                if val > 5:  return "color: #00e5a0"
                if val < -5: return "color: #ff4d6d"
                return "color: #f4b942"

            styled = df_f.style\
                .applymap(color_pct, subset=["1M %","3M %","6M %","12M %","vs SPY 3M","vs SPY 12M","52W High%"])\
                .format({
                    "Price": "${:.2f}",
                    "1M %": "{:+.1f}%",  "3M %": "{:+.1f}%",
                    "6M %": "{:+.1f}%",  "12M %": "{:+.1f}%",
                    "vs SPY 3M": "{:+.1f}%", "vs SPY 12M": "{:+.1f}%",
                    "52W High%": "{:+.1f}%", "52W Low%": "{:+.1f}%",
                    "MomScore": "{:.2f}"
                })

            st.dataframe(styled, use_container_width=True, height=450)

            # ── Time-series momentum chart (top 10)
            st.markdown('<div class="section-title">Time-Series Momentum — Top 10 by Score</div>', unsafe_allow_html=True)
            top10 = df_f.head(10)["Ticker"].tolist()

            fig_ts = go.Figure()
            palette = ["#00e5a0","#7b8cde","#f4b942","#ff4d6d","#a78bfa",
                       "#38bdf8","#fb923c","#4ade80","#f472b6","#94a3b8"]

            # reuse the batch already fetched during screening
            batch_ts = batch_download(top10 + [benchmark_input], period="1y")
            for i, sym in enumerate(top10):
                try:
                    close = extract_close(batch_ts, sym)
                    if close.empty:
                        continue
                    norm_close = close / close.iloc[0] * 100
                    fig_ts.add_trace(go.Scatter(
                        x=norm_close.index, y=norm_close,
                        name=sym, line=dict(color=palette[i % len(palette)], width=1.5)
                    ))
                except Exception:
                    continue

            fig_ts.add_hline(y=100, line_color="#6b7694", line_dash="dot", annotation_text="Baseline")
            fig_ts.update_layout(**PLOT_THEME, height=380, title="Normalized 1Y Price (Base=100)")
            st.plotly_chart(fig_ts, use_container_width=True)

            # ── Cross-sectional momentum bar chart
            st.markdown('<div class="section-title">Cross-Sectional 3M Momentum — Top 20</div>', unsafe_allow_html=True)
            top20 = df_f.head(20).copy()
            fig_cs = go.Figure(go.Bar(
                x=top20["Ticker"], y=top20["3M %"],
                marker_color=["#00e5a0" if v > 0 else "#ff4d6d" for v in top20["3M %"]],
                text=top20["3M %"].map("{:+.1f}%".format),
                textposition="outside"
            ))
            fig_cs.update_layout(**PLOT_THEME, height=300)
            st.plotly_chart(fig_cs, use_container_width=True)

            # ── Relative Momentum
            st.markdown(f'<div class="section-title">Relative Momentum vs {relative_to}</div>', unsafe_allow_html=True)
            try:
                rel_hist = _retry(lambda: get_hist(relative_to, "1y"))
                if not rel_hist.empty and len(rel_hist) >= 63:
                    rel_ret = (rel_hist["Close"].iloc[-1] / rel_hist["Close"].iloc[-63] - 1) * 100
                    rel_top20 = df_f.head(20).copy()
                    rel_top20["vs " + relative_to] = rel_top20["3M %"] - rel_ret

                    fig_rel = go.Figure(go.Bar(
                        x=rel_top20["Ticker"],
                        y=rel_top20["vs " + relative_to],
                        marker_color=["#00e5a0" if v > 0 else "#ff4d6d" for v in rel_top20["vs " + relative_to]],
                        text=rel_top20["vs " + relative_to].map("{:+.1f}%".format),
                        textposition="outside"
                    ))
                    fig_rel.add_hline(y=0, line_color="#6b7694")
                    fig_rel.update_layout(**PLOT_THEME, height=300,
                                          title=f"3M Alpha vs {relative_to}")
                    st.plotly_chart(fig_rel, use_container_width=True)
                else:
                    st.info(f"Could not load enough data for {relative_to}.")
            except Exception:
                st.info(f"Could not load {relative_to} data.")

    else:
        st.markdown('<div style="color:#6b7694; font-family:IBM Plex Mono; font-size:13px; padding:40px 0; text-align:center;">Press Run Screener to load momentum data.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 6 — SECTOR ROTATION
# ══════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="tab-header">Sector Rotation & Momentum</div>', unsafe_allow_html=True)

    run_sector = st.button("📊 Load Sector Data", type="primary")

    if run_sector:
        with st.spinner("Loading sector ETFs..."):
            df_sect = screen_sector_momentum()

        if df_sect.empty:
            st.warning("No sector data.")
        else:
            # Sector heatmap-style cards
            st.markdown('<div class="section-title">Sector Momentum Leaderboard</div>', unsafe_allow_html=True)

            styled_sect = df_sect.style\
                .applymap(color_pct, subset=["1M %","3M %","6M %","12M %"])\
                .format({
                    "Price": "${:.2f}",
                    "1M %": "{:+.1f}%", "3M %": "{:+.1f}%",
                    "6M %": "{:+.1f}%", "12M %": "{:+.1f}%",
                    "vs 52W High": "{:+.1f}%", "MomScore": "{:.2f}"
                })
            st.dataframe(styled_sect, use_container_width=True, hide_index=True)

            # Sector momentum bars
            fig_sect = go.Figure()
            for period, col, color in [("1M %","1M %","#ff4d6d"), ("3M %","3M %","#f4b942"),
                                        ("6M %","6M %","#7b8cde"), ("12M %","12M %","#00e5a0")]:
                fig_sect.add_trace(go.Bar(
                    name=period, x=df_sect["Sector"], y=df_sect[col],
                    marker_color=color
                ))
            fig_sect.update_layout(**PLOT_THEME, height=380, barmode="group",
                                   title="Sector Returns by Period")
            st.plotly_chart(fig_sect, use_container_width=True)

            # Normalized price chart — reuse batch already fetched in screen_sector_momentum
            st.markdown('<div class="section-title">Sector ETF 1Y Performance (Normalized)</div>', unsafe_allow_html=True)
            fig_snorm = go.Figure()
            palette2 = ["#00e5a0","#7b8cde","#f4b942","#ff4d6d","#a78bfa",
                        "#38bdf8","#fb923c","#4ade80","#f472b6","#94a3b8","#cbd5e1"]
            sect_batch = batch_download(list(SECTOR_ETFS.values()), period="1y")
            for i, (_, row) in enumerate(df_sect.iterrows()):
                try:
                    close = extract_close(sect_batch, row["ETF"])
                    if close.empty:
                        continue
                    nc = close / close.iloc[0] * 100
                    fig_snorm.add_trace(go.Scatter(
                        x=nc.index, y=nc,
                        name=row["Sector"],
                        line=dict(color=palette2[i % len(palette2)], width=2)
                    ))
                except Exception:
                    continue
            fig_snorm.add_hline(y=100, line_color="#6b7694", line_dash="dot")
            fig_snorm.update_layout(**PLOT_THEME, height=420, title="Sector ETFs — Normalized 1Y (Base=100)")
            st.plotly_chart(fig_snorm, use_container_width=True)

            # 52W High proximity
            st.markdown('<div class="section-title">52-Week High Proximity by Sector</div>', unsafe_allow_html=True)
            fig_52 = go.Figure(go.Bar(
                x=df_sect["Sector"],
                y=df_sect["vs 52W High"],
                marker_color=["#00e5a0" if v > -5 else "#f4b942" if v > -15 else "#ff4d6d" for v in df_sect["vs 52W High"]],
                text=df_sect["vs 52W High"].map("{:+.1f}%".format),
                textposition="outside"
            ))
            fig_52.add_hline(y=0, line_color="#6b7694")
            fig_52.update_layout(**PLOT_THEME, height=300, title="Distance from 52W High (%)")
            st.plotly_chart(fig_52, use_container_width=True)
    else:
        st.markdown('<div style="color:#6b7694; font-family:IBM Plex Mono; font-size:13px; padding:40px 0; text-align:center;">Press Load Sector Data to analyze sector rotation.</div>', unsafe_allow_html=True)
