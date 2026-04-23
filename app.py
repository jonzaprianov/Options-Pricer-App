import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime
import plotly.express as px

st.set_page_config(layout="wide")
st.title("🧠 Options Trading Dashboard V4")

# -------------------------
# Black-Scholes
# -------------------------
def bs_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S/K) + (r + sigma**2 / 2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

# -------------------------
# Helpers
# -------------------------
def time_to_expiry(exp):
    return max((datetime.strptime(exp, "%Y-%m-%d") - datetime.today()).days / 365, 0.001)

def realized_vol(ticker):
    hist = ticker.history(period="6mo")
    returns = np.log(hist["Close"]/hist["Close"].shift(1)).dropna()
    return np.sqrt(252)*returns.std()

def iv_percentile(ticker):
    hist = ticker.history(period="6mo")
    returns = np.log(hist["Close"]/hist["Close"].shift(1)).dropna()
    rv_series = returns.rolling(20).std() * np.sqrt(252)
    current_rv = rv_series.iloc[-1]
    return (rv_series < current_rv).mean()

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Settings")
ticker_input = st.sidebar.text_input("Ticker", "AAPL")
r = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 10.0, 4.5)/100

# -------------------------
# Load Data
# -------------------------
@st.cache_resource
def load_data(ticker):
    return yf.Ticker(ticker)

try:
    ticker = load_data(ticker_input)
    hist = ticker.history(period="1d")

    if hist.empty:
        st.error("No price data")
        st.stop()

    spot = hist["Close"].iloc[-1]
    expirations = ticker.options

    if not expirations:
        st.error("No options data")
        st.stop()

except Exception as e:
    st.error(e)
    st.stop()

# -------------------------
# METRICS
# -------------------------
rv = realized_vol(ticker)
ivp = iv_percentile(ticker)

col1, col2, col3 = st.columns(3)
col1.metric("Spot", round(spot,2))
col2.metric("Realized Vol", round(rv,3))
col3.metric("IV Percentile", round(ivp,2))

# -------------------------
# SELECT EXPIRATION
# -------------------------
selected_exp = st.selectbox("Expiration", expirations)
T = time_to_expiry(selected_exp)

chain = ticker.option_chain(selected_exp)
calls = chain.calls.copy().dropna(subset=["impliedVolatility"])

# -------------------------
# CALCULATIONS
# -------------------------
calls["intrinsic"] = np.maximum(spot - calls["strike"], 0)
calls["bs"] = calls.apply(lambda x: bs_call(
    spot, x["strike"], T, r, x["impliedVolatility"]), axis=1)

calls["mispricing"] = calls["lastPrice"] - calls["bs"]

# -------------------------
# IV SMILE
# -------------------------
st.subheader("IV Smile")
fig = px.scatter(calls, x="strike", y="impliedVolatility")
st.plotly_chart(fig, use_container_width=True)

# -------------------------
# STRATEGY ENGINE
# -------------------------
st.subheader("🚀 Trade Recommendations")

iv_mean = calls["impliedVolatility"].mean()

recommendations = []

# 1. VOL STRATEGY
if iv_mean - rv > 0.05:
    recommendations.append("🔻 SELL VOL → Consider short straddles / credit spreads")
elif iv_mean - rv < -0.05:
    recommendations.append("🔺 BUY VOL → Consider long straddles / calls")

# 2. MISPRICING
cheap = calls[calls["mispricing"] < -0.5]
rich = calls[calls["mispricing"] > 0.5]

if not cheap.empty:
    best = cheap.iloc[0]
    recommendations.append(f"💰 BUY CALL {best['strike']} (undervalued)")

if not rich.empty:
    best = rich.iloc[0]
    recommendations.append(f"⚠️ SELL CALL {best['strike']} (overvalued)")

# 3. SKEW
skew = calls.sort_values("strike")

if len(skew) > 5:
    low_iv = skew.iloc[:5]["impliedVolatility"].mean()
    high_iv = skew.iloc[-5:]["impliedVolatility"].mean()

    if high_iv > low_iv:
        recommendations.append("📉 CALL SKEW → upside expensive (sell calls / spreads)")
    else:
        recommendations.append("📈 PUT SKEW → downside expensive (sell puts)")

# -------------------------
# DISPLAY SIGNALS
# -------------------------
for rec in recommendations:
    st.write(rec)

# -------------------------
# OPTIONS TABLE
# -------------------------
st.subheader("Options Table")

display_df = calls[[
    "strike","lastPrice","impliedVolatility","bs","mispricing"
]].copy()

display_df.columns = [
    "Strike",
    "Market Price",
    "Implied Volatility",
    "Black-Scholes Value",
    "Mispricing"
]
st.dataframe(display_df.round(3))
]].round(3))
