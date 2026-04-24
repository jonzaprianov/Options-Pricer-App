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
    "strike",
    "lastPrice",
    "impliedVolatility",
    "bs",
    "mispricing"
]].copy()

display_df.columns = [
    "Strike",
    "Market Price",
    "Implied Volatility",
    "Black-Scholes Value",
    "Mispricing"
]

st.dataframe(display_df.round(3))

st.subheader("🧠 Trade Quality Engine")

# Inputs
trade_type = st.selectbox("Trade Type", ["Long Call", "Short Call"])
strike = st.number_input("Strike", value=float(spot))
premium = st.number_input("Premium", value=5.0)
fees = st.number_input("Fees (total)", value=0.5)
slippage = st.number_input("Slippage", value=0.5)

# Use ATM IV
atm_iv = calls.iloc[(calls['strike']-spot).abs().argsort()[:1]]["impliedVolatility"].values[0]

# Expected move
EM = spot * atm_iv * np.sqrt(T)

# Simple payoff scenarios
upside_price = spot + EM
downside_price = spot - EM

def call_pnl(S, K, premium):
    return max(S - K, 0) - premium

pnl_up = call_pnl(upside_price, strike, premium)
pnl_down = call_pnl(downside_price, strike, premium)

# Expected value (very simplified)
EV = 0.5 * pnl_up + 0.5 * pnl_down - fees - slippage

st.metric("Expected Move", round(EM, 2))
st.metric("EV (Approx)", round(EV, 2))

if EV > 0:
    st.success("✅ Trade has positive expected value")
else:
    st.warning("⚠️ Trade not attractive after costs")

iv_mean = calls["impliedVolatility"].mean()
spread = iv_mean - current_rv

st.metric("IV - RV Spread", round(spread, 3))

score = 0

if EV > 0:
    score += 1
if spread > 0.05:
    score += 1
if percentile > 0.8:
    score += 1

st.metric("Trade Score (0-3)", score)

st.subheader("🎯 Monte Carlo EV Engine")

num_sims = st.slider("Simulations", 100, 5000, 1000)

sigma = atm_iv
mu = 0  # assume drift ~0 for short-term

Z = np.random.normal(size=num_sims)

ST = spot * np.exp((mu - 0.5 * sigma**2)*T + sigma*np.sqrt(T)*Z)

def call_pnl(S, K, premium):
    return np.maximum(S - K, 0) - premium

mc_pnl = call_pnl(ST, strike, premium)

EV_mc = np.mean(mc_pnl) - fees - slippage
prob_profit = np.mean(mc_pnl > 0)

st.metric("Monte Carlo EV", round(EV_mc, 2))
st.metric("Probability of Profit", round(prob_profit, 2))

if EV_mc > 0:
    st.success("Positive EV trade")
else:
    st.warning("Negative EV trade")

st.subheader("📊 Forward Vol Signal")

# short vs long RV
rv_short = returns.rolling(10).std() * np.sqrt(252)
rv_long = returns.rolling(60).std() * np.sqrt(252)

forward_vol = rv_short.iloc[-1] - rv_long.iloc[-1]

st.metric("Forward Vol Signal", round(forward_vol, 3))

if forward_vol > 0:
    st.info("Volatility expanding")
else:
    st.info("Volatility compressing")

st.subheader("⚡ Event Vol Signal")

iv_series = calls["impliedVolatility"]

iv_z = (iv_series.mean() - rv) / rv

st.metric("IV Z-Score (approx)", round(iv_z, 2))

if iv_z > 1:
    st.warning("Event-level IV → likely vol crush candidate")
elif iv_z < -1:
    st.success("IV depressed → potential vol expansion")
else:
    st.info("Normal IV regime")

st.subheader("📈 Position Risk Profile")

price_range = np.linspace(0.5*spot, 1.5*spot, 100)

pnl_curve = np.maximum(price_range - strike, 0) - premium

max_loss = np.min(pnl_curve)
max_gain = np.max(pnl_curve)

st.metric("Max Loss", round(max_loss,2))
st.metric("Max Gain", round(max_gain,2))

fig = px.line(x=price_range, y=pnl_curve,
              labels={"x":"Underlying Price","y":"PnL"})
fig.add_hline(y=0)

st.plotly_chart(fig, use_container_width=True)

st.subheader("🧠 Strategy Builder")

strategy = st.selectbox("Strategy", [
    "Long Call",
    "Bull Call Spread",
    "Bear Call Spread",
    "Long Straddle"
])

strike1 = st.number_input("Strike 1", value=float(spot))
strike2 = st.number_input("Strike 2", value=float(spot * 1.1))

premium1 = st.number_input("Premium 1", value=5.0)
premium2 = st.number_input("Premium 2", value=2.0)

price_range = np.linspace(0.5*spot, 1.5*spot, 200)

def payoff(strategy, S):
    if strategy == "Long Call":
        return np.maximum(S - strike1, 0) - premium1
    
    elif strategy == "Bull Call Spread":
        return (np.maximum(S - strike1, 0) - premium1) - \
               (np.maximum(S - strike2, 0) - premium2)
    
    elif strategy == "Bear Call Spread":
        return (np.maximum(S - strike2, 0) - premium2) - \
               (np.maximum(S - strike1, 0) - premium1)
    
    elif strategy == "Long Straddle":
        return (np.maximum(S - strike1, 0) + np.maximum(strike1 - S, 0)) - premium1

pnl = payoff(strategy, price_range)

fig = px.line(x=price_range, y=pnl,
              labels={"x":"Underlying Price","y":"PnL"},
              title="Strategy Payoff")

fig.add_hline(y=0)

st.plotly_chart(fig, use_container_width=True)

st.subheader("🎯 Strategy Monte Carlo")

num_sims = st.slider("Simulations", 500, 5000, 2000)

Z = np.random.normal(size=num_sims)
ST = spot * np.exp((0 - 0.5 * atm_iv**2)*T + atm_iv*np.sqrt(T)*Z)

mc_pnl = payoff(strategy, ST)

EV = np.mean(mc_pnl)
prob_profit = np.mean(mc_pnl > 0)

st.metric("Strategy EV", round(EV, 2))
st.metric("Prob Profit", round(prob_profit, 2))

st.subheader("📊 Backtest (Simplified)")

lookback = 60

hist = ticker.history(period="1y")
returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()

results = []

for i in range(lookback, len(returns)-1):
    window_vol = returns.iloc[i-lookback:i].std() * np.sqrt(252)
    
    # simulate next day move
    next_return = returns.iloc[i+1]
    next_price = hist["Close"].iloc[i] * np.exp(next_return)
    
    pnl_bt = payoff(strategy, next_price)
    results.append(pnl_bt)

results = np.array(results)

st.metric("Avg Backtest PnL", round(results.mean(),2))
st.metric("Win Rate", round(np.mean(results > 0),2))

st.subheader("🚀 Strategy Recommendation Engine")

recommendations = []

# Vol regime
if iv_mean > rv:
    recommendations.append("High IV → Favor credit spreads / short premium")
else:
    recommendations.append("Low IV → Favor long premium trades")

# Forward vol
if forward_vol > 0:
    recommendations.append("Vol expanding → favor long gamma")
else:
    recommendations.append("Vol compressing → favor short gamma")

# Monte Carlo
if EV > 0:
    recommendations.append("Positive EV strategy")
else:
    recommendations.append("Negative EV → avoid or hedge")

for rec in recommendations:
    st.write("•", rec)
