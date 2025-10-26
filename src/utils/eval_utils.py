"""
eval_utils.py
Utility helpers for evaluation: action parsing, price caching and forward-return lookup.
Now uses the price_data module for eodhd.com API + yfinance integration.

Functions:
- extract_action(text): returns normalized action token string (STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL) or 'UNKNOWN'
- load_price_cache(path): DEPRECATED - use PriceDataClient from price_data module
- get_forward_returns_for_sample(ticker, as_of_date, forward_days, cache): returns forward return or NaN
"""
import re
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.data.price_data import PriceDataClient, get_price_cliente, timedelta

ACTION_PATTERN = re.compile(r"<action>\s*(.*?)\s*</action>", re.IGNORECASE)

def extract_action(text: str) -> str:
    if not text or not isinstance(text, str):
        return "UNKNOWN"
    m = ACTION_PATTERN.search(text)
    if not m:
        # fallback: look for common keywords
        txt = text.upper()
        if "STRONG BUY" in txt or "STRONG_BUY" in txt:
            return "STRONG_BUY"
        if "BUY" in txt:
            return "BUY"
        if "HOLD" in txt:
            return "HOLD"
        if "SELL" in txt:
            return "SELL"
        if "STRONG SELL" in txt or "STRONG_SELL" in txt:
            return "STRONG_SELL"
        return "UNKNOWN"
    act = m.group(1).strip().upper().replace(" ", "_")
    return act

def load_price_cache(path: str):
    if path is None:
        return {}
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, parse_dates=["Date"])
    # expects columns: ticker, Date, Open, High, Low, Close, Adj Close, Volume
    cache = {}
    for t, g in df.groupby("ticker"):
        g_sorted = g.sort_values("Date").reset_index(drop=True)
        cache[t] = g_sorted
    return cache

def get_forward_returns_for_sample(ticker: str, as_of_date: str, forward_days: int, cache: dict):
    """
    Return forward return (close_{t+forward_days} - close_t) / close_t
    If ticker not in cache, return NaN
    If t not found exactly, pick next trading day >= as_of_date
    """
    if ticker is None or as_of_date is None:
        return np.nan
    try:
        if ticker not in cache:
            return np.nan
        df = cache[ticker]
        df_dates = pd.to_datetime(df["Date"])
        t0 = pd.to_datetime(as_of_date)
        # find first index >= t0
        idx = df_dates.searchsorted(t0)
        if idx >= len(df):
            return np.nan
        idx1 = idx + forward_days
        if idx1 >= len(df):
            return np.nan
        p0 = df.loc[idx, "Close"]
        p1 = df.loc[idx1, "Close"]
        if pd.isna(p0) or pd.isna(p1):
            return np.nan
        return float((p1 - p0) / p0)
    except Exception:
        return np.nan
