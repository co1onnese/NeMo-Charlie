#!/usr/bin/env python3
"""
trading_backtest.py
Simple event-driven backtest engine that reads evaluation CSV (predictions + realized returns)
and simulates a portfolio.

Features:
- Configurable initial capital, transaction cost (bps), slippage (bps)
- Position sizing: fixed percent per signal or equal-weight per day
- Entry price assumptions: next-day open simulated via realized returns mapping (approx)
- Outputs equity curve and metrics JSON/CSV

Usage:
  python src/backtest/trading_backtest.py --eval_csv results/eval_results.csv --out backtests/run1.csv --config configs/backtest_config.yaml
"""
import argparse
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def compute_metrics(equity_series, daily_returns):
    total_return = equity_series.iloc[-1] / equity_series.iloc[0] - 1.0
    ann_ret = (1 + total_return) ** (252 / len(daily_returns)) - 1 if len(daily_returns) > 0 else np.nan
    ann_vol = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else np.nan
    sharpe = ann_ret / ann_vol if ann_vol and ann_vol > 0 else np.nan
    drawdown = (equity_series / equity_series.cummax() - 1).min()
    return {"total_return": total_return, "annualized_return": ann_ret, "annualized_vol": ann_vol, "sharpe": sharpe, "max_drawdown": drawdown}

def run_backtest(eval_df, cfg):
    # Expect eval_df to have: ticker, as_of_date, pred_action, realized_return
    eval_df = eval_df.copy()
    # drop rows without returns
    eval_df = eval_df.dropna(subset=["realized_return"])
    # sort by date
    eval_df["as_of_date"] = pd.to_datetime(eval_df["as_of_date"])
    eval_df = eval_df.sort_values("as_of_date").reset_index(drop=True)

    initial_cash = cfg.get("initial_cash", 1_000_000)
    trans_cost_bps = cfg.get("transaction_cost_bps", 5)
    slippage_bps = cfg.get("slippage_bps", 10)
    position_sizing = cfg.get("position_sizing", "fixed_pct")
    fixed_pct_value = cfg.get("fixed_pct_value", 0.02)
    allow_shorts = cfg.get("allow_shorts", False)

    # We'll simulate day-by-day. For each day, collect predictions and open/close positions.
    # For simplicity: for each prediction on day t for ticker i, we allocate weight w and realize return over forward window as precomputed realized_return.
    dates = sorted(eval_df["as_of_date"].unique())
    equity = initial_cash
    equity_history = []
    daily_returns = []
    positions_history = []

    for date in dates:
        day_df = eval_df[eval_df["as_of_date"] == date]
        n_signals = len(day_df)
        if n_signals == 0:
            equity_history.append(equity)
            daily_returns.append(0.0)
            positions_history.append([])
            continue
        # determine allocation per signal
        if position_sizing == "fixed_pct":
            allocations = [fixed_pct_value] * n_signals
        elif position_sizing == "equal_weight":
            allocations = [1.0 / n_signals] * n_signals
        else:
            allocations = [fixed_pct_value] * n_signals

        day_pnl = 0.0
        trades = []
        for (idx, row), alloc in zip(day_df.iterrows(), allocations):
            weight = alloc
            if row["pred_action"] in ("SELL", "STRONG_SELL") and not allow_shorts:
                # skip or invert to no-position
                continue
            # entry notional
            notional = equity * weight
            # gross return from realized_return (already computed)
            gross_ret = float(row["realized_return"])
            # apply slippage and transaction costs (approx bps)
            slippage = slippage_bps / 10000.0
            trans_cost = trans_cost_bps / 10000.0
            # net return
            net_ret = gross_ret - slippage - trans_cost
            pnl = notional * net_ret
            day_pnl += pnl
            trades.append({"ticker": row["ticker"], "notional": notional, "gross_ret": gross_ret, "net_ret": net_ret, "pnl": pnl})
        prev_equity = equity
        equity = equity + day_pnl
        daily_returns.append((equity - prev_equity) / prev_equity if prev_equity != 0 else 0.0)
        equity_history.append(equity)
        positions_history.append(trades)

    equity_series = pd.Series(equity_history, index=dates)
    metrics = compute_metrics(equity_series, pd.Series(daily_returns))
    return equity_series, metrics, positions_history

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_csv", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    df = pd.read_csv(args.eval_csv, parse_dates=["as_of_date"])
    equity_series, metrics, positions = run_backtest(df, cfg)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # save equity series as CSV and metrics as JSON
    equity_df = pd.DataFrame({"date": equity_series.index, "equity": equity_series.values})
    equity_df.to_csv(args.out, index=False)
    with open(args.out + ".metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    print("Backtest saved:", args.out)
    print("Metrics:", metrics)

if __name__ == "__main__":
    main()
