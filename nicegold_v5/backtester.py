import pandas as pd
from datetime import datetime
from nicegold_v5.risk import calc_lot
from nicegold_v5.exit import should_exit


def run_backtest(df: pd.DataFrame):
    capital = 100.0
    trades = []
    equity = []
    open_trade = None

    for i, row in df.iterrows():
        equity.append({"timestamp": row["timestamp"], "equity": capital})

        # --- Exit ---
        if open_trade:
            exit_now, reason = should_exit(open_trade, row)
            if exit_now:
                pnl = (row["close"] - open_trade["entry"] if open_trade["type"] == "buy"
                       else open_trade["entry"] - row["close"]) * open_trade["lot"] * 10
                capital += pnl
                open_trade["exit"] = row["timestamp"]
                open_trade["pnl"] = pnl
                trades.append(open_trade)
                open_trade = None

        # --- Entry ---
        if not open_trade and row.get("entry_signal") in ["buy", "sell"]:
            lot = calc_lot(capital)
            open_trade = {
                "entry": row["close"],
                "entry_time": row["timestamp"],
                "type": row["entry_signal"],
                "lot": lot
            }

    return pd.DataFrame(trades), pd.DataFrame(equity)
