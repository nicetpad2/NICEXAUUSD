import pandas as pd
from datetime import datetime
from nicegold_v5.risk import calc_lot
from nicegold_v5.exit import should_exit
from tqdm import tqdm
import time


def run_backtest(df: pd.DataFrame):
    """Backtest แบบง่ายพร้อมแถบสถานะและบันทึกเวลา"""
    capital = 100.0
    trades = []
    equity = []
    open_trade = None
    start = time.time()

    for i, row in tqdm(df.iterrows(), total=len(df), desc="⏱️ Running Backtest", unit="rows"):
        ts = row["timestamp"]
        price = row["close"]
        equity.append({"timestamp": ts, "equity": capital})

        if open_trade:
            exit_now, reason = should_exit(open_trade, row)
            if exit_now:
                pnl = (price - open_trade["entry"] if open_trade["type"] == "buy"
                       else open_trade["entry"] - price) * open_trade["lot"] * 10
                capital += pnl
                open_trade["exit"] = ts
                open_trade["pnl"] = pnl
                trades.append(open_trade)
                open_trade = None

        if not open_trade and row.get("entry_signal") in ["buy", "sell"]:
            lot = calc_lot(capital)
            open_trade = {
                "entry": price,
                "entry_time": ts,
                "type": row.get("entry_signal"),
                "lot": lot
            }

    end = time.time()
    print(
        f"⏱️ Backtest Duration: {end - start:.2f}s | Trades: {len(trades)} | Avg per row: {(end - start)/len(df):.6f}s"
    )
    return pd.DataFrame(trades), pd.DataFrame(equity)
