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
        ts = pd.to_datetime(row["timestamp"])
        price = row["close"]
        equity.append({"timestamp": ts, "equity": capital})

        if open_trade:
            gain = price - open_trade["entry"] if open_trade["type"] == "buy" else open_trade["entry"] - price
            sl_dist = row.get("atr", 1.0) * 1.5
            tp1 = sl_dist * 1.5
            tp2 = sl_dist * 2.5

            if not open_trade.get("tp1_hit") and gain >= tp1:
                partial_lot = open_trade["lot"] * 0.5
                partial_pnl = tp1 * open_trade["lot"] * 10 * 0.5
                capital += partial_pnl
                trades.append({
                    "entry_time": open_trade["entry_time"],
                    "exit_time": ts,
                    "entry": open_trade["entry"],
                    "exit": price,
                    "type": open_trade["type"],
                    "lot": partial_lot,
                    "pnl": partial_pnl,
                    "exit_reason": "TP1",
                    "session": "Asia" if ts.hour < 8 else "London" if ts.hour < 16 else "NY",
                    "duration_min": (ts - open_trade["entry_time"]).total_seconds() / 60,
                })
                open_trade["tp1_hit"] = True
                open_trade["lot"] = open_trade["lot"] * 0.5

            elif open_trade.get("tp1_hit"):
                exit_now, reason = should_exit(open_trade, row)
                if exit_now:
                    pnl = (price - open_trade["entry"] if open_trade["type"] == "buy"
                           else open_trade["entry"] - price) * open_trade["lot"] * 10
                    capital += pnl
                    trades.append({
                        "entry_time": open_trade["entry_time"],
                        "exit_time": ts,
                        "entry": open_trade["entry"],
                        "exit": price,
                        "type": open_trade["type"],
                        "lot": open_trade["lot"],
                        "pnl": pnl,
                        "exit_reason": reason or "TP2",
                        "session": "Asia" if ts.hour < 8 else "London" if ts.hour < 16 else "NY",
                        "duration_min": (ts - open_trade["entry_time"]).total_seconds() / 60,
                    })
                    open_trade = None

            else:
                exit_now, reason = should_exit(open_trade, row)
                if exit_now:
                    pnl = (price - open_trade["entry"] if open_trade["type"] == "buy"
                           else open_trade["entry"] - price) * open_trade["lot"] * 10
                    capital += pnl
                    trades.append({
                        "entry_time": open_trade["entry_time"],
                        "exit_time": ts,
                        "entry": open_trade["entry"],
                        "exit": price,
                        "type": open_trade["type"],
                        "lot": open_trade["lot"],
                        "pnl": pnl,
                        "exit_reason": reason or "TP",
                        "session": "Asia" if ts.hour < 8 else "London" if ts.hour < 16 else "NY",
                        "duration_min": (ts - open_trade["entry_time"]).total_seconds() / 60,
                    })
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
