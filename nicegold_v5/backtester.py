import pandas as pd
from datetime import datetime
import random
import logging
from nicegold_v5.risk import (
    calc_lot_risk,
    kill_switch,
    adaptive_tp_multiplier,
    get_sl_tp,
    calc_lot_recovery,
    get_sl_tp_recovery,
    RECOVERY_SL_TRIGGER,
)
from nicegold_v5.exit import should_exit
from tqdm import tqdm
import time


def run_backtest(df: pd.DataFrame):
    """Backtest พร้อม Recovery Mode และ Logging เต็มรูปแบบ"""
    capital = 100.0
    trades = []
    equity = []
    open_trade = None
    sl_streak = 0
    equity_curve: list[float] = []
    recovery_mode = False  # เริ่มแบบปกติ
    start = time.time()

    for i, row in tqdm(df.iterrows(), total=len(df), desc="⏱️ Running Backtest", unit="rows"):
        ts = pd.to_datetime(row["timestamp"])
        price = row["close"]
        equity.append({"timestamp": ts, "equity": capital})
        equity_curve.append(capital)

        if kill_switch(equity_curve):
            break

        # ตรวจว่าเข้าสู่ Recovery Mode หรือไม่
        if sl_streak >= RECOVERY_SL_TRIGGER and not recovery_mode:
            recovery_mode = True
            logging.info(
                f"[Patch B] Recovery Mode Triggered at {ts} after SL streak = {sl_streak}"
            )

        if open_trade:
            direction = open_trade["type"]
            session = open_trade["session"]
            atr_entry = open_trade["atr"]
            sl_dist = atr_entry * 1.2
            tp1 = sl_dist * 1.5
            tp2 = sl_dist * adaptive_tp_multiplier(session)
            gain = price - open_trade["entry"] if direction == "buy" else open_trade["entry"] - price

            if not open_trade.get("tp1_hit") and gain >= tp1:
                partial_lot = open_trade["lot"] * 0.5
                partial_pnl = tp1 * open_trade["lot"] * 10 * 0.5
                commission = partial_lot * 2
                spread_cost = partial_lot * 0.2
                slippage_cost = abs(random.uniform(-0.3, 0.3)) * partial_lot
                capital += partial_pnl - commission - spread_cost - slippage_cost
                trades.append({
                    "entry_time": open_trade["entry_time"],
                    "exit_time": ts,
                    "entry": open_trade["entry"],
                    "exit": price,
                    "type": open_trade["type"],
                    "lot": partial_lot,
                    "pnl": partial_pnl - commission - spread_cost - slippage_cost,
                    "commission": commission,
                    "spread_cost": spread_cost,
                    "slippage_cost": slippage_cost,
                    "exit_reason": "TP1",
                    "session": session,
                    "risk_mode": "recovery" if recovery_mode else "normal",
                    "duration_min": (ts - open_trade["entry_time"]).total_seconds() / 60,
                })
                open_trade["tp1_hit"] = True
                open_trade["lot"] = open_trade["lot"] * 0.5

            elif open_trade.get("tp1_hit"):
                exit_now, reason = should_exit(open_trade, row)
                if exit_now or (direction == "buy" and gain >= tp2) or (direction == "sell" and gain >= tp2):
                    pnl = (price - open_trade["entry"] if direction == "buy" else open_trade["entry"] - price) * open_trade["lot"] * 10
                    commission = open_trade["lot"] * 2
                    spread_cost = open_trade["lot"] * 0.2
                    slippage_cost = abs(random.uniform(-0.3, 0.3)) * open_trade["lot"]
                    capital += pnl - commission - spread_cost - slippage_cost
                    trades.append({
                        "entry_time": open_trade["entry_time"],
                        "exit_time": ts,
                        "entry": open_trade["entry"],
                        "exit": price,
                        "type": direction,
                        "lot": open_trade["lot"],
                        "pnl": pnl - commission - spread_cost - slippage_cost,
                        "commission": commission,
                        "spread_cost": spread_cost,
                        "slippage_cost": slippage_cost,
                        "exit_reason": reason or "TP2",
                        "session": session,
                        "risk_mode": "recovery" if recovery_mode else "normal",
                        "duration_min": (ts - open_trade["entry_time"]).total_seconds() / 60,
                    })
                    if (reason or "").startswith("SL"):
                        sl_streak += 1
                        recovery_mode = sl_streak >= RECOVERY_SL_TRIGGER
                    else:
                        sl_streak = 0
                        recovery_mode = False
                    open_trade = None

            else:
                exit_now, reason = should_exit(open_trade, row)
                if exit_now:
                    pnl = (price - open_trade["entry"] if direction == "buy" else open_trade["entry"] - price) * open_trade["lot"] * 10
                    commission = open_trade["lot"] * 2
                    spread_cost = open_trade["lot"] * 0.2
                    slippage_cost = abs(random.uniform(-0.3, 0.3)) * open_trade["lot"]
                    capital += pnl - commission - spread_cost - slippage_cost
                    trades.append({
                        "entry_time": open_trade["entry_time"],
                        "exit_time": ts,
                        "entry": open_trade["entry"],
                        "exit": price,
                        "type": direction,
                        "lot": open_trade["lot"],
                        "pnl": pnl - commission - spread_cost - slippage_cost,
                        "commission": commission,
                        "spread_cost": spread_cost,
                        "slippage_cost": slippage_cost,
                        "exit_reason": reason or "TP",
                        "session": session,
                        "risk_mode": "recovery" if recovery_mode else "normal",
                        "duration_min": (ts - open_trade["entry_time"]).total_seconds() / 60,
                    })
                    if (reason or "").startswith("SL"):
                        sl_streak += 1
                        recovery_mode = sl_streak >= RECOVERY_SL_TRIGGER
                    else:
                        sl_streak = 0
                        recovery_mode = False
                    open_trade = None

        if not open_trade and row.get("entry_signal") in ["buy", "sell"]:
            session = "Asia" if ts.hour < 8 else "London" if ts.hour < 16 else "NY"
            if recovery_mode:
                lot = calc_lot_recovery(capital, row.get("atr", 1.0), 1.5)
            else:
                lot = calc_lot_risk(capital, row.get("atr", 1.0), 1.5)
            open_trade = {
                "entry": price,
                "entry_time": ts,
                "type": row.get("entry_signal"),
                "lot": lot,
                "session": session,
                "atr": row.get("atr", 1.0),
                "risk_mode": "recovery" if recovery_mode else "normal",
            }

    end = time.time()
    print(
        f"⏱️ Backtest Duration: {end - start:.2f}s | Trades: {len(trades)} | Avg per row: {(end - start)/len(df):.6f}s"
    )
    return pd.DataFrame(trades), pd.DataFrame(equity)
