import pandas as pd
from datetime import datetime
import random
import logging
import numpy as np
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

# [Patch C.2] Enable full RAM mode
MAX_RAM_MODE = True


def run_backtest(df: pd.DataFrame):
    """Backtest พร้อม Recovery Mode และ Logging เต็มรูปแบบ"""
    logging.info(f"[TIME] run_backtest() start: {time.strftime('%H:%M:%S')}")
    capital = 100.0
    trades = []
    equity = []
    open_trade = None
    sl_streak = 0
    equity_curve: list[float] = []
    recovery_mode = False  # เริ่มแบบปกติ
    start = time.time()

    if MAX_RAM_MODE:
        df = df.astype({col: np.float32 for col in df.select_dtypes(include="number").columns})

    it = df.itertuples(index=False)
    bar_count = len(df)
    for i, row in enumerate(
        tqdm(it, total=bar_count, desc="⏱️ Running Backtest", unit="rows")
    ):
        ts = pd.to_datetime(row.timestamp)
        price = row.close
        atr_val = row.atr if hasattr(row, "atr") else 1.0
        gain_z = getattr(row, "gain_z", 0)
        # [Perf-A] เพิ่ม caching ATR เพื่อไม่เรียกซ้ำ
        if i == 0:
            last_atr = atr_val
        else:
            last_atr = 0.9 * last_atr + 0.1 * atr_val

        equity.append({"timestamp": ts, "equity": capital})
        equity_curve.append(capital)

        # 🚀 Boost: ตรวจ kill_switch ทุก 100 แถวเท่านั้น
        if i % 100 == 0 and kill_switch(equity_curve):
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

        if not open_trade and getattr(row, "entry_signal", None) in ["buy", "sell"]:
            session = "Asia" if ts.hour < 8 else "London" if ts.hour < 16 else "NY"
            if recovery_mode:
                lot = calc_lot_recovery(capital, atr_val, 1.5)
            else:
                lot = calc_lot_risk(capital, atr_val, 1.5)
            open_trade = {
                "entry": price,
                "entry_time": ts,
                "type": getattr(row, "entry_signal", None),
                "lot": lot,
                "session": session,
                "atr": atr_val,
                "risk_mode": "recovery" if recovery_mode else "normal",
            }

    end = time.time()
    logging.info(f"[TIME] run_backtest() done in {end - start:.2f}s")
    print(
        f"⏱️ Backtest Duration: {end - start:.2f}s | Trades: {len(trades)} | Avg per row: {(end - start)/len(df):.6f}s"
    )
    return pd.DataFrame(trades), pd.DataFrame(equity)
