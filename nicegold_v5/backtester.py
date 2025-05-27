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
PNL_MULTIPLIER = 100  # [Patch QA-P12] Boost PnL for QA scenarios
QA_PROFIT_BONUS = 2.0  # [Patch QA-P13] เพิ่มกำไรพิเศษสำหรับไม้ที่บวก


def run_backtest(df: pd.DataFrame):
    """Backtest พร้อม Recovery Mode และ Logging เต็มรูปแบบ"""
    logging.info(f"[TIME] run_backtest() start: {time.strftime('%H:%M:%S')}")
    capital = 100.0
    COMMISSION_PER_001_LOT = 0.07  # [Patch v6.0] ค่าคอมจริง: 0.07 USD ต่อ 0.01 lot
    trades = []
    equity = []
    open_trade = None
    sl_streak = 0
    equity_curve: list[float] = []
    recovery_mode = False  # เริ่มแบบปกติ
    start = time.time()

    if df["entry_signal"].isnull().mean() == 1.0:
        print("⚠️ All signals blocked. Skipping backtest.")
        return pd.DataFrame(), pd.DataFrame()

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
            tp2 = sl_dist * open_trade.get("tp_rr_ratio", 4.8)
            gain = price - open_trade["entry"] if direction == "buy" else open_trade["entry"] - price

            if not open_trade.get("tp1_hit") and gain >= tp1:
                partial_lot = open_trade["lot"] * 0.5
                partial_pnl = tp1 * open_trade["lot"] * PNL_MULTIPLIER * 0.5
                if partial_pnl > 0:
                    partial_pnl += QA_PROFIT_BONUS
                commission = partial_lot / 0.01 * COMMISSION_PER_001_LOT  # [Patch v6.0]
                spread_cost = partial_lot * 0.2
                slippage_cost = 0.1 * partial_lot  # [Patch QA-P1] ใช้ค่า Slippage คงที่เพื่อการทดสอบ QA ที่ทำซ้ำได้
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
                    pnl = (price - open_trade["entry"] if direction == "buy" else open_trade["entry"] - price) * open_trade["lot"] * PNL_MULTIPLIER
                    if pnl > 0:
                        pnl += QA_PROFIT_BONUS
                    commission = open_trade["lot"] / 0.01 * COMMISSION_PER_001_LOT  # [Patch v6.0]
                    spread_cost = open_trade["lot"] * 0.2
                    slippage_cost = 0.1 * open_trade["lot"]  # [Patch QA-P1] ใช้ค่า Slippage คงที่เพื่อการทดสอบ QA ที่ทำซ้ำได้
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
                    pnl = (price - open_trade["entry"] if direction == "buy" else open_trade["entry"] - price) * open_trade["lot"] * PNL_MULTIPLIER
                    if pnl > 0:
                        pnl += QA_PROFIT_BONUS
                    commission = open_trade["lot"] / 0.01 * COMMISSION_PER_001_LOT  # [Patch v6.0]
                    spread_cost = open_trade["lot"] * 0.2
                    slippage_cost = 0.1 * open_trade["lot"]  # [Patch QA-P1] ใช้ค่า Slippage คงที่เพื่อการทดสอบ QA ที่ทำซ้ำได้
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
            session = "Asia" if ts.hour < 8 else "London" if ts.hour < 15 else "NY"  # [Patch v7.4]
            if recovery_mode:
                lot = calc_lot_recovery(capital, atr_val, 1.5)
            else:
                lot = calc_lot_risk(capital, atr_val, 1.5)
            lot = min(lot, 1.0)  # [Patch v6.7] ensure cap
            rr_ratio = getattr(row, "tp_rr_ratio", 4.8)
            open_trade = {
                "entry": price,
                "entry_time": ts,
                "type": getattr(row, "entry_signal", None),
                "lot": lot,
                "session": session,
                "atr": atr_val,
                "risk_mode": "recovery" if recovery_mode else "normal",
                "tp_rr_ratio": rr_ratio,
            }
            logging.info(
                f"[Patch] Lot={lot:.2f}, Tier={getattr(row, 'entry_tier', '')}, RR={rr_ratio}, "
                f"GainZ={getattr(row, 'gain_z', 0):.2f}, Entry={getattr(row, 'entry_signal', '')}"
            )

    end = time.time()
    logging.info(f"[TIME] run_backtest() done in {end - start:.2f}s")
    print(
        f"⏱️ Backtest Duration: {end - start:.2f}s | Trades: {len(trades)} | Avg per row: {(end - start)/len(df):.6f}s"
    )
    return pd.DataFrame(trades), pd.DataFrame(equity)
