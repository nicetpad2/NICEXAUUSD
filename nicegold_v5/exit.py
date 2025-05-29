import logging
from datetime import timedelta
import pandas as pd
import numpy as np

# [Patch v12.2.x] Auto session detection by timestamp (no config required)
def detect_session_auto(timestamp):
    hour = pd.to_datetime(timestamp).hour
    if 3 <= hour <= 7:
        return "Asia"
    elif 8 <= hour <= 14:
        return "London"
    elif 15 <= hour <= 23 or hour == 0:
        return "NY"
    return "Unknown"

TSL_TRIGGER_GAIN = 2.0  # [Patch v12.1.x] เริ่ม trail เร็วขึ้น
BE_TRIGGER_GAIN = 1.0   # [Patch v12.1.x] Trigger BE เร็วขึ้น
BE_HOLD_MIN = 10        # [Patch v12.1.x] ถือขั้นต่ำ 10 นาที
TSL_ATR_MULTIPLIER = 1.2  # [Patch v12.1.x] ใช้ ATR ในการ trail SL
MAX_HOLD_MINUTES = 360
MIN_PROFIT_TRIGGER = 0.5
MICRO_LOCK_THRESHOLD = 0.3


def _rget(row, key, default=None):
    if isinstance(row, dict):
        return row.get(key, default)
    if hasattr(row, key):
        return getattr(row, key)
    if hasattr(row, '__getitem__'):
        try:
            return row[key]
        except Exception:
            return default
    return default


def should_exit(trade, row):
    price_now = _rget(row, "close")
    entry = trade["entry"]
    direction = trade["type"]
    gain = price_now - entry if direction == "buy" else entry - price_now

    risk_mode = trade.get("risk_mode", "normal")
    recovery_prefix = "recovery_" if risk_mode == "recovery" else ""

    atr = _rget(row, "atr", 1.0)
    atr_ma = _rget(row, "atr_ma", 1.0)
    gain_z = _rget(row, "gain_z", 0)
    sl_threshold = atr * 1.2
    tsl_trigger = atr * TSL_TRIGGER_GAIN
    be_trigger = atr * BE_TRIGGER_GAIN

    # [Patch QA-P7] หมายเหตุ: ควรพิจารณาเพิ่ม Logic การคำนวณ TSL แบบ Dynamic โดยใช้ ATR ณ ปัจจุบัน
    # TSL_ATR_FACTOR = 1.5 # ตัวอย่างค่าคงที่สำหรับ Dynamic TSL
    now = _rget(row, "timestamp")
    entry_time = trade["entry_time"]
    holding_min = (now - entry_time).total_seconds() / 60

    if holding_min > MAX_HOLD_MINUTES:
        logging.info(f"[EXIT] Max hold exceeded ({holding_min:.1f} min)")
        return True, "timeout_exit"

    if gain < -sl_threshold and not trade.get("breakeven"):
        logging.info(f"[{recovery_prefix.upper()}SL] Hit @ {price_now:.2f}")
        return True, f"{recovery_prefix}sl"

    if gain >= tsl_trigger and not trade.get("tsl_activated"):
        trade["tsl_activated"] = True
        high_val = _rget(row, "high", price_now)
        low_val = _rget(row, "low", price_now)
        trade["trailing_sl"] = high_val - atr if direction == "buy" else low_val + atr
        logging.info(f"[{recovery_prefix.upper()}TSL] Activated: SL = {trade['trailing_sl']:.2f}")

    if trade.get("tsl_activated"):
        tsl = trade.get("trailing_sl")
        if (direction == "buy" and price_now <= tsl) or (direction == "sell" and price_now >= tsl):
            logging.info(f"[{recovery_prefix.upper()}TSL] Triggered @ {price_now:.2f}")
            return True, f"{recovery_prefix}tsl"
        high = _rget(row, "high", price_now) if direction == "buy" else _rget(row, "low", price_now)
        new_trail = high - atr if direction == "buy" else high + atr
        if (direction == "buy" and new_trail > tsl) or (direction == "sell" and new_trail < tsl):
            trade["trailing_sl"] = new_trail
            logging.debug(f"[TSL] Updated trail to {new_trail:.2f}")

    if gain >= be_trigger and holding_min >= BE_HOLD_MIN and not trade.get("breakeven"):
        trade["breakeven"] = True
        trade["breakeven_price"] = entry
        trade["break_even_time"] = now
        logging.info(f"[BE] Triggered: price = {entry:.2f}, hold = {holding_min:.1f} min")

    if trade.get("breakeven"):
        if (direction == "buy" and price_now <= entry) or (direction == "sell" and price_now >= entry):
            logging.info(f"[{recovery_prefix.upper()}BE_SL] Triggered @ {price_now:.2f}")
            return True, f"{recovery_prefix}be_sl"

        # [Patch F] ปิดเงื่อนไข gain_z_reverse เพื่อลดการใช้ข้อมูลอนาคต
        # if gain_z < -0.6:
        #     logging.info("[Patch D.14] Exit: gain_z reversal after profit")
        #     return True, "gain_z_reverse"

    # if gain > atr * 0.5 and gain_z < 0: # [Patch QA-P7] ยังคงปิดใช้งาน Early Profit Lock
    #     logging.info("[Patch D.14] Exit: early profit lock before gain_z turns negative")
    #     return True, "early_profit_lock"

    if gain > atr * MIN_PROFIT_TRIGGER and atr_ma < atr * 0.75:
        logging.info("[Patch D.14] Exit: volatility contraction after profit")
        return True, "atr_contract_exit"

    # [Patch F] ปิด micro_gain_lock เพื่อลดการออกแบบใช้ข้อมูลอนาคต
    # if gain > MICRO_LOCK_THRESHOLD and gain_z <= 0:
    #     logging.info("[Patch D.14] Exit: micro gain locked as momentum decayed")
    #     return True, "micro_gain_lock"

    return False, None

# [Patch v12.1.x] simulate_partial_tp_safe + BE/TSL Integration


def simulate_partial_tp_safe(df: pd.DataFrame, capital: float = 1000.0):
    trades = []
    logs = []
    open_position = None

    for i, row in df.iterrows():
        if pd.isna(row.get("entry_signal")):
            continue

        row_session = detect_session_auto(row["timestamp"])

        if open_position is None:
            entry_price = row["close"]
            atr = row.get("atr", 1.0)
            direction = "buy" if row["entry_signal"].startswith("buy") else "sell"

            sl = entry_price - atr * 1.2 if direction == "buy" else entry_price + atr * 1.2
            tp1 = entry_price + atr * 1.5 if direction == "buy" else entry_price - atr * 1.5
            tp2 = entry_price + atr * 3.0 if direction == "buy" else entry_price - atr * 3.0

            lot = 0.1
            open_position = {
                "entry_time": row["timestamp"],
                "entry": entry_price,
                "sl": sl,
                "tp1": tp1,
                "tp2": tp2,
                "lot": lot,
                "type": direction,
                "risk_mode": "normal",
                "tsl_activated": False,
                "breakeven": False,
                "tp1_hit": False,
                "session": row_session,
            }
            continue

        if open_position:
            # [Patch v12.2.x] Enhanced: wait TP2 if TP1 hit recently
            price_now = row["close"]
            direction = open_position["type"]
            tp1_hit = price_now >= open_position["tp1"] if direction == "buy" else price_now <= open_position["tp1"]
            if tp1_hit:
                open_position["tp1_hit"] = True

            if open_position["tp1_hit"] and not open_position["tsl_activated"]:
                if direction == "buy":
                    open_position["trailing_sl"] = max(row["high"] - row.get("atr", 1.0), open_position["sl"])
                else:
                    open_position["trailing_sl"] = min(row["low"] + row.get("atr", 1.0), open_position["sl"])
                open_position["tsl_activated"] = True

            exit_triggered, reason = should_exit(open_position, row)
            if exit_triggered:
                exit_price = row["close"]
                duration = (row["timestamp"] - open_position["entry_time"]).total_seconds() / 60
                mfe = np.abs(row["high"] - open_position["entry"] if open_position["type"] == "buy" else open_position["entry"] - row["low"])
                trades.append({
                    "entry_time": open_position["entry_time"],
                    "exit_time": row["timestamp"],
                    "entry": open_position["entry"],
                    "exit_price": exit_price,
                    "tp1_price": open_position["tp1"],
                    "tp2_price": open_position["tp2"],
                    "sl_price": open_position["sl"],
                    "lot": open_position["lot"],
                    "exit_reason": reason,
                    "session": open_position["session"],
                    "duration_min": duration,
                    "mfe": mfe,
                    "pnl": round((exit_price - open_position["entry"] if open_position["type"] == "buy" else open_position["entry"] - exit_price) * 10 * open_position["lot"], 2),
                })
                open_position = None

    return pd.DataFrame(trades), logs

