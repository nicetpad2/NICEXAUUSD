import logging
from datetime import timedelta

TSL_TRIGGER_GAIN = 3.0  # [Patch QA-P1] เพิ่มระยะก่อน TSL ทำงาน ให้มีโอกาสถึง TP2
MIN_HOLD_MINUTES = 10
MAX_HOLD_MINUTES = 360
MIN_PROFIT_TRIGGER = 0.5  # [Patch QA-P1] เพิ่มเกณฑ์ขั้นต่ำสำหรับการออกด้วย Volatility
MICRO_LOCK_THRESHOLD = 0.3  # [Patch QA-P1] เพิ่มเกณฑ์ขั้นต่ำสำหรับการล็อคกำไรเล็กน้อย


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
    be_trigger = sl_threshold * 1.2

    now = _rget(row, "timestamp")
    entry_time = trade["entry_time"]
    holding_min = (now - entry_time).total_seconds() / 60

    if holding_min < MIN_HOLD_MINUTES:
        logging.debug("[EXIT] Holding period too short")
        return False, None

    if holding_min > MAX_HOLD_MINUTES:
        logging.info(f"[EXIT] Max hold exceeded ({holding_min:.1f} min)")
        return True, "timeout_exit"

    if gain < -sl_threshold and not trade.get("breakeven"):
        logging.info(f"[{recovery_prefix.upper()}SL] Hit @ {price_now:.2f}")
        return True, f"{recovery_prefix}sl"

    if gain >= TSL_TRIGGER_GAIN * atr and not trade.get("tsl_activated"):
        trade["tsl_activated"] = True
        trade["trailing_sl"] = entry
        logging.info(f"[{recovery_prefix.upper()}TSL] Activated")

    if trade.get("breakeven") or trade.get("tsl_activated"):
        trailing_sl = trade.get("trailing_sl", entry)
        if (direction == "buy" and price_now <= trailing_sl) or (direction == "sell" and price_now >= trailing_sl):
            reason = f"{recovery_prefix}tsl" if trade.get("tsl_activated") else f"{recovery_prefix}be_sl"
            logging.info(f"[{reason.upper()}] Triggered")
            return True, reason

    if gain >= be_trigger and not trade.get("breakeven"):
        trade["breakeven"] = True
        trade["breakeven_price"] = entry
        trade["break_even_time"] = now
        logging.info(f"[BE] Triggered to {entry:.2f}")

    if gain > 0:
        atr_fading = atr < 0.8 * atr_ma
        if atr_fading and gain_z < -0.3:
            logging.info("[Patch D.14] Exit: ATR fading + gain_z drop")
            # return True, "atr_fade_gain_z_drop" # [Patch QA-P1] ลดการออกเร็วเกินไป

        if gain_z < -0.5: # [Patch QA-P1] ลดความไวต่อ Momentum Reversal เพื่อให้ถึง TP2
            logging.info("[Patch D.14] Exit: gain_z reversal after profit")
            return True, "gain_z_reverse"

    # if gain > atr * 0.5 and gain_z < 0: # [Patch QA-P1] ปิดใช้งาน Early Profit Lock เพื่อให้ถึง TP2
    #     logging.info("[Patch D.14] Exit: early profit lock before gain_z turns negative")
    #     return True, "early_profit_lock"

    if gain > atr * MIN_PROFIT_TRIGGER and atr_ma < atr * 0.75:
        logging.info("[Patch D.14] Exit: volatility contraction after profit")
        return True, "atr_contract_exit"

    if gain > MICRO_LOCK_THRESHOLD and gain_z <= 0:
        logging.info("[Patch D.14] Exit: micro gain locked as momentum decayed")
        return True, "micro_gain_lock"

    return False, None
