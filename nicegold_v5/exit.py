import logging
from datetime import timedelta

TSL_TRIGGER_GAIN = 2.0   # Trigger TSL เมื่อ gain ≥ SL × 2.0
MIN_HOLD_MINUTES = 10    # ห้าม exit ถ้ายังถือไม่ถึง 10 นาที
MAX_HOLD_MINUTES = 360   # [Patch C.3] ถือสูงสุด 6 ชม.


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
        return False, None

    if holding_min > MAX_HOLD_MINUTES:
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
            logging.info("[Patch D.10] Exit: ATR fading + gain_z drop")
            return True, "atr_fade_gain_z_drop"

        if gain_z < -0.3:
            logging.info("[Patch D.10] Exit: gain_z reversal after profit")
            return True, "gain_z_reverse"

    return False, None
