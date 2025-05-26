import logging
from datetime import timedelta

TSL_TRIGGER_GAIN = 1.5   # Trigger TSL เมื่อ gain ≥ SL × 1.5
MIN_HOLD_MINUTES = 10    # ห้าม exit ถ้ายังถือไม่ถึง 10 นาที


def should_exit(trade, row):
    price_now = row["close"]
    entry = trade["entry"]
    direction = trade["type"]
    gain = price_now - entry if direction == "buy" else entry - price_now

    atr = row.get("atr", 1.0)
    sl_threshold = atr * 1.2
    be_trigger = sl_threshold * 1.2

    now = row["timestamp"]
    entry_time = trade["entry_time"]
    holding_min = (now - entry_time).total_seconds() / 60

    # [Patch A.2] ห้าม exit ก่อนถึงระยะถือขั้นต่ำ
    if holding_min < MIN_HOLD_MINUTES:
        return False, None

    # ✅ SL Hit (ก่อน TSL)
    if gain < -sl_threshold and not trade.get("breakeven"):
        logging.info(f"[Patch] SL Hit: {direction.upper()} @ {price_now:.2f} (Loss: {gain:.2f})")
        return True, "SL"

    # ✅ TSL Triggered
    if gain >= TSL_TRIGGER_GAIN * atr and not trade.get("tsl_activated"):
        trade["tsl_activated"] = True
        trade["trailing_sl"] = entry  # ตัดขาดทุนจะถูกเลื่อนขึ้น
        logging.info(f"[Patch] TSL Activated @ {now} | SL moved to BE")

    # ✅ ถ้ามี BE หรือ TSL แล้ว SL โดน
    if trade.get("breakeven") or trade.get("tsl_activated"):
        trailing_sl = trade.get("trailing_sl", entry)
        if (direction == "buy" and price_now <= trailing_sl) or (direction == "sell" and price_now >= trailing_sl):
            logging.info(f"[Patch] TSL Stop Hit @ {price_now:.2f}")
            return True, "TSL" if trade.get("tsl_activated") else "BE_SL"

    # ✅ BE Trigger (ยังไม่ได้ชน)
    if gain >= be_trigger and not trade.get("breakeven"):
        trade["breakeven"] = True
        trade["breakeven_price"] = entry
        trade["break_even_time"] = row["timestamp"]
        logging.info(f"[Patch] BE Move Triggered at {now} | SL moved to {entry:.2f}")

    # ✅ Exit Guard (เช่น gain_z < -0.5 หลัง gain บวก)
    if gain > 0 and row.get("gain_z", 0) < -0.5:
        return True, "gain_z drop"
    if gain > 0 and row.get("atr", 1.0) < row.get("atr_ma", 1.0):
        return True, "atr fading"

    return False, None
