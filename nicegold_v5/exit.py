import logging


def should_exit(trade, row):
    price_now = row["close"]
    entry = trade["entry"]
    direction = trade["type"]
    gain = price_now - entry if direction == "buy" else entry - price_now

    atr = row.get("atr", 1.0)
    sl_threshold = atr * 1.5
    be_trigger = sl_threshold * 1.2

    if gain < -sl_threshold and not trade.get("breakeven"):
        logging.info(f"[Patch] SL Hit: {direction.upper()} @ {price_now:.2f} (Loss: {gain:.2f})")
        return True, "SL"

    if gain >= be_trigger and not trade.get("breakeven"):
        trade["breakeven"] = True
        trade["breakeven_price"] = entry
        trade["break_even_time"] = row["timestamp"]
        logging.info(
            f"[Patch] BE Move Triggered at {row['timestamp']} | SL moved to {entry:.2f}"
        )

    if trade.get("breakeven") and (
        (direction == "buy" and price_now <= entry) or
        (direction == "sell" and price_now >= entry)
    ):
        logging.info(f"[Patch] BE Stop-Loss Hit @ {price_now:.2f}")
        return True, "BE_SL"

    if gain > 0 and row.get("gain_z", 0) < -0.5:
        return True, "gain_z drop"
    if gain > 0 and row.get("atr", 1.0) < row.get("atr_ma", 1.0):
        return True, "atr fading"

    return False, None
