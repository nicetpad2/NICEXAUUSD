def should_exit(trade, row):
    gain = (row["close"] - trade["entry"] if trade["type"] == "buy" else trade["entry"] - row["close"])
    if gain > 0 and row.get("gain_z", 0) < -0.5:
        return True, "gain_z drop"
    if gain > 0 and row.get("atr", 1.0) < row.get("atr_ma", 1.0):
        return True, "atr fading"
    return False, None
