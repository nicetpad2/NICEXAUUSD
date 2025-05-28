import pandas as pd


def simulate_tp_exit(df_trades: pd.DataFrame, df_m1: pd.DataFrame, window_minutes: int = 60) -> pd.DataFrame:
    """ตรวจ high/low หลัง entry_time เพื่อกำหนดว่า TP1/TP2/SL ถูกชนก่อน
    ต้องมี column: timestamp, entry_price, tp1_price, tp2_price, sl_price, direction
    """
    df_trades = df_trades.copy()
    df_m1 = df_m1.copy()
    df_m1["timestamp"] = pd.to_datetime(df_m1["timestamp"])

    exit_prices: list[float] = []
    exit_reasons: list[str] = []

    for _, row in df_trades.iterrows():
        t0 = pd.to_datetime(row["timestamp"])
        direction = row.get("direction")
        if direction is None:
            direction = "sell" if row["entry_price"] > row["tp1_price"] else "buy"
        t1 = t0 + pd.Timedelta(minutes=window_minutes)

        window = df_m1[(df_m1["timestamp"] >= t0) & (df_m1["timestamp"] <= t1)]
        hit = None

        if direction == "buy":
            for _, bar in window.iterrows():
                if bar["low"] <= row["sl_price"]:
                    hit = ("SL", row["sl_price"])
                    break
                elif bar["high"] >= row["tp2_price"]:
                    hit = ("TP2", row["tp2_price"])
                    break
                elif bar["high"] >= row["tp1_price"]:
                    hit = ("TP1", row["tp1_price"])
                    break
        else:
            for _, bar in window.iterrows():
                if bar["high"] >= row["sl_price"]:
                    hit = ("SL", row["sl_price"])
                    break
                elif bar["low"] <= row["tp2_price"]:
                    hit = ("TP2", row["tp2_price"])
                    break
                elif bar["low"] <= row["tp1_price"]:
                    hit = ("TP1", row["tp1_price"])
                    break

        if hit:
            exit_reasons.append(hit[0])
            exit_prices.append(hit[1])
        else:
            exit_reasons.append("TIMEOUT")
            exit_prices.append(row["entry_price"])

    df_trades["exit_reason"] = exit_reasons
    df_trades["exit_price"] = exit_prices
    return df_trades
