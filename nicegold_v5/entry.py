import pandas as pd
import numpy as np


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["entry_signal"] = None

    df["ema_fast"] = df["close"].ewm(span=15, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=50, adjust=False).mean()
    df["rsi"] = rsi(df["close"], 14)
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()

    up = df["high"].diff().clip(lower=0)
    down = -df["low"].diff().clip(upper=0)
    atr = df["atr"]
    plus_di = 100 * up.rolling(14).sum() / atr
    minus_di = 100 * down.rolling(14).sum() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    df["adx"] = dx.rolling(14).mean()

    trend_up = (df["ema_fast"] > df["ema_slow"]) & (df["adx"] > 20) & (df["rsi"] > 55)
    trend_dn = (df["ema_fast"] < df["ema_slow"]) & (df["adx"] > 20) & (df["rsi"] < 45)

    df.loc[trend_up, "entry_signal"] = "buy"
    df.loc[trend_dn, "entry_signal"] = "sell"
    return df


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """คำนวณ RSI แบบเวกเตอร์เพื่อลดเวลาประมวลผล"""
    delta = series.diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))
