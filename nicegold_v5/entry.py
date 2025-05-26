import pandas as pd
import numpy as np


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """คำนวณ RSI แบบเวกเตอร์เพื่อลดเวลาประมวลผล"""
    delta = series.diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["entry_signal"] = None
    df["entry_blocked_reason"] = None  # [Patch A] QA column

    # === Core Indicator Calculation ===
    df["ema_fast"] = df["close"].ewm(span=15, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=50, adjust=False).mean()
    df["rsi"] = rsi(df["close"], 14)
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
    df["atr_ma"] = df["atr"].rolling(50).mean()
    df["ema_slope"] = df["ema_fast"].diff()

    # === Trend Conditions ===
    trend_up = (df["ema_fast"] > df["ema_slow"]) & (df["rsi"] > 55)
    trend_dn = (df["ema_fast"] < df["ema_slow"]) & (df["rsi"] < 45)

    # === [Patch A] SL Filtering Guard ===
    # เงื่อนไขการบล็อกสัญญาณ
    block_gainz = df.get("gain_z", pd.Series(0, index=df.index)) < -0.3
    block_atr_spike = df["atr"] / (df["atr_ma"] + 1e-9) > 2.5
    block_ema_down = df["ema_slope"] < 0

    # สร้างคำอธิบายเหตุผลบล็อก
    df.loc[block_gainz, "entry_blocked_reason"] = "gain_z < -0.3"
    df.loc[block_atr_spike, "entry_blocked_reason"] = "ATR spike"
    df.loc[block_ema_down & trend_up, "entry_blocked_reason"] = "EMA slope < 0"

    # บล็อก entry จริงๆ
    allow_buy = trend_up & df["entry_blocked_reason"].isnull()
    allow_sell = trend_dn & df["entry_blocked_reason"].isnull()

    df.loc[allow_buy, "entry_signal"] = "buy"
    df.loc[allow_sell, "entry_signal"] = "sell"

    return df
