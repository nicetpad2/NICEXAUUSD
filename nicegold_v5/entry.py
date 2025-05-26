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
    df["entry_blocked_reason"] = None

    # --- Core Indicator ---
    df["ema_fast"] = df["close"].ewm(span=15, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=50, adjust=False).mean()
    df["rsi"] = rsi(df["close"], 14)
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
    df["atr_ma"] = df["atr"].rolling(50).mean()
    df["ema_slope"] = df["ema_fast"].diff()

    # --- Patch C.1: Fallback gain_z ---
    gainz = df["gain_z"] if "gain_z" in df.columns else pd.Series(1.0, index=df.index)

    # --- Filter Logic (Relaxed) ---
    gainz_guard = gainz < -0.1
    ema_flat = df["ema_slope"] <= 0
    if "timestamp" in df.columns:
        df["entry_time"] = df["timestamp"]
        time_gap_ok = df["entry_time"].diff().dt.total_seconds().fillna(999999) > 15 * 60
    else:
        time_gap_ok = pd.Series(True, index=df.index)

    recovery_block = gainz_guard | ema_flat | ~time_gap_ok
    df.loc[recovery_block, "entry_blocked_reason"] = "Recovery Filter Blocked"

    # --- Trend Logic ---
    trend_up = (df["ema_fast"] > df["ema_slow"]) & (df["rsi"] > 55)
    trend_dn = (df["ema_fast"] < df["ema_slow"]) & (df["rsi"] < 45)

    # --- Final Entry Assignment ---
    df.loc[trend_up & df["entry_blocked_reason"].isnull(), "entry_signal"] = "buy"
    df.loc[trend_dn & df["entry_blocked_reason"].isnull(), "entry_signal"] = "sell"

    # --- Logging QA Summary ---
    blocked_pct = df["entry_signal"].isnull().mean() * 100
    print(f"[Patch C.1] Entry Signal Blocked: {blocked_pct:.2f}%")

    return df
