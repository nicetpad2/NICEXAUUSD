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


def generate_signals(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    df = df.copy()
    df["entry_signal"] = None
    df["entry_blocked_reason"] = None

    # --- Indicators ---
    df["ema_fast"] = df["close"].ewm(span=15, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_slope"] = df["ema_fast"].diff()
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
    atr_ma_calc = df["atr"].rolling(50).mean()
    if "atr_ma" in df.columns:
        df["atr_ma"] = atr_ma_calc.fillna(df["atr_ma"])
    else:
        df["atr_ma"] = atr_ma_calc

    # --- Patch C.1: Fallback gain_z ---
    gainz = df["gain_z"] if "gain_z" in df.columns else pd.Series(1.0, index=df.index)

    # --- ใช้ config fold-based ---
    gainz_threshold = config.get("gain_z_thresh", -0.1) if config else -0.1
    ema_slope_threshold = config.get("ema_slope_min", 0.0) if config else 0.0

    gainz_guard = gainz < gainz_threshold
    ema_flat = df["ema_slope"] <= ema_slope_threshold
    if "timestamp" in df.columns:
        df["entry_time"] = df["timestamp"]
        time_gap_ok = df["entry_time"].diff().dt.total_seconds().fillna(999999) > 15 * 60
    else:
        time_gap_ok = pd.Series(True, index=df.index)

    recovery_block = gainz_guard | ema_flat | ~time_gap_ok
    df.loc[recovery_block, "entry_blocked_reason"] = "Recovery Filter Blocked"

    # --- Momentum Rebalance Filter ---
    gainz_thresh = config.get("gain_z_thresh", -0.1) if config else -0.1
    ema_slope_min = config.get("ema_slope_min", 0.0) if config else 0.0
    volatility_ratio = config.get("volatility_thresh", 0.8) if config else 0.8

    session_ok = (
        df["timestamp"].dt.hour.between(13, 20)
        if "timestamp" in df.columns
        else pd.Series(True, index=df.index)
    )
    trend_up = df["ema_fast"] > df["ema_slow"]
    trend_dn = df["ema_fast"] < df["ema_slow"]
    envelope_up = df["close"] > df["ema_slow"] + 0.3
    envelope_dn = df["close"] < df["ema_slow"] - 0.3
    volatility = df["atr"] > df["atr_ma"] * volatility_ratio
    momentum = gainz > gainz_thresh

    buy_cond = trend_up & envelope_up & volatility & momentum & session_ok
    sell_cond = trend_dn & envelope_dn & volatility & momentum & session_ok

    # --- Final Entry Assignment ---
    df.loc[buy_cond & df["entry_blocked_reason"].isnull(), "entry_signal"] = "buy"
    df.loc[sell_cond & df["entry_blocked_reason"].isnull(), "entry_signal"] = "sell"

    # --- Logging QA Summary ---
    blocked_pct = df["entry_signal"].isnull().mean() * 100
    print(f"[Patch D.2] Entry Signal Blocked: {blocked_pct:.2f}%")

    return df
