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
    """สร้างสัญญาณตามกลยุทธ์ VBTB (Volatility Breakout + Trend Bias)"""
    df = df.copy()
    df["entry_signal"] = None
    df["entry_blocked_reason"] = None

    # [Patch VBTB] คำนวณอินดิเคเตอร์พื้นฐาน
    df["ema_15"] = df["close"].ewm(span=15).mean()
    df["ema_50"] = df["close"].ewm(span=50).mean()
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
    df["atr_ma"] = df["atr"].rolling(50).mean()
    gain = df["close"].diff()
    df["gain_z"] = (gain - gain.rolling(20).mean()) / (gain.rolling(20).std() + 1e-9)

    # คอลัมน์เดิมสำหรับโมดูลอื่นยังคงใช้งานได้
    df["ema_fast"] = df["ema_15"]
    df["ema_slow"] = df["ema_50"]
    df["ema_slope"] = df["ema_fast"].diff()

    df["entry_time"] = df["timestamp"]

    # --- เงื่อนไข VBTB ---
    trend_up = df["ema_15"] > df["ema_50"]
    trend_dn = df["ema_15"] < df["ema_50"]

    breakout_up = df["close"] > df["ema_50"] + 0.3
    breakout_dn = df["close"] < df["ema_50"] - 0.3

    volatility_ok = df["atr"] > df["atr_ma"] * 0.8
    momentum_ok = df["gain_z"] > -0.1
    session_ok = df["timestamp"].dt.hour.between(13, 20)

    buy_cond = trend_up & breakout_up & volatility_ok & momentum_ok & session_ok
    sell_cond = trend_dn & breakout_dn & volatility_ok & momentum_ok & session_ok
    df.loc[buy_cond, "entry_signal"] = "buy"
    df.loc[sell_cond, "entry_signal"] = "sell"

    # [Patch VBTB] บันทึกเหตุผลที่บล็อกสัญญาณอย่างย่อ
    reasons = (
        np.where(~(trend_up | trend_dn), "no_trend|", "")
        + np.where(~volatility_ok, "low_vol|", "")
        + np.where(~momentum_ok, "gain_z_low|", "")
        + np.where(~session_ok, "off_session|", "")
    )
    df.loc[df["entry_signal"].isnull(), "entry_blocked_reason"] = (
        pd.Series(reasons).str.strip("|")
    )

    blocked_pct = df["entry_signal"].isnull().mean() * 100
    print(f"[Patch VBTB] Entry Signal Blocked: {blocked_pct:.2f}%")

    return df


def generate_signals_qa_clean(df: pd.DataFrame) -> pd.DataFrame:
    """สร้างสัญญาณแบบย่อสำหรับชุดข้อมูล QA"""
    df = df.copy()

    if "timestamp" not in df.columns and {"Date", "Timestamp"}.issubset(df.columns):
        df["year"] = df["Date"].astype(str).str[:4].astype(int) - 543
        df["month"] = df["Date"].astype(str).str[4:6]
        df["day"] = df["Date"].astype(str).str[6:8]
        df["datetime_str"] = (
            df["year"].astype(str)
            + "-"
            + df["month"]
            + "-"
            + df["day"]
            + " "
            + df["Timestamp"]
        )
        df["timestamp"] = pd.to_datetime(df["datetime_str"])

    df["ema_fast"] = df["close"].ewm(span=15).mean()
    df["ema_slow"] = df["close"].ewm(span=50).mean()
    df["ema_slope"] = df["ema_fast"].diff()
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
    df["atr_ma"] = df["atr"].rolling(50).mean()
    gain = df["close"].diff()
    df["gain_z"] = (gain - gain.rolling(20).mean()) / (gain.rolling(20).std() + 1e-9)

    df["entry_signal"] = None
    trend = df["ema_fast"] > df["ema_slow"]
    envelope = df["close"] > df["ema_slow"] + 0.1
    volatility = df["atr"] > df["atr_ma"] * 0.85
    momentum = df["gain_z"] > -0.15
    time_ok = df["timestamp"].dt.hour.between(9, 21)

    df.loc[trend & envelope & volatility & momentum & time_ok, "entry_signal"] = "buy"
    return df
