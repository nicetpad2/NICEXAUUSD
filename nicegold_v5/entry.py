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
    """สร้างคอลัมน์ entry_signal และเหตุผลที่บล็อกแบบละเอียด"""
    df = df.copy()
    df["entry_signal"] = None
    df["entry_blocked_reason"] = None

    df["ema_fast"] = df["close"].ewm(span=15).mean()
    df["ema_slow"] = df["close"].ewm(span=50).mean()
    df["ema_slope"] = df["ema_fast"].diff()
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
    df["atr_ma"] = df["atr"].rolling(50).mean()
    gain = df["close"].diff()
    df["gain_z"] = (gain - gain.rolling(20).mean()) / (gain.rolling(20).std() + 1e-9)

    gainz = df["gain_z"]
    gainz_thresh = config.get("gain_z_thresh", -0.2) if config else -0.2
    ema_slope_min = config.get("ema_slope_min", 0.0) if config else 0.0
    volatility_ratio = config.get("volatility_thresh", 0.8) if config else 0.8

    df["entry_time"] = df["timestamp"]
    time_filter = df["timestamp"].dt.hour.between(6, 22)
    trend_up = df["ema_fast"] > df["ema_slow"]
    trend_dn = df["ema_fast"] < df["ema_slow"]
    envelope_up = df["close"] > df["ema_slow"] + 0.2
    envelope_dn = df["close"] < df["ema_slow"] - 0.2
    volatility = df["atr"] > df["atr_ma"] * volatility_ratio
    momentum = gainz > gainz_thresh

    buy_cond = trend_up & envelope_up & volatility & momentum & time_filter
    sell_cond = trend_dn & envelope_dn & volatility & momentum & time_filter

    df.loc[buy_cond, "entry_signal"] = "buy"
    df.loc[sell_cond, "entry_signal"] = "sell"

    if df["entry_signal"].notnull().sum() < len(df) * 0.05:
        df["entry_signal"] = None
        gainz_thresh = -0.3
        ema_slope_min = -0.01
        momentum = gainz > gainz_thresh
        buy_cond = trend_up & envelope_up & volatility & momentum & time_filter
        sell_cond = trend_dn & envelope_dn & volatility & momentum & time_filter
        df.loc[buy_cond, "entry_signal"] = "buy"
        df.loc[sell_cond, "entry_signal"] = "sell"
        df["entry_blocked_reason"] = df["entry_blocked_reason"].fillna("fallback_config")

    no_trend = ~(trend_up | trend_dn)
    low_volatility = ~volatility
    no_momentum = ~momentum
    out_of_session = ~time_filter

    blocked_reason = []
    for t, v, m, s in zip(no_trend, low_volatility, no_momentum, out_of_session):
        reasons = []
        if t:
            reasons.append("trend")
        if v:
            reasons.append("volatility")
        if m:
            reasons.append("momentum")
        if s:
            reasons.append("session")
        blocked_reason.append(",".join(reasons) if reasons else None)

    df.loc[df["entry_signal"].isnull(), "entry_blocked_reason"] = blocked_reason
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
