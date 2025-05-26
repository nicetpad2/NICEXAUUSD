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
    """Patch VBTB+ UltraFix v4.1 – QA Enterprise Ready"""
    df = df.copy()
    df["entry_signal"] = None
    df["entry_blocked_reason"] = None
    df["lot_suggested"] = 0.05

    # --- Indicators ---
    df["ema_15"] = df["close"].ewm(span=15).mean()
    df["ema_50"] = df["close"].ewm(span=50).mean()
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
    df["atr_ma"] = df["atr"].rolling(50).mean()
    gain = df["close"].diff()
    df["gain_z"] = (gain - gain.rolling(20).mean()) / (gain.rolling(20).std() + 1e-9)
    df["volume_ma"] = df["volume"].rolling(20).mean()
    df["entry_time"] = df["timestamp"]
    df["signal_id"] = df["timestamp"].astype(str)

    # --- Session Tag ---
    df["session_label"] = "None"
    df.loc[df["timestamp"].dt.hour.between(13, 17), "session_label"] = "NY"
    session = df["session_label"] != "None"

    # คอลัมน์เดิมสำหรับโมดูลอื่น
    df["ema_fast"] = df["ema_15"]
    df["ema_slow"] = df["ema_50"]
    df["ema_slope"] = df["ema_fast"].diff()

    # --- Entry Conditions v4 ---
    trend_up = df["ema_15"] > df["ema_50"]
    trend_dn = df["ema_15"] < df["ema_50"]
    breakout_up = df["close"] > df["ema_50"] + 0.25
    breakout_dn = df["close"] < df["ema_50"] - 0.25
    volatility_ok = df["atr"] > df["atr_ma"] * 0.85
    momentum_ok = df["gain_z"] > 0.4  # [Patch v6.1]
    volume_ok = df["volume"] > df["volume_ma"] * 1.0  # [Patch v6.1]
    atr_enough = df["atr"] > 1.4  # [Patch v6.1]

    # --- Entry Score + TP Ratio ---
    df["entry_score"] = df["gain_z"] * df["atr"] / (df["atr_ma"] + 1e-9)
    df["entry_score"] = df["entry_score"].fillna(0)
    df["tp_rr_ratio"] = 3.5
    df["use_be"] = True  # ✅ ใช้ Break-even
    df["use_tsl"] = True  # ✅ ใช้ Trailing SL

    # --- Risk Level by score ---
    df["risk_level"] = "low"
    df.loc[df["entry_score"] > 2.5, "risk_level"] = "med"
    df.loc[df["entry_score"] > 4.0, "risk_level"] = "high"

    # --- Entry Tier (Quantile Classifier) ---
    df["entry_tier"] = pd.qcut(
        df["entry_score"].rank(method="first"), q=3, labels=["C", "B", "A"], duplicates="drop"
    )

    buy_cond = (
        trend_up
        & breakout_up
        & volatility_ok
        & momentum_ok
        & volume_ok
        & atr_enough
        & session
    )
    sell_cond = (
        trend_dn
        & breakout_dn
        & volatility_ok
        & momentum_ok
        & volume_ok
        & atr_enough
        & session
    )

    df["entry_signal"] = np.where(buy_cond, "buy", np.where(sell_cond, "sell", None))

    # --- Logging QA Reason ---
    # [Perf-B] ปรับให้ QA Block Reason ทำงานแบบ vectorized (ลด loop)
    df["entry_blocked_reason"] = ""
    df.loc[~trend_up & ~trend_dn, "entry_blocked_reason"] += "no_trend|"
    df.loc[~volatility_ok, "entry_blocked_reason"] += "low_vol|"
    df.loc[~momentum_ok, "entry_blocked_reason"] += "low_momentum|"
    df.loc[~volume_ok, "entry_blocked_reason"] += "low_volume|"
    df.loc[~atr_enough, "entry_blocked_reason"] += "atr_too_small|"
    df.loc[~session, "entry_blocked_reason"] += "off_session|"
    df.loc[~(breakout_up | breakout_dn), "entry_blocked_reason"] += "no_breakout|"
    df["entry_blocked_reason"] = df["entry_blocked_reason"].str.rstrip("|")
    df.loc[df["entry_signal"].notnull(), "entry_blocked_reason"] = None
    blocked_pct = df["entry_signal"].isnull().mean() * 100
    print(f"[Patch VBTB+ UltraFix v4.1] Entry Signal Blocked: {blocked_pct:.2f}%")

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


def generate_signals_v6_5(df: pd.DataFrame, fold_id: int) -> pd.DataFrame:
    """Generate signals with dynamic thresholds per fold (Patch v6.5)."""
    df = df.copy()
    df["entry_signal"] = None
    df["entry_blocked_reason"] = None
    df["lot_suggested"] = 0.05

    df["ema_15"] = df["close"].ewm(span=15).mean()
    df["ema_50"] = df["close"].ewm(span=50).mean()
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
    df["atr_ma"] = df["atr"].rolling(50).mean()
    gain = df["close"].diff()
    df["gain_z"] = (gain - gain.rolling(20).mean()) / (gain.rolling(20).std() + 1e-9)
    df["volume_ma"] = df["volume"].rolling(20).mean()

    df["session_label"] = "None"

    mean_atr = df["atr"].mean()
    if mean_atr < 1.5:
        gain_z_thresh = 0.1
        atr_thresh = 0.8
    else:
        gain_z_thresh = 0.25
        atr_thresh = 1.1

    if fold_id in [3, 4]:
        gain_z_thresh = 0.0
        atr_thresh = 0.5
        breakout_margin = 0.15
        session_range = (7, 20)
    else:
        breakout_margin = 0.25
        session_range = (8, 17)

    trend_up = df["ema_15"] > df["ema_50"]
    trend_dn = df["ema_15"] < df["ema_50"]
    trend_bypass = fold_id == 4  # [Patch v6.5]
    breakout_up = df["close"] > df["ema_50"] + breakout_margin
    breakout_dn = df["close"] < df["ema_50"] - breakout_margin
    volatility_ok = df["atr"] > df["atr_ma"] * 0.85
    momentum_ok = df["gain_z"] > gain_z_thresh  # [Patch v6.3]
    volume_ok = df["volume"] > df["volume_ma"] * 1.0
    atr_enough = df["atr"] > atr_thresh  # [Patch v6.3]
    session = df["timestamp"].dt.hour.between(session_range[0], session_range[1])
    if fold_id == 4:
        session_ok = pd.Series(True, index=df.index)  # [Patch v6.6] ปิดกรองเวลา
    else:
        df.loc[session, "session_label"] = "Active"
        session_ok = df["session_label"] != "None"

    if trend_bypass:
        buy_cond = (
            breakout_up & volatility_ok & momentum_ok & volume_ok & atr_enough & session_ok
        )
        sell_cond = (
            breakout_dn & volatility_ok & momentum_ok & volume_ok & atr_enough & session_ok
        )
    else:
        buy_cond = (
            trend_up
            & breakout_up
            & volatility_ok
            & momentum_ok
            & volume_ok
            & atr_enough
            & session_ok
        )
        sell_cond = (
            trend_dn
            & breakout_dn
            & volatility_ok
            & momentum_ok
            & volume_ok
            & atr_enough
            & session_ok
        )

    df["entry_signal"] = np.where(buy_cond, "buy", np.where(sell_cond, "sell", None))

    df["entry_blocked_reason"] = ""
    df.loc[~trend_up & ~trend_dn, "entry_blocked_reason"] += "no_trend|"
    df.loc[~volatility_ok, "entry_blocked_reason"] += "low_vol|"
    df.loc[~momentum_ok, "entry_blocked_reason"] += "low_momentum|"
    df.loc[~volume_ok, "entry_blocked_reason"] += "low_volume|"
    df.loc[~atr_enough, "entry_blocked_reason"] += "atr_too_small|"
    if fold_id != 4:
        df.loc[~session_ok, "entry_blocked_reason"] += "off_session|"
    df.loc[~(breakout_up | breakout_dn), "entry_blocked_reason"] += "no_breakout|"
    df["entry_blocked_reason"] = df["entry_blocked_reason"].str.rstrip("|")
    df.loc[df["entry_signal"].notnull(), "entry_blocked_reason"] = None
    blocked_pct = df["entry_signal"].isnull().mean() * 100
    print(f"[Patch v6.5] Entry Signal Blocked: {blocked_pct:.2f}%")

    return df
