import pandas as pd
import numpy as np

# --- CONFIG FLAGS (Patch v11.1) ---
ENABLE_TP1_TP2 = True
ENABLE_SESSION_FILTER = True
ENABLE_SIGNAL_LOG = True


def apply_tp_logic(entry_price: float, direction: str, rr1: float = 1.5, rr2: float = 3.0, sl_distance: float = 5.0) -> tuple[float, float]:
    """คำนวณเป้าหมาย TP1/TP2 ตาม Risk Reward"""
    tp1 = entry_price + rr1 * sl_distance if direction == "buy" else entry_price - rr1 * sl_distance
    tp2 = entry_price + rr2 * sl_distance if direction == "buy" else entry_price - rr2 * sl_distance
    return tp1, tp2


def generate_entry_signal(row: dict, log_list: list) -> str | None:
    """ระบุตำแหน่งเข้าเทรดและบันทึกลง log"""
    signal = None
    if row.get("rsi", 50) < 30 and row.get("pattern") == "inside_bar":
        signal = "RSI_InsideBar"
    elif row.get("pattern") == "qm":
        signal = "QM"
    elif row.get("pattern") == "fractal_v":
        signal = "FractalV"

    if ENABLE_SIGNAL_LOG:
        log_list.append(
            {
                "time": row.get("timestamp"),
                "entry_price": row.get("close"),
                "signal": signal,
                "session": row.get("session"),
                "risk_mode": row.get("risk_mode", "normal"),
            }
        )

    return signal


def session_filter(row: dict) -> bool:
    """กรองไม่ให้เข้าออเดอร์หากเงื่อนไขเซสชันไม่ผ่าน"""
    if ENABLE_SESSION_FILTER and row.get("session") == "NY":
        if row.get("ny_sl_count", 0) > 3:
            return False
    return True


trade_log_fields = [
    "timestamp",
    "entry_price",
    "exit_price",
    "sl_price",
    "tp1_price",
    "tp2_price",
    "exit_reason",
    "session",
    "risk_mode",
    "entry_signal",
    "mfe",
    "duration_min",
]


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """คำนวณ RSI แบบเวกเตอร์เพื่อลดเวลาประมวลผล"""
    delta = series.diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def generate_signals_v8_0(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """ใช้ logic sniper + TP1/TSL แบบล่าสุด (Patch v8.0)."""
    df = df.copy()

    config = config or {}
    gain_z_thresh = config.get("gain_z_thresh", 0.25)
    ema_slope_min = config.get("ema_slope_min", 0.2)
    atr_thresh_val = config.get("atr_thresh", 1.0)
    sniper_risk_score_min = config.get("sniper_risk_score_min", 2.5)  # [Patch v8.1.7]
    tp_rr_ratio_cfg = config.get("tp_rr_ratio", 4.8)
    volume_ratio = config.get("volume_ratio", 1.0)
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

    # --- Session Info ---
    hour = df["timestamp"].dt.hour
    df["session_name"] = np.where(
        hour < 8, "Asia", np.where(hour < 15, "London", "NY")
    )

    # --- Session Tag ---
    df["session_label"] = "None"
    df.loc[df["timestamp"].dt.hour.between(13, 17), "session_label"] = "NY"
    session = df["session_label"] != "None"

    # คอลัมน์เดิมสำหรับโมดูลอื่น
    df["ema_fast"] = df["ema_15"]
    df["ema_slow"] = df["ema_50"]
    df["ema_slope"] = df["ema_fast"].diff()
    df["rsi_14"] = 100 - (100 / (1 + (
        df["close"].diff().clip(lower=0).rolling(14).mean() /
        (-df["close"].diff().clip(upper=0).rolling(14).mean() + 1e-9)
    )))  # [Patch v7.9] RSI Confirm

    # --- Entry Conditions v4 ---
    trend_up = df["ema_15"] > df["ema_50"]
    trend_dn = df["ema_15"] < df["ema_50"]
    breakout_margin = 0.25
    breakout_up = df["close"].shift(2) > df["ema_50"].shift(2) + breakout_margin  # [Patch v8.0] เพิ่ม delay
    breakout_up_asia = df["close"].shift(1) > df["ema_50"].shift(1)
    breakout_up = breakout_up.where(df["session_name"] != "Asia", breakout_up_asia)  # [Patch v7.6]
    breakout_dn = df["close"].shift(1) < df["ema_50"].shift(1) - breakout_margin  # [Patch v7.2]
    volatility_ok = df["atr"] > df["atr_ma"] * 0.85
    momentum_thresh = np.where(df["session_name"] == "Asia", 0.0, 0.4)
    momentum_ok = df["gain_z"] > momentum_thresh  # [Patch v7.7]
    volume_ok = df["volume"] > df["volume_ma"] * volume_ratio  # [Patch v8.1.1]
    atr_enough = df["atr"] > atr_thresh_val

    df["sniper_risk_score"] = (
        df["gain_z"].clip(0, 2) * 2.5
        + df["ema_slope"].clip(0, 2) * 2.0
        + (df["atr"] / df["atr_ma"]).clip(0.5, 1.5) * 2.0
        + (df["volume"] / df["volume_ma"]).clip(0.8, 1.5) * 3.5
    )
    df["sniper_risk_score"] = df["sniper_risk_score"].fillna(0)  # [Patch v8.1.1]

    # --- Entry Score + TP Ratio ---
    df["entry_score"] = df["gain_z"] * df["atr"] / (df["atr_ma"] + 1e-9)
    df["entry_score"] = df["entry_score"].fillna(0)
    df["tp_rr_ratio"] = tp_rr_ratio_cfg
    df["use_be"] = True  # ✅ ใช้ Break-even
    df["use_tsl"] = True  # ✅ ใช้ Trailing SL
    df["tp1_rr_ratio"] = 2.0  # เพิ่ม TP1 ระยะ
    df["use_dynamic_tsl"] = True  # TSL เฉพาะเมื่อ RR ≥ 1.0

    # --- Risk Level by score ---
    df["risk_level"] = "low"
    df.loc[df["entry_score"] > 2.5, "risk_level"] = "med"
    df.loc[df["entry_score"] > 4.0, "risk_level"] = "high"

    # --- Entry Tier (Quantile Classifier) ---
    ranks = df["entry_score"].rank(method="first")
    try:
        if ranks.notna().sum() >= 3:
            df["entry_tier"] = pd.qcut(
                ranks, q=3, labels=["C", "B", "A"], duplicates="drop"
            )
        else:
            raise ValueError("insufficient data")
    except ValueError:
        df["entry_tier"] = "C"
    df["confirm_zone"] = (
        (df["gain_z"] > 0.0)
        & (df["ema_slope"] > 0.02)
        & ((df["atr"] > 0.15) | ((df["atr"] / df["atr_ma"]) > 0.8))
        & ((df["volume"] > df["volume_ma"] * 0.4) | df["volume"].isna())
        & (df["entry_score"] > 0)
    )  # [Patch] ปรับ ConfirmZone ให้เข้มงวดขึ้น

    sniper_zone = (
        (df["sniper_risk_score"] >= sniper_risk_score_min)
        & (df["gain_z"] > gain_z_thresh)
        & (df["ema_slope"] > ema_slope_min)
        & df["confirm_zone"]
    )
    sniper_zone |= (
        (df["sniper_risk_score"] >= sniper_risk_score_min + 0.5)
        & (df["entry_tier"] == "B")
    )  # [Patch v8.0]

    buy_cond = (
        sniper_zone
        & breakout_up
        & volatility_ok
        & volume_ok
        & atr_enough
        & session
    )
    sell_cond = (
        sniper_zone
        & breakout_dn
        & volatility_ok
        & volume_ok
        & atr_enough
        & session
    )

    df["entry_signal"] = np.where(buy_cond, "buy", np.where(sell_cond, "sell", None))

    # --- Logging QA Reason ---
    # [Patch v7.2] Restore entry_blocked_reason logic
    trend_ok = trend_up | trend_dn
    session_ok = session
    conditions = {
        "no_trend": ~trend_ok,
        "low_vol": ~volatility_ok,
        "low_momentum": ~momentum_ok,
        "low_volume": ~volume_ok,
        "atr_too_small": ~atr_enough,
        "off_session": ~session_ok,
        "no_breakout": ~(breakout_up | breakout_dn),
    }

    reason_series = [cond.map({True: name, False: ""}) for name, cond in conditions.items()]
    reason_df = pd.concat(reason_series, axis=1)
    reason_string = reason_df.apply(lambda row: "|".join(filter(None, row)), axis=1)

    # ✅ [Patch v11.9.5] Fix index assignment safely
    reason_string_safe = reason_string.reset_index(drop=True)
    entry_reason_column = pd.Series("N/A", index=df.index)

    if len(reason_string_safe) != len(entry_reason_column):
        raise ValueError(
            f"[Patch QA] ❌ reason_string length mismatch: {len(reason_string_safe)} vs df: {len(entry_reason_column)}"
        )

    entry_reason_column[:] = reason_string_safe.values
    entry_reason_column.loc[df["entry_signal"].notnull()] = None
    df["entry_blocked_reason"] = entry_reason_column
    print(
        f"[Patch v11.9.5] \u2705 entry_blocked_reason assigned: {df['entry_blocked_reason'].notnull().sum()} filled"
    )
    blocked_pct = df["entry_signal"].isnull().mean() * 100
    print(f"[Patch v8.0] Entry Signal Blocked: {blocked_pct:.2f}%")
    return df


def generate_signals_v9_0(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """[Patch v9.0] Gold AI v3.5.3 Signal Generator wrapper."""
    return generate_signals_v8_0(df, config=config)


def generate_signals_unblock_v9_1(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """[Patch v9.1] ปลดล็อกทุกชั้น ใช้ gain_z + sniper_score เป็นหลัก"""
    df = df.copy()
    config = config or {}

    # --- Setup params ---
    gain_z_thresh = config.get("gain_z_thresh", -0.2)
    ema_slope_min = config.get("ema_slope_min", -0.01)
    atr_thresh = config.get("atr_thresh", 0.0)
    sniper_score_min = config.get("sniper_risk_score_min", 2.0)
    tp_rr_ratio = config.get("tp_rr_ratio", 4.0)

    # --- Indicator ---
    df["ema_15"] = df["close"].ewm(span=15).mean()
    df["ema_50"] = df["close"].ewm(span=50).mean()
    df["ema_slope"] = df["ema_15"].diff()
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
    df["atr_ma"] = df["atr"].rolling(50).mean()
    df["gain"] = df["close"].diff()
    df["gain_z"] = (df["gain"] - df["gain"].rolling(20).mean()) / (df["gain"].rolling(20).std() + 1e-9)

    # --- Score ---
    df["sniper_score"] = (
        df["gain_z"].clip(0, 2) * 2.5 +
        df["ema_slope"].clip(0, 2) * 2.0 +
        (df["atr"] / df["atr_ma"]).clip(0.5, 1.5) * 1.5
    )
    df["tp_rr_ratio"] = tp_rr_ratio

    # --- Filter ---
    valid = (
        (df["gain_z"] > gain_z_thresh) &
        (df["ema_slope"] > ema_slope_min) &
        (df["atr"] > atr_thresh) &
        (df["sniper_score"] >= sniper_score_min)
    )
    df["entry_signal"] = None
    df.loc[valid & (df["ema_15"] > df["ema_50"]), "entry_signal"] = "buy"
    df.loc[valid & (df["ema_15"] < df["ema_50"]), "entry_signal"] = "sell"

    # --- Logging ---
    blocked_pct = df["entry_signal"].isnull().mean() * 100
    print(f"[Patch v9.1] Entry Signal Blocked: {blocked_pct:.2f}%")
    return df


def generate_signals_profit_v10(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """[Patch v10.0] Entry logic เต็มระบบ: gain_z slope, atr slope, confirm zone, dynamic RR"""
    df = df.copy()
    config = config or {}
    gain_z_thresh = config.get("gain_z_thresh", -0.05)
    ema_slope_min = config.get("ema_slope_min", 0.01)
    atr_thresh = config.get("atr_thresh", 0.15)
    sniper_score_min = config.get("sniper_risk_score_min", 3.0)
    tp_rr_ratio = config.get("tp_rr_ratio", 5.5)

    # Indicators
    df["ema_fast"] = df["close"].ewm(span=15).mean()
    df["ema_slow"] = df["close"].ewm(span=50).mean()
    df["ema_slope"] = df["ema_fast"].diff()
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
    df["atr_ma"] = df["atr"].rolling(50).mean()
    df["gain"] = df["close"].diff()
    df["gain_z"] = (df["gain"] - df["gain"].rolling(20).mean()) / (df["gain"].rolling(20).std() + 1e-9)
    df["gain_z_slope"] = df["gain_z"].diff()
    df["atr_slope"] = df["atr"].diff()
    df["tp_rr_ratio"] = tp_rr_ratio

    df["sniper_score"] = (
        df["gain_z"].clip(0, 2) * 2.5 +
        df["ema_slope"].clip(0, 2) * 2.0 +
        (df["atr"] / df["atr_ma"]).clip(0.5, 1.5) * 2.0
    )

    # Confirm Zone
    confirm = (
        (df["gain_z"] > gain_z_thresh) &
        (df["ema_slope"] > ema_slope_min) &
        (df["atr"] > atr_thresh) &
        (df["sniper_score"] >= sniper_score_min) &
        (df["gain_z_slope"] > 0) & (df["atr_slope"] > 0)
    )

    df["entry_signal"] = None
    df.loc[confirm & (df["ema_fast"] > df["ema_slow"]), "entry_signal"] = "buy"
    df.loc[confirm & (df["ema_fast"] < df["ema_slow"]), "entry_signal"] = "sell"
    blocked_pct = df["entry_signal"].isnull().mean() * 100
    print(f"[Patch v10.0] Entry Signal Blocked: {blocked_pct:.2f}%")
    return df


def generate_signals_v11_scalper_m1(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """[Patch v10.1] QM + Inside Bar + RSI + Fractal สำหรับ M1 Scalper."""
    import numpy as np
    df = df.copy()
    config = config or {}

    gain_z_thresh = config.get("gain_z_thresh", -0.05)
    ema_slope_min = config.get("ema_slope_min", 0.01)
    atr_thresh = config.get("atr_thresh", 0.15)
    sniper_score_min = config.get("sniper_risk_score_min", 3.0)
    tp_rr_ratio = config.get("tp_rr_ratio", 5.5)

    # --- Indicators ---
    df["ema_fast"] = df["close"].ewm(span=15).mean()
    df["ema_slow"] = df["close"].ewm(span=50).mean()
    df["ema_slope"] = df["ema_fast"].diff()
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
    df["atr_ma"] = df["atr"].rolling(50).mean()
    df["gain"] = df["close"].diff()
    df["gain_z"] = (df["gain"] - df["gain"].rolling(20).mean()) / (df["gain"].rolling(20).std() + 1e-9)
    df["gain_z_slope"] = df["gain_z"].diff()
    df["atr_slope"] = df["atr"].diff()
    df["rsi"] = 100 - (100 / (1 + (
        df["close"].diff().clip(lower=0).rolling(14).mean() /
        (-df["close"].diff().clip(upper=0).rolling(14).mean() + 1e-9)
    )))

    df["entry_signal"] = None
    df["tp_rr_ratio"] = tp_rr_ratio

    confirm_zone = (
        (df["gain_z"] > gain_z_thresh)
        & (df["ema_slope"] > ema_slope_min)
        & (df["atr"] > atr_thresh)
        & (df["gain_z_slope"] > 0)
        & (df["atr_slope"] > 0)
    )

    df["qml_zone"] = (
        (df["high"] > df["high"].shift(1))
        & (df["low"] < df["low"].shift(1))
        & (df["close"] < df["close"].shift(1))
    )

    df["inside_bar"] = (df["high"] < df["high"].shift(1)) & (df["low"] > df["low"].shift(1))
    df["break_up"] = df["high"] > df["high"].shift(1)
    df["break_down"] = df["low"] < df["low"].shift(1)

    df["fractal_high"] = (df["high"] > df["high"].shift(1)) & (df["high"] > df["high"].shift(-1))
    df["fractal_low"] = (df["low"] < df["low"].shift(1)) & (df["low"] < df["low"].shift(-1))

    rsi_buy = df["rsi"] < 30
    rsi_sell = df["rsi"] > 70

    buy_cond = (
        (
            (df["qml_zone"] & df["fractal_low"] & df["inside_bar"] & df["break_up"])
            | (df["inside_bar"] & df["break_up"] & rsi_buy)
            | (df["fractal_low"] & rsi_buy)
        )
        & confirm_zone
    )

    sell_cond = (
        (
            (df["qml_zone"] & df["fractal_high"] & df["inside_bar"] & df["break_down"])
            | (df["inside_bar"] & df["break_down"] & rsi_sell)
            | (df["fractal_high"] & rsi_sell)
        )
        & confirm_zone
    )

    df.loc[buy_cond, "entry_signal"] = "buy"
    df.loc[sell_cond, "entry_signal"] = "sell"
    print("[Patch v10.1] Entry Signal Activated: QM + InsideBar + RSI + Fractal")
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
    breakout_up = df["close"].shift(2) > df["ema_50"].shift(2) + breakout_margin  # [Patch v8.0] เพิ่ม delay
    volatility_ok = df["atr"] > df["atr_ma"] * 0.85
    breakout_dn = df["close"].shift(1) < df["ema_50"].shift(1) - breakout_margin  # [Patch v7.2]
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

    conditions = {
        "no_trend": ~(trend_up | trend_dn),
        "low_vol": ~volatility_ok,
        "low_momentum": ~momentum_ok,
        "low_volume": ~volume_ok,
        "atr_too_small": ~atr_enough,
        "off_session": ~session_ok,
        "no_breakout": ~(breakout_up | breakout_dn),
    }

    reason_series = []
    for name, cond in conditions.items():
        reason_series.append(cond.map({True: name, False: ""}))
    reason_df = pd.concat(reason_series, axis=1)
    reason_string = reason_df.apply(lambda row: "|".join(filter(None, row)), axis=1)

    entry_reason_column = pd.Series("N/A", index=df.index)
    entry_reason_column.loc[reason_string.index] = reason_string
    entry_reason_column.loc[df["entry_signal"].notnull()] = None

    df["entry_blocked_reason"] = entry_reason_column
    blocked_pct = df["entry_signal"].isnull().mean() * 100
    print(f"[Patch v6.5] Entry Signal Blocked: {blocked_pct:.2f}%")

    return df


# --- Patch v7.1 ---


def generate_signals_v7_1(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Apply Sniper TP ratio boost (Patch v7.1)."""
    df = generate_signals(df, config=config)
    df["tp_rr_ratio"] = 4.8
    sniper_boost = (df["entry_tier"] == "A") & (df["gain_z"] > 0.8)
    df.loc[sniper_boost, "tp_rr_ratio"] = 7.5
    return df


# --- Patch v8.1.4 ---


def generate_signals(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """[Patch v8.1.4] Fallback ถูกยกเลิก — forward ไปยัง logic sniper v8.0"""
    return generate_signals_v8_0(df, config=config)  # ใช้ sniper confirm zone, risk score, delay, TP boost เท่านั้น



def simulate_trades_with_tp(df: pd.DataFrame, sl_distance: float = 5.0):
    """Simple trade simulator with TP1/TP2 and logging"""
    logs: list[dict] = []
    trades: list[dict] = []

    for _, row in df.iterrows():
        if not session_filter(row):
            continue

        entry_price = row["close"]
        direction = "buy" if row.get("signal") == "long" else "sell"
        tp1_price, tp2_price = apply_tp_logic(entry_price, direction, sl_distance=sl_distance)

        trade = {
            "timestamp": row["timestamp"],
            "entry_price": entry_price,
            "tp1_price": tp1_price,
            "tp2_price": tp2_price,
            "sl_price": entry_price - sl_distance if direction == "buy" else entry_price + sl_distance,
            "exit_price": None,
            "exit_reason": None,
            "entry_signal": generate_entry_signal(row, logs),
            "session": row.get("session"),
            "risk_mode": row.get("risk_mode", "normal"),
        }
        trades.append(trade)

    return trades, logs
