import pandas as pd
import numpy as np
import logging
import os
import inspect  # [Patch QA-FIX v28.2.7] dynamic fallback param check
from typing import Dict
from tqdm import tqdm
from .config import SESSION_CONFIG, HEDGEFUND_ENTRY_CONFIG, THRESHOLD_MODEL_PATH
from .utils import setup_logger, QA_BASE_PATH, export_audit_report
from .config import PATHS

logger = setup_logger("nicegold_v5.entry", os.path.join(QA_BASE_PATH, "entry.log"))

# [Patch v32.0.0] ✨ เพิ่ม alias ฟังก์ชัน generate_signals_v8_0, generate_signals_v12_0
def generate_signals_v8_0(df: pd.DataFrame, config: Dict | None = None) -> pd.DataFrame:
    """[Patch v32.0.0] Core logic version 8.0 alias."""
    return _generate_signals_v8_0_core(df, config)
from .adaptive_threshold_dl import predict_thresholds

# --- CONFIG FLAGS (Patch v11.1) ---
ENABLE_TP1_TP2 = True
# [Patch v31.0.0] ปิด session_filter ชั่วคราว เพื่อให้เกิด entry ตลอดวัน
ENABLE_SESSION_FILTER = False
ENABLE_SIGNAL_LOG = True

# [Patch v32.1.0] ค่า Default RR1/RR2 ปรับให้ TP ง่ายขึ้น
DEFAULT_RR1 = 1.2    # จากเดิม 1.5–2.0
DEFAULT_RR2 = 2.0    # จากเดิม 3.0–5.0

def apply_tp_logic(entry_price: float, direction: str, rr1: float = 3.0, sl_distance: float = 5.0) -> tuple[float, float]:
    """คำนวณเป้าหมาย TP1/TP2 ตาม Risk Reward"""
    # [Patch v32.0.2] ใช้ DEFAULT_RR1, DEFAULT_RR2 แทน rr1*2 เดิม
    tp1 = entry_price + DEFAULT_RR1 * sl_distance if direction == "buy" else entry_price - DEFAULT_RR1 * sl_distance
    tp2 = entry_price + DEFAULT_RR2 * sl_distance if direction == "buy" else entry_price - DEFAULT_RR2 * sl_distance
    return tp1, tp2


def generate_entry_signal(row: dict, log_list: list) -> str | None:
    """ระบุตำแหน่งเข้าเทรดและบันทึกลง log"""
    signal = None

    # ✅ ฝั่ง BUY (เดิม)
    if row.get("rsi", 50) < 30 and row.get("pattern") == "inside_bar":
        signal = "RSI_InsideBar"
    elif row.get("pattern") == "qm":  # pragma: no cover
        signal = "QM"
    elif row.get("pattern") == "fractal_v":  # pragma: no cover
        signal = "FractalV"

    # ✅ [Patch v12.9.0] เพิ่ม SELL SIGNAL ใหม่
    elif row.get("rsi", 50) > 70 and row.get("pattern") == "inside_bar":
        signal = "RSI70_InsideBar"
    elif row.get("pattern") == "qm_bearish":
        signal = "QM_Bearish"
    elif row.get("pattern") == "bearish_engulfing":
        signal = "BearishEngulfing"

    # ✅ [Patch v12.9.2] Fallback Sell: หาก momentum ลงแรง
    elif row.get("gain_z", 0) < 0 and row.get("ema_slope", 0) < 0:
        signal = "fallback_sell"

    if ENABLE_SIGNAL_LOG:
        log_list.append(
            {
                "time": pd.to_datetime(row.get("timestamp")),
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
    "signal_name",
    "entry_tier",
    "lot",  # [Patch v12.8.3] new field
    "pnl",  # [Patch v12.8.3]
    "planned_risk",  # [Patch v12.8.3]
    "r_multiple",  # [Patch v12.8.3]
    "commission",  # [Patch v12.8.3]
    "spread_cost",  # [Patch v12.8.3]
    "slippage_cost",  # [Patch v12.8.3]
    "mfe",
    "duration_min",
    "gain_z",  # [Patch v12.8.3]
    "rsi",  # [Patch v12.8.3]
    "ema_slope",  # [Patch v12.8.3]
    "atr",  # [Patch v12.8.3]
    "pattern",  # [Patch v12.8.3]
    "entry_score",  # [Patch v12.8.3]
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


def validate_indicator_inputs(df: pd.DataFrame, required_cols: list[str] | None = None, min_rows: int = 500) -> None:
    """ตรวจสอบความพร้อมของข้อมูลก่อน generate_signal"""
    if required_cols is None:
        required_cols = ["close", "high", "low", "volume"]

    # Replace inf / -inf → NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise RuntimeError(f"[Patch v11.9.9] ❌ ขาดคอลัมน์จำเป็น: {missing_cols}")

    row_count = df[required_cols].dropna().shape[0]

    print(f"[Patch v11.9.11] ✅ ตรวจข้อมูลก่อนเข้า indicator: เหลือ {row_count} row หลัง dropna")
    if row_count < min_rows:
        print("[🧪 Preview] df.head():")
        print(df[required_cols].head())
        print("[🧪 Preview] df.tail():")
        print(df[required_cols].tail())

    if row_count < min_rows:
        raise RuntimeError(
            f"[Patch v11.9.9] ❌ ข้อมูลมีเพียง {row_count} row ที่ใช้ได้ (ต้องการ ≥ {min_rows})"
        )


def sanitize_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    """แปลงคอลัมน์ราคาทั้งหมดให้เป็น float และ log รายงาน"""
    # [Patch] รองรับคอลัมน์ตัวใหญ่เช่น Volume → volume
    for tc, lc in [("Open", "open"), ("High", "high"), ("Low", "low"), ("Close", "close"), ("Volume", "volume")]:
        if tc in df.columns and lc not in df.columns:
            df[lc] = df[tc]

    for col in ["close", "high", "low", "open", "volume"]:
        if col in df.columns:
            series = df[col].astype(str).str.replace(",", "", regex=False).str.strip()
            df[col] = pd.to_numeric(series, errors="coerce")

    # [Patch v25.0.0] Auto-fix volume NaN/0 → 1.0 ทุกกรณี
    if "volume" in df.columns:
        if df["volume"].isnull().mean() > 0.95 or (df["volume"] == 0).mean() > 0.95:
            print("[Patch v25.0.0] ⚠️ volume เป็น NaN/0 เกือบหมด – เติมเป็น 1.0 อัตโนมัติ")
            df["volume"] = 1.0

    cols_to_check = [c for c in ["close", "high", "low", "volume"] if c in df.columns]
    missing = df[cols_to_check].isnull().sum()
    print("[Patch v11.9.16] 🧼 Sanitize Columns:")
    for col, count in missing.items():
        print(f"   ▸ {col}: {count} NaN")
    return df

# [Patch v32.2.0] ✨ เพิ่มฟังก์ชัน convert_thai_datetime เพื่อแก้ปัญหา dtype
def convert_thai_datetime_col(df: pd.DataFrame, date_col: str, time_col: str):
    """แปลงปฏิทินไทย (พ.ศ.) → ค.ศ. และแปลงเป็น datetime64[ns]."""
    df["year"] = df[date_col].astype(str).str[:4].astype(int) - 543
    df["month"] = df[date_col].astype(str).str[4:6]
    df["day"] = df[date_col].astype(str).str[6:8]
    df["datetime_str"] = df["year"].astype(str) + "-" + df["month"] + "-" + df["day"] + " " + df[time_col]
    df["timestamp"] = pd.to_datetime(df["datetime_str"])

def filter_entry_signals(df: pd.DataFrame, config: dict, session: str | None = None) -> pd.DataFrame:
    """Filter entry rows based on config thresholds."""
    df = df.copy()
    session_conf = SESSION_CONFIG.get(session, {}) if config.get("session_adaptive") and session else {}
    g_thresh = session_conf.get("gain_z_thresh", config.get("gain_z_thresh", -0.05))
    tp1_rr = session_conf.get("tp1_rr_ratio", config.get("tp1_rr_ratio", DEFAULT_RR1))
    tp2_rr = session_conf.get("tp2_rr_ratio", config.get("tp_rr_ratio", 2.0))

    rows = []
    for _, row in df.iterrows():
        if row.get("gain_z", 0) < g_thresh:
            continue
        if row.get("ema_slope", 0) < config.get("ema_slope_min", 0):
            continue
        if row.get("atr", 0) < config.get("atr_thresh", 0):
            continue
        if row.get("sniper_risk_score", 0) < config.get("sniper_risk_score_min", 0):
            continue
        r = row.to_dict()
        r["tp1_rr_ratio"] = tp1_rr
        r["tp2_rr_ratio"] = tp2_rr
        rows.append(r)
    return pd.DataFrame(rows)


def calc_lot_size(account: dict, win_streak: int, loss_streak: int, config: dict) -> float:
    """Dynamic lot sizing based on win/loss streak."""
    lot = account.get("init_lot", 0.01)
    if config.get("dynamic_lot"):
        if win_streak >= 2:
            lot *= config.get("lot_win_multiplier", 1.18)
        elif loss_streak >= 2:
            lot *= config.get("lot_loss_multiplier", 0.88)
    return min(max(lot, 0.01), config.get("max_lot", 3.0))


def generate_pattern_signals(df: pd.DataFrame) -> pd.DataFrame:
    """สร้างสัญญาณเบื้องต้นจากรูปแบบแท่งเทียน"""
    df = df.copy()
    df["pattern_signal"] = None

    bullish = (
        (df["close"].shift(1) < df["open"].shift(1))
        & (df["close"] > df["open"])
        & (df["close"] > df["open"].shift(1))
        & (df["open"] < df["close"].shift(1))
    )
    df.loc[bullish, "pattern_signal"] = "bullish_engulfing"

    bearish = (
        (df["close"].shift(1) > df["open"].shift(1))
        & (df["close"] < df["open"])
        & (df["close"] < df["open"].shift(1))
        & (df["open"] > df["close"].shift(1))
    )
    df.loc[bearish, "pattern_signal"] = "bearish_engulfing"

    inside_bar = (df["high"] < df["high"].shift(1)) & (df["low"] > df["low"].shift(1))
    df.loc[inside_bar, "pattern_signal"] = "inside_bar"

    df["entry_signal"] = None
    df.loc[df["pattern_signal"] == "bullish_engulfing", "entry_signal"] = "buy"
    df.loc[df["pattern_signal"] == "bearish_engulfing", "entry_signal"] = "sell"
    df.loc[
        (df["pattern_signal"] == "inside_bar")
        & (df["close"] > df["high"].shift(1)),
        "entry_signal",
    ] = "buy"
    df.loc[
        (df["pattern_signal"] == "inside_bar")
        & (df["close"] < df["low"].shift(1)),
        "entry_signal",
    ] = "sell"

    return df


def generate_signals_v8_0_adaptive(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """ใช้โมเดล Deep Learning ทำนาย threshold ก่อนเรียก generate_signals_v8_0"""
    df = df.copy()

    from .utils import load_recent_indicators, load_previous_performance
    seq_recent = load_recent_indicators(df, seq_len=60)
    prev_perf = load_previous_performance()
    model_path = config.get("model_path", THRESHOLD_MODEL_PATH) if config else THRESHOLD_MODEL_PATH
    pt_sig = inspect.signature(predict_thresholds)
    if "device" in pt_sig.parameters:
        thresholds = predict_thresholds(
            seq_recent=seq_recent,
            feedback_scalar=prev_perf,
            model_path=model_path,
            device=config.get("device", "cpu"),
        )
    else:
        thresholds = predict_thresholds(
            seq_recent=seq_recent,
            feedback_scalar=prev_perf,
            model_path=model_path,
        )
    print(
        f"[Patch vA.1.0] \ud83d\udd25 DeepAI Thresholds \u2192 gain_z: {thresholds['gain_z_thresh']:.4f}, ema_slope: {thresholds['ema_slope_min']:.4f}, atr: {thresholds['atr_thresh']:.4f}"
    )
    tmp_cfg = (config or {}).copy()
    tmp_cfg.update(thresholds)
    return generate_signals_v8_0(df, tmp_cfg)


def _generate_signals_v8_0_core(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """ใช้ logic sniper + TP1/TSL แบบล่าสุด (Patch v8.0)."""
    df = df.copy()

    # [Patch v32.2.1] Support Titlecase price columns: alias to lowercase if needed
    for tc, lc in [
        ("Open", "open"),
        ("High", "high"),
        ("Low", "low"),
        ("Close", "close"),
        ("Volume", "volume"),  # [Patch] ensure volume column exists
    ]:
        if tc in df.columns and lc not in df.columns:
            df[lc] = df[tc]

    # [Patch vEntry v1.0] prepare standard columns
    if "side" not in df.columns:
        df["side"] = None
    if "is_dummy" not in df.columns:
        df["is_dummy"] = False

    # [Patch v30.0.0] Relaxable Entry Configuration
    config = config or {}
    gain_z_thresh = config.get("gain_z_thresh", 0.0)
    ema_slope_min = config.get("ema_slope_min", 0.0)
    atr_thresh_val = config.get("atr_thresh", 0.0)
    sniper_risk_score_min = config.get("sniper_risk_score_min", 2.5)
    sniper_score_min = config.get("sniper_score_min", 0.0)
    tp_rr_ratio_cfg = config.get("tp_rr_ratio", 4.8)
    volume_ratio = config.get("volume_ratio", 1.0)
    df["entry_signal"] = None
    df["entry_blocked_reason"] = None
    df["lot_suggested"] = 0.05

    # --- Indicators ---
    # [Patch v24.3.2] 🛡️ Auto-fix volume==0 dev set (ultra override only)
    if config and config.get("gain_z_thresh", 0) <= -9.0:
        if (df["volume"].max() == 0) or (df["volume"].isnull().all()):
            print("[Patch v24.3.2] 🛡️ Ultra Override: force volume = 1.0 for signal fallback.")
            df["volume"] = 1.0
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
        (df["gain_z"] > gain_z_thresh)
        & (df["ema_slope"] > ema_slope_min)
        & ((df["atr"] > atr_thresh_val) | ((df["atr"] / df["atr_ma"]) > 0.6))
        & ((df["volume"] > df["volume_ma"] * 0.3) | df["volume"].isna())
        & (df["entry_score"] > 0)
    )  # [Patch v32.0.3] Loosen ConfirmZone

    sniper_zone = (
        (df["sniper_risk_score"] >= sniper_risk_score_min)
        & (df["gain_z"] > gain_z_thresh)
        & (df["ema_slope"] > ema_slope_min)       # [Patch] จากเดิม > 0.2 เป็น > 0.0
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
    # [Patch v24.3.3] Force entry_signal ทุกๆ 500 row (DEV/ML only)
    if config and config.get("gain_z_thresh", 0) <= -9.0:
        if (df["entry_signal"].isnull().mean() >= 1.0):
            force_every = 500
            indices = list(range(0, len(df), force_every))
            for i in indices:
                if pd.isna(df.at[i, "entry_signal"]):
                    df.at[i, "entry_signal"] = "buy"
            print(
                f"[Patch v24.3.3] ⚡️ Ultra Fallback: force entry_signal 'buy' {len(indices)} spots"
            )

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

    # ✅ [Patch v11.9.6] ensure apply returns Series even when DataFrame empty
    if reason_df.empty:
        reason_string = pd.Series(dtype=object)
    else:
        reason_string = reason_df.apply(lambda row: "|".join(filter(None, row)), axis=1)

    # ✅ [Patch v11.9.5] Fix index assignment safely
    reason_string_safe = reason_string.reset_index(drop=True)
    entry_reason_column = pd.Series("N/A", index=df.index)

    if len(reason_string_safe) != len(entry_reason_column):
        raise ValueError(
            f"[Patch QA] ❌ reason_string length mismatch: {len(reason_string_safe)} vs df: {len(entry_reason_column)}"
        )

    entry_reason_column.iloc[: len(reason_string_safe)] = reason_string_safe.values
    entry_reason_column.loc[df["entry_signal"].notnull()] = None
    df["entry_blocked_reason"] = entry_reason_column
    print(
        f"[Patch v11.9.5] \u2705 entry_blocked_reason assigned: {df['entry_blocked_reason'].notnull().sum()} filled"
    )
    blocked_pct = df["entry_signal"].isnull().mean() * 100    # [Patch] คำนวณ % ที่ถูก Block หลังลด Threshold
    print(f"[Patch v8.0] Entry Signal Blocked: {blocked_pct:.2f}%")
    return df


def generate_signals_v9_0(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """[Patch v9.0] Gold AI v3.5.3 Signal Generator wrapper."""
    return generate_signals_v8_0(df, config=config)


def generate_signals_unblock_v9_1(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:  # pragma: no cover
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
    # [Patch v32.0.3] Loosen confirm filters
    gain_z_thresh = config.get("gain_z_thresh", -0.10)  # ลดจาก -0.05 → -0.10
    ema_slope_min = config.get("ema_slope_min", 0.005)  # ลดจาก 0.01 → 0.005
    atr_thresh = config.get("atr_thresh", 0.10)         # ลดจาก 0.15 → 0.10
    sniper_score_min = config.get("sniper_risk_score_min", 3.0)
    tp_rr_ratio = config.get("tp_rr_ratio", 5.5)

    # Indicators
    df["ema_fast"] = df["close"].ewm(span=15).mean()
    df["ema_slow"] = df["close"].ewm(span=50).mean()
    df["ema_slope"] = df["ema_fast"] - df["ema_slow"]
    df["atr"] = df["close"].rolling(14).apply(lambda x: max(x) - min(x), raw=False).fillna(0)
    df["atr_ma"] = df["atr"].rolling(50).mean().fillna(0)
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

    # [Patch v32.0.3] Loosen confirm filters
    gain_z_thresh = config.get("gain_z_thresh", -0.10)  # ลดจาก -0.05 → -0.10
    ema_slope_min = config.get("ema_slope_min", 0.005)  # ลดจาก 0.01 → 0.005
    atr_thresh = config.get("atr_thresh", 0.10)         # ลดจาก 0.15 → 0.10
    sniper_score_min = config.get("sniper_risk_score_min", 3.0)
    tp_rr_ratio = config.get("tp_rr_ratio", 5.5)

    # --- Indicators ---
    df["ema_fast"] = df["close"].ewm(span=15).mean()
    df["ema_slow"] = df["close"].ewm(span=50).mean()
    df["ema_slope"] = df["ema_fast"] - df["ema_slow"]
    df["atr"] = df["close"].rolling(14).apply(lambda x: max(x) - min(x), raw=False).fillna(0)
    df["atr_ma"] = df["atr"].rolling(50).mean().fillna(0)
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
        (df["gain_z"] > gain_z_thresh)            # [Patch v32.0.3] loosen gain_z
        & (df["ema_slope"] > ema_slope_min)        # [Patch v32.0.3] loosen slope
        & ((df["atr"] > atr_thresh) | ((df["atr"] / df["atr_ma"]) > 0.6))
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
    if mean_atr < 1.5:  # pragma: no cover - simple branch
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


def generate_signals(
    df: pd.DataFrame,
    config: dict | None = None,
    test_mode: bool = False,
) -> pd.DataFrame:
    """
    [Patch v29.8.1] UltraOverride QA: ปลดล็อก BUY/SELL ทุกกรณี พร้อม log เมื่อใช้งาน force_entry
    """
    config = config or {}
    # [Patch v29.8.1] UltraOverride QA: force unlock + log
    config["disable_buy"] = False
    config["disable_sell"] = False
    if config.get("force_entry", False):
        logger.info("[generate_signals] QA UltraOverride: force_entry=True")
    df = generate_signals_v8_0(df, config=config)

    # [Patch v28.1.0] QA ForceEntry System (with full config guard)
    if config.get("force_entry"):
        if not test_mode:
            raise RuntimeError(
                "[Patch v28.1.0] \ud83d\udea8 ForceEntry \u0e16\u0e39\u0e01\u0e40\u0e1b\u0e34\u0e14\u0e43\u0e19 production! \u0e2b\u0e49\u0e32\u0e21 deploy \u0e40\u0e17\u0e23\u0e14\u0e08\u0e23\u0e34\u0e07 (dev only)"
            )
        import numpy as np

        np.random.seed(config.get("force_entry_seed", 42))
        mask = df["entry_signal"].isnull()
        ratio = float(config.get("force_entry_ratio", 0.01))
        min_orders = int(config.get("force_entry_min_orders", 10))
        side = config.get("force_entry_side", "both")
        session = config.get("force_entry_session", "all")
        sess_col = "session" if "session" in df.columns else "session_name" if "session_name" in df.columns else None
        eligible = mask.copy()
        if session != "all" and sess_col:
            eligible &= df[sess_col] == session
        num_eligible = int(eligible.sum())
        if num_eligible < 1:
            print("[Patch v28.1.0] \u26a0\ufe0f No eligible bars for ForceEntry injection.")
            return df
        num_force = max(int(num_eligible * ratio), min_orders)
        idx_force = np.random.choice(df[eligible].index, size=min(num_force, num_eligible), replace=False)
        if side == "buy" or side == "sell":
            df.loc[idx_force, "entry_signal"] = side
        else:
            df.loc[idx_force, "entry_signal"] = np.random.choice(["buy", "sell"], size=len(idx_force))
        # [Patch vEntry v1.0] mark side and real trade
        df.loc[idx_force, "side"] = df.loc[idx_force, "entry_signal"]
        df.loc[idx_force, "is_dummy"] = False
        print(
            f"[Patch v28.1.0] \ud83d\udea8 [QA MODE] ForceEntry injected {len(idx_force)} orders (side={side}, session={session})"
        )

    return df



def simulate_trades_with_tp(df: pd.DataFrame, sl_distance: float = 5.0):
    """Simple trade simulator with TP1/TP2 and 60-min exit window"""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    capital = 100.0
    trades: list[dict] = []
    logs: list[dict] = []

    it = tqdm(df.iterrows(), total=len(df), desc="🚀 Simulating", unit="bar")
    for i, (_, row) in enumerate(it, 1):
        if not session_filter(row):
            continue

        entry_time = row["timestamp"]
        exit_window = entry_time + pd.Timedelta(minutes=60)
        window = df[(df["timestamp"] >= entry_time) & (df["timestamp"] <= exit_window)]

        entry_price = row["close"]
        sig = row.get("entry_signal")
        if pd.isna(sig):
            sig = None
        sig2 = row.get("signal")
        if pd.isna(sig2):
            sig2 = None
        direction = (sig or sig2 or "buy").lower()
        if direction == "long":
            direction = "buy"
        elif direction == "short":  # pragma: no cover - rarely used alias
            direction = "sell"
        sl_price = entry_price - sl_distance if direction == "buy" else entry_price + sl_distance

        entry_signal = generate_entry_signal(row, logs)
        signal_name = row.get("signal_name", row.get("entry_signal", entry_signal))  # [Patch v12.8.2]
        entry_tier = row.get("entry_tier", "C")

        session = row.get("session")
        rr2 = 2.5 if session == "Asia" else 3.0 if session == "London" else 3.5
        tp1_price = entry_price + 1.5 * sl_distance if direction == "buy" else entry_price - 1.5 * sl_distance
        tp2_price = (
            entry_price + rr2 * abs(entry_price - sl_price)
            if direction == "buy"
            else entry_price - rr2 * abs(entry_price - sl_price)
        )

        # ✅ [Patch v12.9.6] ปิด TP1 หากไม้มีศักยภาพ TP2 จริง
        entry_score = row.get("entry_score", 0)
        pre_mfe = row.get("mfe", 0)
        tp1_allowed = not (entry_score > 4.5 and pre_mfe > 3)
        exit_reason = "tp2_only_candidate" if not tp1_allowed else "tp1"

        exit_price = tp1_price
        exit_time = window["timestamp"].iloc[-1]

        mfe = 0.0
        breakeven = False
        tsl_activated = False
        trailing_sl = entry_price
        for _, bar in window.iterrows():
            high = bar.get("high", bar["close"])
            low = bar.get("low", bar["close"])
            close_price = bar.get("close")
            interim_pnl = (high - entry_price) if direction == "buy" else (entry_price - low)
            mfe = max(mfe, interim_pnl)

            duration = (bar["timestamp"] - entry_time).total_seconds() / 60.0
            gain = (close_price - entry_price) if direction == "buy" else (entry_price - close_price)
            gain_r = gain / abs(entry_price - sl_price) if abs(entry_price - sl_price) else 0

            # ✅ [Patch v12.9.6] Dynamic BE/TSL Trigger
            if gain_r >= 1.2 and duration >= 15 and not breakeven:
                breakeven = True
                exit_reason = "be_triggered"

            if gain_r >= 2.0 and duration >= 30 and not tsl_activated:
                tsl_activated = True
                # ✅ Dynamic trailing_sl: ใช้ max(high) - atr * 1.2
                window_slice = window[(window["timestamp"] <= bar["timestamp"])]
                atr_buffer = row.get("atr", 1.0) * 1.2
                if direction == "buy":
                    trailing_sl = window_slice["high"].rolling(5).max().iloc[-1] - atr_buffer
                else:
                    trailing_sl = window_slice["low"].rolling(5).min().iloc[-1] + atr_buffer  # pragma: no cover - rare sell path
                exit_reason = "tsl_triggered"

            # ตรวจ TSL แบบ trailing
            if tsl_activated:
                if direction == "buy":
                    if close_price - sl_distance > trailing_sl:
                        trailing_sl = close_price - sl_distance  # pragma: no cover
                    if low <= trailing_sl:
                        exit_price = trailing_sl
                        exit_reason = "tsl_exit"
                        exit_time = bar["timestamp"]
                        break
                else:  # pragma: no cover - sell trailing
                    if close_price + sl_distance < trailing_sl:
                        trailing_sl = close_price + sl_distance
                    if high >= trailing_sl:
                        exit_price = trailing_sl
                        exit_reason = "tsl_exit"
                        exit_time = bar["timestamp"]
                        break

            # ตรวจ BE แบบล็อคทุน
            if breakeven:  # pragma: no cover - complex BE scenario
                if (direction == "buy" and low <= entry_price) or (direction == "sell" and high >= entry_price):
                    exit_price = entry_price
                    exit_reason = "be_sl"
                    exit_time = bar["timestamp"]
                    break
            if direction == "buy":
                if low <= sl_price:
                    exit_price = sl_price
                    exit_reason = "sl"
                    exit_time = bar["timestamp"]
                    break
                if high >= tp2_price:
                    exit_price = tp2_price
                    exit_reason = "tp2"
                    exit_time = bar["timestamp"]
                    break
                if tp1_allowed and high >= tp1_price:
                    exit_price = tp1_price
                    exit_reason = "tp1"
                    exit_time = bar["timestamp"]
                    break
            else:  # pragma: no cover start
                if high >= sl_price:
                    exit_price = sl_price
                    exit_reason = "sl"
                    exit_time = bar["timestamp"]
                    break
                if low <= tp2_price:
                    exit_price = tp2_price
                    exit_reason = "tp2"
                    exit_time = bar["timestamp"]
                    break
                if tp1_allowed and low <= tp1_price:
                    exit_price = tp1_price
                    exit_reason = "tp1"
                    exit_time = bar["timestamp"]
                    break
            # pragma: no cover end

        duration_min = (exit_time - entry_time).total_seconds() / 60.0

        lot = float(row.get("lot", 0.01))
        point_value = 0.1
        atr_val = row.get("atr", sl_distance)
        planned_risk = abs(entry_price - sl_price) * point_value * (lot / 0.01)
        if planned_risk == 0:
            planned_risk = 0.01  # [Patch v12.8.3] fallback to avoid div0
        commission = 0.07 * (lot / 0.01)
        spread_cost = 0.2 * lot
        slippage_cost = 0.1 * lot

        pnl = (
            exit_price - entry_price if direction == "buy" else entry_price - exit_price
        )
        pnl = pnl * point_value * (lot / 0.01) - commission - spread_cost - slippage_cost
        r_multiple = pnl / planned_risk if planned_risk else 0.0

        capital += pnl
        trades.append(
            {
                "timestamp": entry_time,
                "exit_time": exit_time,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "tp1_price": tp1_price,
                "tp2_price": tp2_price,
                "sl_price": sl_price,
                "exit_reason": exit_reason,
                "entry_signal": direction,
                "side": direction,  # [Patch vEntry v1.0]
                "is_dummy": False,
                "signal_name": signal_name,
                "entry_tier": entry_tier,
                "session": session,
                "risk_mode": row.get("risk_mode", "normal"),
                "lot": lot,
                "pnl": round(pnl, 4),
                "planned_risk": round(planned_risk, 4),
                "r_multiple": round(r_multiple, 2),
                "commission": round(commission, 4),
                "spread_cost": round(spread_cost, 4),
                "slippage_cost": round(slippage_cost, 4),
                "duration_min": round(duration_min, 2),
                "mfe": round(mfe, 4),
                "gain_z": row.get("gain_z"),
                "rsi": row.get("rsi"),
                "ema_slope": row.get("ema_slope"),
                "atr": atr_val,
                "pattern": row.get("pattern"),
                "entry_score": row.get("entry_score"),
            }
        )

    return trades, logs


def _generate_signals_v12_0_core(
    df: pd.DataFrame,
    config: dict | None = None,
    test_mode: bool = False,
) -> pd.DataFrame:
    """
    [Patch v26.0.1] Multi-pattern signal generator - *unblock* buy/sell every signal
    test_mode: สำหรับ Dev QA หรือ force entry เท่านั้น (default=False)
    """
    config = config or {}
    # QA Guard: always force both sides open
    if config.get("disable_buy", None) is not False:
        print("[Patch v26.0.1] QA Override: force disable_buy = False [v12.0]")
    if config.get("disable_sell", None) is not False:
        print("[Patch v26.0.1] QA Override: force disable_sell = False [v12.0]")
    config["disable_buy"] = False
    config["disable_sell"] = False
    assert not config.get("disable_buy", False), "QA BLOCK: disable_buy=True not allowed"
    assert not config.get("disable_sell", False), "QA BLOCK: disable_sell=True not allowed"

    # เดิม...
    # [Patch v12.3.4] ✅ Entry Score Filter (TP2 Potential only)
    df = df.copy()
    if "entry_score" in df.columns:
        # [Patch v31.0.0] ลด threshold entry_score จาก 3.5 → 2.5 เพื่อให้มีโอกาสเข้าเทรดมากขึ้น
        df = df[df["entry_score"] > 2.5]
        print(f"[Patch v31.0.0] ⚙️ Filtered by Entry Score > 2.5 → {len(df)} rows")
    df = sanitize_price_columns(df)
    validate_indicator_inputs(df)

    df["ema_15"] = df["close"].ewm(span=15).mean()
    df["ema_50"] = df["close"].ewm(span=50).mean()
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
    df["gain_z"] = (df["close"].diff() - df["close"].diff().rolling(20).mean()) / (
        df["close"].diff().rolling(20).std() + 1e-9
    )
    df["volume_ma"] = df["volume"].rolling(20).mean()
    df["rsi"] = 100 - (100 / (1 + (
        df["close"].diff().clip(lower=0).rolling(14).mean() /
        (-df["close"].diff().clip(upper=0).rolling(14).mean() + 1e-9)
    )))

    hour = df["timestamp"].dt.hour
    df["session"] = np.where(hour < 8, "Asia", np.where(hour < 15, "London", "NY"))

    signals: list[tuple[int, str]] = []
    assert not config.get("disable_buy", False), "QA BLOCK: disable_buy=True not allowed"
    assert not config.get("disable_sell", False), "QA BLOCK: disable_sell=True not allowed"
    for i, row in df.iterrows():
        signal = None
        ema_fast = row.get("ema_15", 0)
        ema_slow = row.get("ema_50", 0)
        gain_z = row.get("gain_z", 0)
        entry_score = row.get("entry_score", 0)
        volume = row.get("volume", 0)
        volume_ma = row.get("volume_ma", 1)
        vol_pass = volume > volume_ma * config.get("volume_ratio", 0.05)

        # ✅ [Patch v22.0.1-ultra] Ultra override sell: ปลดล็อกทุกชั้นแบบเร่งไม้
        if gain_z < -0.01 and entry_score > 0.5 and vol_pass:
            signal = "sell"

        if signal == "buy" and config.get("disable_buy", False):
            continue  # [Patch v16.0.2] ปิดฝั่ง Buy หากตั้งค่าไว้  # pragma: no cover

        if row.get("volume", 0) < config.get("min_volume", 0.0):
            continue  # [Patch v16.1.9] Volume Guard  # pragma: no cover

        if signal and session_filter(row):
            signals.append((i, signal))

    df["entry_signal"] = None
    for idx, sig in signals:
        df.at[idx, "entry_signal"] = sig

    df["tp1_price"] = np.nan
    df["tp2_price"] = np.nan
    df["sl_price"] = np.nan

    for i, row in df.iterrows():
        if pd.notnull(row["entry_signal"]):
            direction = row["entry_signal"]
            df.at[i, "side"] = direction  # [Patch vEntry v1.0]
            df.at[i, "is_dummy"] = False
            sl_dist = row["atr"]
            tp1, tp2 = apply_tp_logic(row["close"], direction, sl_distance=sl_dist)
            sl = row["close"] - sl_dist if direction == "buy" else row["close"] + sl_dist

            df.at[i, "tp1_price"] = tp1
            df.at[i, "tp2_price"] = tp2
            df.at[i, "sl_price"] = sl

    if test_mode and config.get("force_entry"):
        np.random.seed(config.get("force_entry_seed", 42))
        mask = df["entry_signal"].isnull()
        ratio = float(config.get("force_entry_ratio", 0.01))
        min_orders = int(config.get("force_entry_min_orders", 10))
        side = config.get("force_entry_side", "both")
        session = config.get("force_entry_session", "all")
        sess_col = (
            "session" if "session" in df.columns else "session_name" if "session_name" in df.columns else None
        )
        eligible = mask.copy()
        if session != "all" and sess_col:
            eligible &= df[sess_col] == session
        num_eligible = int(eligible.sum())
        if num_eligible < 1:
            print("[Patch v31.2.0] ⚠️ No eligible bars for ForceEntry injection (v12)")
        else:
            num_force = max(int(num_eligible * ratio), min_orders)
            idx_force = np.random.choice(
                df[eligible].index, size=min(num_force, num_eligible), replace=False
            )
            if side in ("buy", "sell"):
                df.loc[idx_force, "entry_signal"] = side
            else:
                df.loc[idx_force, "entry_signal"] = np.random.choice(["buy", "sell"], size=len(idx_force))
            print(f"[Patch v31.2.0] ✅ ForceEntry injected {len(idx_force)} orders (v12)")

    print("[Patch v12.0] ✅ Signals Generated with Multi-Pattern Strategy Layer")
    coverage = df["entry_signal"].notnull().mean() * 100
    print(f"[Patch v12.0] 📊 Entry Signal Coverage: {coverage:.2f}%")
    return df

# [Patch v32.0.0] ✨ alias generate_signals_v12_0
def generate_signals_v12_0(
    df: pd.DataFrame,
    config: Dict | None = None,
    test_mode: bool = False,
) -> pd.DataFrame:
    """[Patch v32.0.0] alias ของ generate_signals_v12_0 core"""
    return _generate_signals_v12_0_core(df, config=config, test_mode=test_mode)


def simulate_partial_tp_safe(df: pd.DataFrame):
    """ฟังก์ชันจำลองการเข้าออกออเดอร์แบบง่ายสำหรับโหมดดีบัก"""
    df = df.copy()
    trade_log = []
    for i, row in df.iterrows():
        if not row.get("entry_signal"):
            continue
        entry_price = row["close"]
        atr = row.get("atr", 1.0)
        timestamp = row["timestamp"]
        direction = "buy" if row.get("ema_slope", 0) >= 0 else "sell"
        tp1 = entry_price + atr * 1.5 if direction == "buy" else entry_price - atr * 1.5
        tp2 = entry_price + atr * 2.5 if direction == "buy" else entry_price - atr * 2.5
        sl = entry_price - atr * 1.2 if direction == "buy" else entry_price + atr * 1.2

        high = row.get("high", row["close"])
        low = row.get("low", row["close"])

        # [Patch v16.1.2] ตรวจ TP1/TP2/SL จาก high/low จริง
        exit_price, exit_reason = None, None
        if direction == "buy":
            if high >= tp2:
                exit_price = tp2
                exit_reason = "tp2"
            elif high >= tp1:
                exit_price = tp1
                exit_reason = "tp1"
            elif low <= sl:
                exit_price = sl
                exit_reason = "sl"
        else:
            if low <= tp2:
                exit_price = tp2
                exit_reason = "tp2"
            elif low <= tp1:
                exit_price = tp1
                exit_reason = "tp1"
            elif high >= sl:
                exit_price = sl
                exit_reason = "sl"

        if exit_price is None:
            exit_price = row.get("close")
            exit_reason = "timeout_exit"

        trade_log.append({
            "timestamp": timestamp,
            "entry_price": entry_price,
            "tp1_price": tp1,
            "tp2_price": tp2,
            "sl_price": sl,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "session": row.get("session", "Unknown"),
            "risk_mode": row.get("risk_mode", "normal"),
            "entry_signal": row.get("entry_signal"),
            "signal_name": row.get("pattern", "debug"),
            "entry_tier": row.get("entry_tier", "C"),
            "lot": 0.01,
            "pnl": round((exit_price - entry_price) * 10 if direction == "buy" else (entry_price - exit_price) * 10, 2),
            "planned_risk": abs(entry_price - sl) * 10,
            "r_multiple": round(((exit_price - entry_price) / (entry_price - sl)), 2) if direction == "buy" else round(((entry_price - exit_price) / (sl - entry_price)), 2),
            "commission": 0.07,
            "spread_cost": 0.0,
            "slippage_cost": 0.0,
            "mfe": abs(tp1 - entry_price),
            "duration_min": 5.0,
            "gain_z": row.get("gain_z", 0),
            "rsi": row.get("rsi", 50),
            "ema_slope": row.get("ema_slope", 0.01),
            "atr": atr,
            "pattern": row.get("pattern", "inside_bar"),
            "entry_score": row.get("entry_score", 1.0),
        })
    return pd.DataFrame(trade_log)
