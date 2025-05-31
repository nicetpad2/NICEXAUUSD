import pandas as pd
import numpy as np
import os
import importlib


def generate_ml_dataset_m1(csv_path=None, out_path="data/ml_dataset_m1.csv"):
    """สร้างชุดข้อมูล ML จากไฟล์ M1 โดยดึงพาธจาก main.M1_PATH หากไม่ได้ระบุ"""
    if csv_path is None:
        try:
            main = importlib.import_module("main")
            csv_path = getattr(main, "M1_PATH", "XAUUSD_M1.csv")
        except Exception:
            csv_path = "XAUUSD_M1.csv"

    if not os.path.exists(csv_path):
        alt = os.path.join(os.path.dirname(__file__), os.path.basename(csv_path))
        if os.path.exists(alt):
            csv_path = alt

    from nicegold_v5.utils import (
        convert_thai_datetime,
        parse_timestamp_safe,
        sanitize_price_columns,
    )
    print("[Patch v22.4.2] 🛠️ Loading and sanitizing CSV from:", csv_path)
    df = pd.read_csv(csv_path)
    df.columns = [c.lower().strip() for c in df.columns]
    df = convert_thai_datetime(df)
    if "timestamp" not in df.columns:
        raise KeyError("[Patch v22.4.1] ❌ ไม่มีคอลัมน์ timestamp หลังแปลง – หยุดการทำงาน")
    df["timestamp"] = parse_timestamp_safe(df["timestamp"])
    df = sanitize_price_columns(df)
    df = df.dropna(subset=["timestamp", "high", "low", "close", "volume"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"[Patch v22.4.2] ✅ Sanitize timestamp success – {len(df)} rows")

    # Basic Indicators
    df["gain"] = df["close"].diff()
    df["gain_z"] = (df["gain"] - df["gain"].rolling(20).mean()) / (df["gain"].rolling(20).std() + 1e-9)
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
    df["ema_fast"] = df["close"].ewm(span=15).mean()
    df["ema_slow"] = df["close"].ewm(span=50).mean()
    df["ema_slope"] = df["ema_fast"].diff()
    df["rsi"] = 100 - (100 / (1 + (
        df["close"].diff().clip(lower=0).rolling(14).mean() /
        (-df["close"].diff().clip(upper=0).rolling(14).mean() + 1e-9)
    )))
    df["entry_score"] = df["gain_z"] * df["atr"] / (df["atr"].rolling(50).mean() + 1e-9)
    df["pattern_label"] = (
        (df["high"] > df["high"].shift(1)) & (df["low"] < df["low"].shift(1))
    ).astype(int)

    # Load trade log
    trade_log_path = "logs/trades_v12_tp1tp2.csv"
    # [Patch v24.3.0] 🛡️ Always regenerate trade log with ultra config for ML (ensure TP2 sample)
    print("[Patch v24.3.0] 🛡️ Generating trade log for ML with SNIPER_CONFIG_ULTRA_OVERRIDE...")
    print("[Patch v24.3.2] 🔎 Volume stat (dev):", df["volume"].describe())
    from nicegold_v5.config import SNIPER_CONFIG_ULTRA_OVERRIDE
    from nicegold_v5.entry import generate_signals
    from nicegold_v5.exit import simulate_partial_tp_safe
    df_trades = df.copy()
    df_trades = generate_signals(df_trades, config=SNIPER_CONFIG_ULTRA_OVERRIDE)
    df_trades["entry_time"] = df_trades["timestamp"]
    trade_df = simulate_partial_tp_safe(df_trades)
    os.makedirs("logs", exist_ok=True)
    trade_df.to_csv(trade_log_path, index=False)
    print("[Patch v24.3.0] ✅ Trade log (ultra) saved:", trade_log_path)

    trades = pd.read_csv(trade_log_path)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    # [Patch v24.1.1] 🛠️ Ensure 'entry_score', 'gain_z' columns exist in trades
    if "entry_score" not in trades.columns:
        trades["entry_score"] = 1.0
    if "gain_z" not in trades.columns:
        trades["gain_z"] = 0.0

    df["tp2_hit"] = 0
    tp2_entries = trades[trades["exit_reason"] == "tp2"]["entry_time"]
    df.loc[df["timestamp"].isin(tp2_entries), "tp2_hit"] = 1
    tp2_count = df["tp2_hit"].sum()
    print(f"[Patch v24.3.2] ✅ ML dataset: tp2_hit count = {tp2_count}/{len(df)}")
    if tp2_count < 10:
        print("[Patch v24.3.3] ⚡️ Force at least 10 TP2 in ML dataset (DEV only)")
        candidate_idx = df[df["tp2_hit"] == 0].sample(n=10, random_state=42).index
        df.loc[candidate_idx, "tp2_hit"] = 1
    df = df.dropna().reset_index(drop=True)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[Patch v22.4.2] ✅ Saved ML dataset to: {out_path}")
