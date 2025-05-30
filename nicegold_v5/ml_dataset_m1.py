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

    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.dropna(subset=["timestamp", "high", "low", "close", "volume"])
    df = df.sort_values("timestamp").reset_index(drop=True)

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
    if not os.path.exists(trade_log_path):
        raise FileNotFoundError("❌ Trade log not found at logs/trades_v12_tp1tp2.csv")

    trades = pd.read_csv(trade_log_path)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])

    df["tp2_hit"] = 0
    df.loc[df["timestamp"].isin(trades[trades["exit_reason"] == "tp2"]["entry_time"]), "tp2_hit"] = 1
    df = df.dropna().reset_index(drop=True)
    df.to_csv(out_path, index=False)
    print(f"✅ Saved ML dataset to: {out_path}")
