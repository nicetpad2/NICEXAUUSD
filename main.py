# main.py - NICEGOLD Assistant (L4 GPU + QA Guard + Full Progress Bars)

import os
import sys
sys.path.append("/content/drive/MyDrive/NICEGOLD")  # Add project root to path

import pandas as pd
import gc
import psutil
from tqdm import tqdm, trange
from nicegold_v5.wfv import run_walkforward_backtest, merge_equity_curves, plot_equity, session_performance, streak_summary
from nicegold_v5.utils import run_auto_wfv
from nicegold_v5.entry import generate_signals
from nicegold_v5.run_tests import run_csv_integrity_check

TRADE_DIR = "/content/drive/MyDrive/NICEGOLD/logs"
M1_PATH = "/content/drive/MyDrive/NICEGOLD/XAUUSD_M1.csv"
M15_PATH = "/content/drive/MyDrive/NICEGOLD/XAUUSD_M15.csv"
os.makedirs(TRADE_DIR, exist_ok=True)

fold_pbar.update(1)
maximize_ram()
    gc.collect()
    ram = psutil.virtual_memory()
    print(f"🚀 Using RAM: {ram.percent:.1f}% | Available: {ram.available / 1024**3:.2f} GB")

def load_csv_safe(path, lowercase=True):
    try:
        with tqdm(total=1, desc=f"📥 Loading {os.path.basename(path)}") as pbar:
            df = pd.read_csv(path, engine="python", on_bad_lines="skip")
            if lowercase:
                df.columns = [c.lower().strip() for c in df.columns]
            pbar.update(1)
        print(f"✅ Loaded {len(df):,} rows from {path}")
        return df
    except Exception as e:
        print(f"❌ Failed to load {path}: {e}")
        raise

def run_wfv_with_progress(df, features, label_col):
    from sklearn.model_selection import TimeSeriesSplit
    import numpy as np

    splits = list(TimeSeriesSplit(n_splits=5).split(df))
    all_trades = []
    print("
📊 Running Walk-Forward Folds:")
    for i, (train_idx, test_idx) in enumerate(splits):
        fold_pbar = tqdm(total=1, desc=f"🔁 Fold {i+1}/5", unit="step")
        try:
            fold_label = f"Fold{i+1}"
            df_train = df.iloc[train_idx].copy()
            df_test = df.iloc[test_idx].copy()
            print(f"🔄 {fold_label}: Train {df_train.shape[0]} rows | Test {df_test.shape[0]} rows")
            trades = run_walkforward_backtest(df_test, features, label_col, strategy_name=fold_label)
            if not trades.empty:
                trades["fold"] = fold_label
                start_time = trades["time"].min() if "time" in trades.columns else "N/A"
                end_time = trades["time"].max() if "time" in trades.columns else "N/A"
                duration_days = (pd.to_datetime(end_time) - pd.to_datetime(start_time)).days if start_time != "N/A" else "-"
                num_orders = len(trades)
                total_lots = trades["lot"].sum() if "lot" in trades.columns else 0
                total_pnl = trades["pnl"].sum()
                win_trades = trades[trades["pnl"] > 0].shape[0]
                loss_trades = trades[trades["pnl"] < 0].shape[0]
                max_dd = trades["drawdown"].max() if "drawdown" in trades.columns else None

                print(f"📈 {fold_label} Summary:")
                print(f"    ▸ Orders     : {num_orders}")
                print(f"    ▸ Total Lots : {total_lots:.2f}")
                print(f"    ▸ Win Trades : {win_trades} | Loss Trades : {loss_trades}")
                print(f"    ▸ Total PnL  : {total_pnl:.2f} USD")
                print(f"    ▸ Duration   : {duration_days} days")
                print(f"    ▸ Max Drawdown: {max_dd:.2%}" if max_dd is not None else "")
            all_trades.append(trades)
            fold_pbar.update(1)
            maximize_ram()
        except Exception as e:
            print(f"❌ Error in {fold_label}: {e}")
    return pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

def show_progress_bar(task_desc, steps=5):
    for _ in trange(steps, desc=task_desc, unit="step"):
        pass

def welcome():
    print("\n🟡 NICEGOLD Assistant พร้อมให้บริการแล้ว (L4 GPU + QA Guard)")
    fold_pbar.update(1)
            maximize_ram()

    show_progress_bar("📊 ตรวจ CSV", steps=2)
    if not run_csv_integrity_check():
        print("❌ ยกเลิกการทำงาน: ตรวจพบข้อผิดพลาดในไฟล์ข้อมูล CSV")
        return

    show_progress_bar("📡 เตรียมระบบ", steps=2)
    print("\n📌 เลือกเมนูที่ต้องการ:")
    print("  1. รัน Walk-Forward Strategy (ML Based)")
    print("  2. วิเคราะห์ Session Performance")
    print("  3. สรุป Drawdown & Win/Loss Streak")
    print("  4. รัน Backtest จาก Signal (Non-ML)")
    print("  5. ออกจากระบบ")

    try:
        choice = int(input("\n🔧 เลือกเมนู [1-5]: "))
    except:
        print("❌ ต้องใส่เป็นตัวเลข 1–5")
        return

    if choice == 1:
        print("\n🚀 เริ่มรัน Walk-Forward ML Strategy...")
        df = pd.read_csv(M15_PATH, parse_dates=["timestamp"], engine="python", on_bad_lines="skip")
        show_progress_bar("🚧 เตรียมฟีเจอร์", steps=5)
        df.set_index("timestamp", inplace=True)
        df["EMA_50"] = df["Close"].ewm(span=50).mean()
        df["RSI_14"] = df["Close"].rolling(14).apply(lambda x: 100 - (100 / (1 + ((x.diff().clip(lower=0).mean()) / (-x.diff().clip(upper=0).mean() + 1e-9)))), raw=False)
        df["ATR_14"] = (df["High"] - df["Low"]).rolling(14).mean()
        df["ATR_14_MA50"] = df["ATR_14"].rolling(50).mean()
        df["EMA_50_slope"] = df["EMA_50"].diff()
        df["target"] = (df["Close"].shift(-10) > df["Close"]).astype(int)
        features = ["EMA_50", "RSI_14", "ATR_14", "ATR_14_MA50", "EMA_50_slope"]
        trades_df = run_wfv_with_progress(df, features, "target")
        df_merged = merge_equity_curves(trades_df)
        plot_equity(df_merged)
        out_path = os.path.join(TRADE_DIR, "merged_trades.csv")
        trades_df.to_csv(out_path, index=False)
        print(f"📦 บันทึก Trade log ที่: {out_path}")
        fold_pbar.update(1)
            maximize_ram()

    elif choice == 2:
        show_progress_bar("📊 Session Analysis", steps=3)
        path = input("📄 ใส่ path ไฟล์ trade_log CSV: ").strip()
        trades = load_csv_safe(path)
        trades["time"] = pd.to_datetime(trades["time"], errors="coerce")
        print(session_performance(trades))
        fold_pbar.update(1)
            maximize_ram()

    elif choice == 3:
        show_progress_bar("📉 Drawdown/Streak", steps=3)
        path = input("📄 ใส่ path ไฟล์ trade_log CSV: ").strip()
        trades = load_csv_safe(path)
        trades["time"] = pd.to_datetime(trades["time"], errors="coerce")
        print(streak_summary(trades))
        fold_pbar.update(1)
            maximize_ram()

    elif choice == 4:
        show_progress_bar("📡 Backtest Signals", steps=3)
        print("\n⚙️ เริ่มรัน Backtest จาก Signal (ไม่ใช้ ML)...")
        df = load_csv_safe(M1_PATH)
        if {"date", "timestamp"}.issubset(df.columns):
            def convert_thai_date(date_int):
                try:
                    y = int(str(date_int)[:4]) - 543
                    m = str(date_int)[4:6]
                    d = str(date_int)[6:8]
                    return f"{y}-{m}-{d}"
                except:
                    return "1900-01-01"
            df["timestamp"] = pd.to_datetime(
                df["date"].apply(convert_thai_date) + " " + df["timestamp"].astype(str),
                format="%Y-%m-%d %H:%M:%S",
                errors="coerce"
            )
            df.dropna(subset=["timestamp"], inplace=True)
        else:
            raise ValueError("❌ ไม่พบคอลัมน์ date/timestamp สำหรับสร้าง timestamp")
        df.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close"}, inplace=True)
        df = generate_signals(df)
        run_auto_wfv(df, outdir=TRADE_DIR)
        fold_pbar.update(1)
            maximize_ram()

    elif choice == 5:
        show_progress_bar("👋 กำลังออกจากระบบ", steps=2)
        print("👋 ขอบคุณที่ใช้ NICEGOLD. พบกันใหม่!")
        fold_pbar.update(1)
            maximize_ram()
    else:
        print("❌ เลือกเมนูไม่ถูกต้อง")
        fold_pbar.update(1)
            maximize_ram()

if __name__ == "__main__":
    welcome()
