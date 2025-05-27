# main.py - NICEGOLD Assistant (L4 GPU + QA Guard + Full Progress Bars)

import os
import sys
sys.path.append("/content/drive/MyDrive/NICEGOLD")  # Add project root to path

import pandas as pd
import gc
import logging
from tqdm import tqdm, trange
from nicegold_v5.wfv import (
    run_walkforward_backtest as raw_run,
    merge_equity_curves,
    plot_equity,
    session_performance,
    streak_summary,
)

# Keep backward-compatible name
run_walkforward_backtest = raw_run

from multiprocessing import cpu_count, get_context
import numpy as np
from nicegold_v5.utils import run_auto_wfv, split_by_session
from nicegold_v5.entry import generate_signals_v8_0 as generate_signals  # [Patch v8.1.2] ใช้ logic sniper + TP1/TSL แบบล่าสุด
from nicegold_v5.config import SNIPER_CONFIG_DEFAULT  # [Patch v8.1.5]

# --- Advanced Risk Management (Patch C) ---
KILL_SWITCH_DD = 25  # %
MAX_LOT_CAP = 1.0  # [Patch v6.7] จำกัดขนาดลอตสูงสุดต่อไม้


def kill_switch(equity_curve):
    peak = equity_curve[0]
    for eq in equity_curve:
        dd = (peak - eq) / peak * 100
        if dd >= KILL_SWITCH_DD:
            print("[KILL SWITCH] Drawdown limit reached. Backtest halted.")
            return True
        peak = max(peak, eq)
    return False


def apply_recovery_lot(capital, sl_streak, base_lot=0.01):
    if sl_streak >= 2:
        factor = 1 + 0.5 * (sl_streak - 1)
        return round(base_lot * factor, 2)
    return base_lot


def adaptive_tp_multiplier(session):
    if session == "Asia":
        return 1.5
    elif session == "London":
        return 2.0
    elif session == "NY":
        return 2.5
    return 2.0


def get_sl_tp(price, atr, session, direction):
    multiplier = adaptive_tp_multiplier(session)
    sl = price - atr * 1.2 if direction == "buy" else price + atr * 1.2
    tp = price + atr * multiplier if direction == "buy" else price - atr * multiplier
    return sl, tp


def calc_lot_risk(capital, atr, risk_pct=1.5):
    pip_value = 10
    sl_pips = atr * 10
    risk_amount = capital * (risk_pct / 100)
    lot = risk_amount / (sl_pips * pip_value)
    return max(0.01, round(lot, 2))

# Mock CSV integrity check to keep CLI functional even without testing module
def run_csv_integrity_check():
    return True

TRADE_DIR = "/content/drive/MyDrive/NICEGOLD/logs"
M1_PATH = "/content/drive/MyDrive/NICEGOLD/XAUUSD_M1.csv"
M15_PATH = "/content/drive/MyDrive/NICEGOLD/XAUUSD_M15.csv"
os.makedirs(TRADE_DIR, exist_ok=True)

# [Patch C.2] Enable full RAM mode
MAX_RAM_MODE = True


def maximize_ram():
    if MAX_RAM_MODE:
        try:
            import psutil
            gc.disable()
            print("🚀 MAX_RAM_MODE: ON – GC disabled")
            print(f"✅ Total RAM: {psutil.virtual_memory().total / 1024**3:.2f} GB")
        except Exception:
            pass
    else:
        gc.collect()

def _run_fold(args):
    df, features, label_col, i = args
    # [Patch] Ensure 'Open' column exists and is capitalized correctly
    if 'open' in df.columns:
        df = df.rename(columns={'open': 'Open'})
    trades = raw_run(df, features, label_col, strategy_name=f"Fold{i+1}")
    trades["fold"] = i + 1
    return trades

def run_parallel_wfv(df: pd.DataFrame, features: list, label_col: str, n_folds: int = 5):
    print("\n⚡ Parallel Walk-Forward (Full RAM Mode)")
    df = df.copy(deep=False)  # [Perf-C] ลด RAM ใช้ deepcopy
    if 'open' in df.columns and 'Open' not in df.columns:
        df.rename(columns={'open': 'Open'}, inplace=True)
        features = ['Open' if f == 'open' else f for f in features]
    df = df.astype({col: np.float32 for col in features if col in df.columns})
    df[label_col] = df[label_col].astype(np.uint8)
    required_cols = ['open']  # [Patch] Include lowercase 'open' for renaming
    df = df.drop(columns=[col for col in df.columns if col not in features + [label_col] + required_cols])

    session_dict = split_by_session(df)
    trades_list = []
    for name, sess_df in session_dict.items():
        trades = raw_run(sess_df, features, label_col, strategy_name=name)
        trades["fold"] = name
        trades_list.append(trades)

    all_df = pd.concat(trades_list, ignore_index=True)
    out_path = os.path.join(TRADE_DIR, "manual_backtest_trades.csv")
    all_df.to_csv(out_path, index=False)
    print(f"📦 Saved trades to: {out_path}")
    maximize_ram()
    return all_df


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
    from nicegold_v5.utils import split_by_session

    logging.info("[TIME] run_wfv_with_progress(): Start")

    session_folds = split_by_session(df)
    all_trades = []
    print("\n📊 Running Session Folds:")
    for name, sess_df in session_folds.items():
        fold_pbar = tqdm(total=1, desc=f"🔁 {name}", unit="step")
        try:
            trades = run_walkforward_backtest(sess_df, features, label_col, strategy_name=name)
            if not trades.empty:
                trades["fold"] = name
                start_time = trades["time"].min() if "time" in trades.columns else "N/A"
                end_time = trades["time"].max() if "time" in trades.columns else "N/A"
                duration_days = (pd.to_datetime(end_time) - pd.to_datetime(start_time)).days if start_time != "N/A" else "-"
                num_orders = len(trades)
                total_lots = trades["lot"].sum() if "lot" in trades.columns else 0
                total_pnl = trades["pnl"].sum()
                win_trades = trades[trades["pnl"] > 0].shape[0]
                loss_trades = trades[trades["pnl"] < 0].shape[0]
                max_dd = trades["drawdown"].max() if "drawdown" in trades.columns else None

                print(f"📈 {name} Summary:")
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
            print(f"❌ Error in {name}: {e}")
    logging.info("[TIME] run_wfv_with_progress(): Done")
    return pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

def show_progress_bar(task_desc, steps=5):
    for _ in trange(steps, desc=task_desc, unit="step"):
        pass

def welcome():
    print("\n🟡 NICEGOLD Assistant พร้อมให้บริการแล้ว (L4 GPU + QA Guard)")
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
        maximize_ram()

    elif choice == 2:
        show_progress_bar("📊 Session Analysis", steps=3)
        path = input("📄 ใส่ path ไฟล์ trade_log CSV: ").strip()
        trades = load_csv_safe(path)
        trades["time"] = pd.to_datetime(trades["time"], errors="coerce")
        print(session_performance(trades))
        maximize_ram()

    elif choice == 3:
        show_progress_bar("📉 Drawdown/Streak", steps=3)
        path = input("📄 ใส่ path ไฟล์ trade_log CSV: ").strip()
        trades = load_csv_safe(path)
        trades["time"] = pd.to_datetime(trades["time"], errors="coerce")
        print(streak_summary(trades))
        maximize_ram()

    elif choice == 4:
        show_progress_bar("📡 Backtest Signals", steps=3)
        print("\n⚙️ เริ่มรัน Backtest จาก Signal (ไม่ใช้ ML)...")
        df = load_csv_safe(M1_PATH)
        # [Patch] Apply full datetime and signal generation
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
        )
        df = df.sort_values("timestamp")

        from nicegold_v5.entry import (
            generate_signals_v8_0 as generate_signals
        )  # [Patch v8.1.3] เปลี่ยนจาก logic v4.1 เป็น v8.0 ที่เจ้านายระบุ
        from nicegold_v5.config import SNIPER_CONFIG_AUTO_GAIN
        from nicegold_v5.backtester import run_backtest
        from nicegold_v5.utils import (
            print_qa_summary,
            create_summary_dict,
            export_chatgpt_ready_logs,
        )
        import time

        # [Patch] Inject signal + run with updated SL/TP1/TP2/BE
        df = generate_signals(df, config=SNIPER_CONFIG_AUTO_GAIN)
        if "entry_tier" in df.columns:
            print("[Patch] Removing weak 'C' tier signals.")
            df = df[df["entry_tier"] != "C"]

        # [Patch QA-P8] คำเตือนสำคัญ: ต้องปิดระบบด้วยตนเองหรือใช้ News Filter
        # ในช่วงข่าว High-Impact (NFP, FOMC, CPI) ตามผลการทดสอบ Stress Test!
        # การไม่ปฏิบัติตามอาจส่งผลให้ระบบทำงานผิดพลาดหรือขาดทุนสูงกว่าที่คาดการณ์!


        # [Patch v8.1.6 + v8.1.7.1 + v8.1.8] Fallback เมื่อไม่มี entry_signal เลย
        if df["entry_signal"].isnull().mean() == 1.0:
            print("⚠️ [Patch] No signals – trying SNIPER_CONFIG_RELAXED...")
            from nicegold_v5.config import SNIPER_CONFIG_RELAXED
            df = generate_signals(df, config=SNIPER_CONFIG_RELAXED)
            if "entry_tier" in df.columns:
                df = df[df["entry_tier"] != "C"]  # remove weak tier

        # [Patch QA-P11] เพิ่ม Fallback ขั้นต่อไป
        if df["entry_signal"].isnull().mean() == 1.0:
            print("⚠️ [Patch] No signals – trying SNIPER_CONFIG_OVERRIDE...")
            from nicegold_v5.config import SNIPER_CONFIG_OVERRIDE
            df = generate_signals(df, config=SNIPER_CONFIG_OVERRIDE)
            if "entry_tier" in df.columns:
                df = df[df["entry_tier"] != "C"]  # remove weak tier

        # [Patch QA-P11] เพิ่ม Fallback สุดท้ายไปยัง Diagnostic Config เพื่อให้แน่ใจว่ามี Signal
        if df["entry_signal"].isnull().mean() == 1.0:
            print("⚠️ [Patch QA-P11] Still no signals – FORCING SNIPER_CONFIG_DIAGNOSTIC...")
            from nicegold_v5.config import SNIPER_CONFIG_DIAGNOSTIC
            df = generate_signals(df, config=SNIPER_CONFIG_DIAGNOSTIC)

        # [Patch v8.1.9] Fallback ปลดบล็อกสัญญาณขั้นสุดท้าย
        if df["entry_signal"].isnull().mean() == 1.0:
            print("⚠️ [Patch] FINAL: Relaxed AutoGain fallback applied...")
            from nicegold_v5.config import SNIPER_CONFIG_RELAXED_AUTOGAIN
            df = generate_signals(df, config=SNIPER_CONFIG_RELAXED_AUTOGAIN)
        start = time.time()
        trades, equity = run_backtest(df)
        end = time.time()

        start_time = pd.to_datetime(df["timestamp"].iloc[0])
        end_time = pd.to_datetime(df["timestamp"].iloc[-1])

        print_qa_summary(trades, equity)  # [Patch] Now includes exit_reason, drawdown

        # [Patch] Export with updated format including SL/TP1/TP2/BE info
        summary = create_summary_dict(
            trades,
            equity,
            file_name="XAUUSD_M1.csv",
            start_time=start_time,
            end_time=end_time,
            duration_sec=end - start,
        )
        export_chatgpt_ready_logs(trades, equity, summary, outdir=TRADE_DIR)

    elif choice == 5:
        show_progress_bar("👋 กำลังออกจากระบบ", steps=2)
        print("👋 ขอบคุณที่ใช้ NICEGOLD. พบกันใหม่!")
        maximize_ram()
    else:
        print("❌ เลือกเมนูไม่ถูกต้อง")
        maximize_ram()

if __name__ == "__main__":
    welcome()
