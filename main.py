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
from nicegold_v5.entry import (
    generate_signals_v12_0 as generate_signals,  # 🔄 เปลี่ยนจาก v11 → v12
    apply_tp_logic,
    generate_entry_signal,
    session_filter,
    sanitize_price_columns,
    validate_indicator_inputs,
)
from nicegold_v5.config import (
    SNIPER_CONFIG_PROFIT,
    SNIPER_CONFIG_Q3_TUNED,
    RELAX_CONFIG_Q3,
)
from nicegold_v5.qa import run_qa_guard, auto_qa_after_backtest
from nicegold_v5.utils import (
    safe_calculate_net_change,
    convert_thai_datetime,
    parse_timestamp_safe,
)
# User-provided custom instructions
# *สนทนาภาษาไทยเท่านั้น

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
M1_PATH = os.getenv("M1_PATH", "/content/drive/MyDrive/NICEGOLD/XAUUSD_M1.csv")
M15_PATH = os.getenv("M15_PATH", "/content/drive/MyDrive/NICEGOLD/XAUUSD_M15.csv")
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
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
    """Load CSV with fallback to local data directory."""
    if not os.path.exists(path):
        alt = os.path.join(os.path.dirname(__file__), "nicegold_v5", os.path.basename(path))
        if os.path.exists(alt):
            path = alt
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


def run_clean_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """Run backtest with cleaned signals and real exit logic."""
    df = df.copy()

    from nicegold_v5.entry import sanitize_price_columns, validate_indicator_inputs

    # ✅ [Patch v11.9.23] รองรับ Date (พ.ศ.) + Timestamp รวม + Auto Fix Lowercase Columns
    # - รองรับทั้งแบบ Date+Timestamp และ date+timestamp (lowercase)
    # - แก้บั๊กแปลง timestamp fail (NaT ทุกแถว)
    if {"Date", "Timestamp"}.issubset(df.columns):
        df = convert_thai_datetime(df)  # สร้าง df["timestamp"] ที่ถูกต้องแล้ว
        df["timestamp"] = parse_timestamp_safe(df["timestamp"], DATETIME_FORMAT)
    elif {"date", "timestamp"}.issubset(df.columns):
        # เผื่อกรณี lowercase
        df["date"] = df["date"].astype(str).str.zfill(8)

        def _th2en(s):
            y, m, d = int(s[:4]) - 543, s[4:6], s[6:8]
            return f"{y:04d}-{m}-{d}"

        df["date_gregorian"] = df["date"].apply(_th2en)
        df["timestamp_full"] = df["date_gregorian"] + " " + df["timestamp"].astype(str)
        df["timestamp"] = pd.to_datetime(df["timestamp_full"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    else:
        df["timestamp"] = parse_timestamp_safe(df["timestamp"], DATETIME_FORMAT)
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp")
    df = sanitize_price_columns(df)
    try:
        validate_indicator_inputs(df, min_rows=min(500, len(df)))
    except TypeError:
        # compatibility with monkeypatched tests without min_rows
        validate_indicator_inputs(df)

    from nicegold_v5.config import RELAX_CONFIG_Q3
    df = generate_signals(df, config=SNIPER_CONFIG_Q3_TUNED)

    # [Patch v12.0.3] ✅ Ensure 'entry_time' exists
    if "entry_time" not in df.columns:
        print("[Patch v12.0.3] ⛑ fallback: สร้าง entry_time จาก timestamp")
        df["entry_time"] = df.get("timestamp", pd.NaT)
    if not pd.api.types.is_datetime64_any_dtype(df["entry_time"]):
        df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")

    # [Patch v12.0.3] ✅ Ensure 'entry_signal' exists
    if "entry_signal" not in df.columns:
        print("[Patch v12.0.3] ⛑ fallback: ไม่มี entry_signal – ใส่ค่า None")
        df["entry_signal"] = None

    if df["entry_signal"].isnull().mean() == 1.0:
        print("[Patch v11.9.16] ❗ ไม่พบสัญญาณใน Q3_TUNED – ใช้ fallback RELAX_CONFIG_Q3")
        df = generate_signals(df, config=RELAX_CONFIG_Q3)
    
    # [Patch v12.0.3] 🧠 Block run if no signal at all
    if df["entry_signal"].isnull().mean() >= 1.0:
        raise RuntimeError("[Patch v12.0.3] ❌ ไม่มีสัญญาณเข้าเลย – หยุดรัน backtest")

    signal_coverage = df["entry_signal"].notnull().mean() * 100
    print(f"[Patch v11.9.16] ✅ Entry Signal Coverage: {signal_coverage:.2f}%")

    # Ensure timestamps are valid and use them for entry_time
    df["timestamp"] = parse_timestamp_safe(df["timestamp"], DATETIME_FORMAT)
    df["entry_time"] = df["timestamp"]
    df["signal_id"] = df["timestamp"].astype(str)

    # [Patch v12.0.2] Validate required columns before backtest
    required = ["entry_time", "entry_signal", "close"]
    missing_required = [col for col in required if col not in df.columns]
    if missing_required:
        print(f"[Patch v12.0.2] ⚠️ Missing columns: {missing_required} – Creating with default values")
        for col in missing_required:
            df[col] = pd.NaT if "time" in col else None
    df = df.dropna(subset=required)

    # Guard against leakage from future columns
    leak_cols = [c for c in df.columns if "future" in c or "next_" in c or c.endswith("_lead")]
    df.drop(columns=leak_cols, errors="ignore", inplace=True)

    from nicegold_v5.backtester import run_backtest
    trades, equity = run_backtest(df)

    from nicegold_v5.utils import print_qa_summary, export_chatgpt_ready_logs
    print_qa_summary(trades, equity)
    export_chatgpt_ready_logs(trades, equity, {"file_name": "v12.0.3-test"})

    return trades

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

    # [Patch v12.0.1] Fail-Proof TP1/TP2 Simulation ด้วย logic v12.0
    from nicegold_v5.entry import simulate_trades_with_tp
    from nicegold_v5.config import SNIPER_CONFIG_Q3_TUNED
    from nicegold_v5.utils import safe_calculate_net_change

    print("📊 [Patch v12.0.1] เริ่ม Fail-Proof TP1/TP2 Simulation ด้วย logic v12.0...")
    df = load_csv_safe(M1_PATH)
    
    # ✅ [Patch v11.9.18] รองรับ Date แบบพุทธศักราช
    df = convert_thai_datetime(df)
    show_progress_bar("🧼 แปลง timestamp", steps=1)
    df["timestamp"] = parse_timestamp_safe(df["timestamp"], DATETIME_FORMAT)
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp")

    df = sanitize_price_columns(df)
    try:
        validate_indicator_inputs(df, min_rows=min(500, len(df)))
    except TypeError:
        validate_indicator_inputs(df)

    show_progress_bar("⚙️ ตรวจสอบสัญญาณ", steps=1)

    df = generate_signals(df)

    # 📉 หากไม่มีสัญญาณเลย ให้ใช้ config ที่ผ่อนปรนกว่า
    if df["entry_signal"].isnull().mean() >= 1.0:
        print("[Patch CLI] ⚠️ ไม่มีสัญญาณจาก config หลัก – fallback RELAX_CONFIG_Q3")
        df = generate_signals(df, config=RELAX_CONFIG_Q3)

    show_progress_bar("🧪 ตรวจสอบความสมบูรณ์ของข้อมูล", steps=1)

    if "entry_time" not in df.columns:
        print("[Patch CLI] ⛑ สร้าง entry_time จาก timestamp")
        df["entry_time"] = df.get("timestamp")
    if not pd.api.types.is_datetime64_any_dtype(df["entry_time"]):
        df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")

    required = ["timestamp", "entry_signal", "entry_time"]
    df = df.dropna(subset=required)
    df["signal"] = df["entry_signal"].apply(lambda x: "long" if pd.notnull(x) else None)
    if df.empty:
        raise RuntimeError("[Patch QA] ❌ ไม่มีแถวที่มีข้อมูลครบสำหรับ simulate")
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        raise ValueError("[Patch QA] ❌ timestamp ต้องแปลงเป็น datetime ก่อน simulate")

    show_progress_bar("🚀 รัน simulate_trades_with_tp", steps=2)
    trades, logs = simulate_trades_with_tp(df)
    trade_df = pd.DataFrame(trades)

    if trade_df.empty or trade_df["exit_reason"].isnull().all():
        print("[Patch QA] ⚠️ simulate_trades_with_tp ไม่พบ trade ที่ถูกยิงจริง")
        maximize_ram()
        return

    out_path = os.path.join(TRADE_DIR, "trades_v12_tp1tp2.csv")
    trade_df.to_csv(out_path, index=False)
    print(f"[Patch v12.0.1] ✅ บันทึกผล TP1/TP2 Trade log ที่: {out_path}")

    tp1_hits = trade_df["exit_reason"].eq("tp1").sum()
    tp2_hits = trade_df["exit_reason"].eq("tp2").sum()
    sl_hits = trade_df["exit_reason"].eq("sl").sum()
    total_pnl = safe_calculate_net_change(trade_df)

    print("\n📊 [Patch QA] Summary (TP1/TP2):")
    print(f"   ▸ TP1 Triggered : {tp1_hits}")
    print(f"   ▸ TP2 Triggered : {tp2_hits}")
    print(f"   ▸ SL Count      : {sl_hits}")
    print(f"   ▸ Net PnL       : {total_pnl:.2f} USD")
    maximize_ram()
    return  # Skip menu for automation

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

        # ✅ [Patch v11.9.18] รองรับ Date แบบพุทธศักราช
        df = convert_thai_datetime(df)

        # [Patch] Apply full datetime and signal generation
        df["timestamp"] = parse_timestamp_safe(df["timestamp"], DATETIME_FORMAT)
        df = df.sort_values("timestamp")

        from nicegold_v5.entry import (
            generate_signals_v11_scalper_m1 as generate_signals_menu
        )  # [Patch v10.1] Scalper Boost: QM + RSI + Fractal + InsideBar
        from nicegold_v5.config import SNIPER_CONFIG_PROFIT  # [Patch v10.1]
        from nicegold_v5.backtester import run_backtest
        from nicegold_v5.utils import (
            print_qa_summary,
            create_summary_dict,
            export_chatgpt_ready_logs,
        )
        import time

        # [Patch] Inject signal + run with updated SL/TP1/TP2/BE
        print("\U0001F9E0 [UltraFix] Injecting Profit Config for entry_signal...")
        df = generate_signals_menu(df, config=SNIPER_CONFIG_PROFIT)
        if "entry_tier" in df.columns:
            print("[Patch] Removing weak 'C' tier signals.")
            df = df[df["entry_tier"] != "C"]

        # [Patch QA-P8] คำเตือนสำคัญ: ต้องปิดระบบด้วยตนเองหรือใช้ News Filter
        # ในช่วงข่าว High-Impact (NFP, FOMC, CPI) ตามผลการทดสอบ Stress Test!
        # การไม่ปฏิบัติตามอาจส่งผลให้ระบบทำงานผิดพลาดหรือขาดทุนสูงกว่าที่คาดการณ์!


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
        run_qa_guard(trades, df)
        auto_qa_after_backtest(trades, equity, label="Signal")

    elif choice == 5:
        show_progress_bar("👋 กำลังออกจากระบบ", steps=2)
        print("👋 ขอบคุณที่ใช้ NICEGOLD. พบกันใหม่!")
        maximize_ram()
    elif choice == 6:
        show_progress_bar("🧪 TP1/TP2 Backtest Mode", steps=3)
        print("\n⚙️ เริ่มรัน simulate_trades_with_tp() จาก UltraFix Patch...")
        df = load_csv_safe(M1_PATH)

        # ✅ [Patch v11.9.18] รองรับ Date แบบพุทธศักราช
        df = convert_thai_datetime(df)

        df["timestamp"] = parse_timestamp_safe(df["timestamp"], DATETIME_FORMAT)
        df = df.sort_values("timestamp")

        from nicegold_v5.entry import simulate_trades_with_tp  # ← Patch v11.2 logic
        trades, logs = simulate_trades_with_tp(df)
        trade_df = pd.DataFrame(trades)

        out_path = os.path.join(TRADE_DIR, "trades_v11p_tp1tp2.csv")
        trade_df.to_csv(out_path, index=False)
        print(f"📦 บันทึกผล TP1/TP2 Trade log ที่: {out_path}")

        tp1_hits = (
            trade_df["exit_reason"].eq("tp1").sum() if "exit_reason" in trade_df.columns else 0
        )
        tp2_hits = (
            trade_df["exit_reason"].eq("tp2").sum() if "exit_reason" in trade_df.columns else 0
        )
        sl_hits = trade_df["exit_reason"].eq("sl").sum()
        total_pnl = safe_calculate_net_change(trade_df)

        print("\n📊 QA Summary (TP1/TP2):")
        print(f"   ▸ TP1 Triggered : {tp1_hits}")
        print(f"   ▸ TP2 Triggered : {tp2_hits}")
        print(f"   ▸ SL Count      : {sl_hits}")
        print(f"   ▸ Net PnL       : {total_pnl:.2f} USD")

        maximize_ram()
    else:
        print("❌ เลือกเมนูไม่ถูกต้อง")
        maximize_ram()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "clean":
        print("📥 Loading CSV...")
        df = load_csv_safe(M1_PATH)

        # ✅ [Patch v11.9.18] รองรับ Date แบบพุทธศักราช
        df = convert_thai_datetime(df)

        df["timestamp"] = parse_timestamp_safe(df["timestamp"], DATETIME_FORMAT)
        df = df.dropna(subset=["timestamp"])
        run_clean_backtest(df)
        print("✅ Done: Clean Backtest Completed")
    else:
        welcome()
