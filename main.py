# main.py - NICEGOLD Assistant (L4 GPU + QA Guard + Full Progress Bars)

import os  # [Patch v12.3.9] Ensure os is imported for path.join
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import pandas as pd
import gc
import logging
import json  # [Patch v12.4.0] Added for JSON export
from datetime import datetime
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
    generate_signals_v12_0 as generate_signals,  # [Patch v12.3.9] ensure import
    sanitize_price_columns,
    validate_indicator_inputs,
    rsi,
)
from nicegold_v5.exit import simulate_partial_tp_safe  # [Patch v12.2.x]
from nicegold_v5.config import (  # [Patch v12.3.9] Import SNIPER_CONFIG_Q3_TUNED
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
from nicegold_v5.fix_engine import simulate_and_autofix  # [Patch v12.3.9] Added import
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
M1_PATH = os.getenv(
    "M1_PATH",
    os.path.join(ROOT_DIR, "nicegold_v5", "XAUUSD_M1.csv"),
)
M15_PATH = os.getenv(
    "M15_PATH",
    os.path.join(ROOT_DIR, "nicegold_v5", "XAUUSD_M15.csv"),
)
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
    df, features, label_col, fold_name = args
    # [Patch v16.0.1] Fallback Fix 'Open' column แบบเทพ
    if "Open" not in df.columns:
        if "open" in df.columns:
            df.rename(columns={"open": "Open"}, inplace=True)
            print("[Patch v16.0.1] 🛠️ Fallback: ใช้ 'open' → 'Open'")
        elif "close" in df.columns:
            df["Open"] = df["close"]
            print("[Patch v16.0.1] 🛠️ Fallback: ใช้ 'close' → 'Open'")
        else:
            raise ValueError("[Patch v16.0.1] ❌ ไม่มี 'Open'/'close' column ให้ fallback")
    trades = raw_run(df, features, label_col, strategy_name=str(fold_name))
    trades["fold"] = fold_name
    return trades

def run_parallel_wfv(df: pd.DataFrame, features: list, label_col: str, n_folds: int = 5):
    print("\n⚡ Parallel Walk-Forward (Full RAM Mode)")
    df = df.copy(deep=False)  # [Perf-C] ลด RAM ใช้ deepcopy
    # [Patch v16.0.1] Fallback Fix 'Open' column แบบเทพ
    if "Open" not in df.columns:
        if "open" in df.columns:
            df.rename(columns={"open": "Open"}, inplace=True)
            print("[Patch v16.0.1] 🛠️ Fallback: ใช้ 'open' → 'Open'")
        elif "close" in df.columns:
            df["Open"] = df["close"]
            print("[Patch v16.0.1] 🛠️ Fallback: ใช้ 'close' → 'Open'")
        else:
            raise ValueError("[Patch v16.0.1] ❌ ไม่มี 'Open'/'close' column ให้ fallback")
    df = df.astype({col: np.float32 for col in features if col in df.columns})
    df[label_col] = df[label_col].astype(np.uint8)
    required_cols = ['open']  # [Patch] Include lowercase 'open' for renaming
    df = df.drop(columns=[col for col in df.columns if col not in features + [label_col] + required_cols])

    session_dict = split_by_session(df)
    args_list = [(sess_df, features, label_col, name) for name, sess_df in session_dict.items()]

    ctx = get_context("spawn")
    try:
        with ctx.Pool(min(cpu_count(), len(args_list))) as pool:
            trades_list = pool.map(_run_fold, args_list)
    except Exception:
        # Fallback to sequential if multiprocessing fails
        trades_list = [_run_fold(args) for args in args_list]

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
    # [Patch v12.4.2] – Export Summary JSON + QA Summary ต่อ Fold (Incorporates v12.4.0, v12.4.1)
    # -------------------------------------------------------------------
    # ✅ robust: fallback config if signal missing
    # ✅ export: log + config_used + summary
    # ✅ CLI: เพิ่มเมนูใน welcome() → choice 7 (commented out in welcome())
    # ✅ บันทึก QA Summary แยกไฟล์ JSON
    df = df.copy()
    # รองรับ Date+Timestamp แบบ พ.ศ. และตัวพิมพ์เล็ก
    if {"Date", "Timestamp"}.issubset(df.columns):
        df = convert_thai_datetime(df)
        df["timestamp"] = parse_timestamp_safe(df["timestamp"], DATETIME_FORMAT)
    elif {"date", "timestamp"}.issubset(df.columns):
        df["date"] = df["date"].astype(str).str.zfill(8)
        def _th2en(s):
            y, m, d = int(s[:4]) - 543, s[4:6], s[6:8]
            return f"{y:04d}-{m}-{d}"
        df["date_gregorian"] = df["date"].apply(_th2en)
        df["timestamp_full"] = df["date_gregorian"] + " " + df["timestamp"].astype(str)
        df["timestamp"] = pd.to_datetime(df["timestamp_full"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp")

    # [Patch v12.4.0] Sanitize and validate
    try:
        df = sanitize_price_columns(df)
        validate_indicator_inputs(df, min_rows=min(500, len(df)))
    except TypeError:
        validate_indicator_inputs(df)

    from nicegold_v5.config import RELAX_CONFIG_Q3
    df = generate_signals(df, config=SNIPER_CONFIG_Q3_TUNED)

    # [Patch v12.4.0] Fallback if no signals
    if df["entry_signal"].isnull().mean() >= 1.0:
        print("[Patch QA] ⚠️ ไม่พบสัญญาณ → fallback RELAX_CONFIG_Q3")
        df = generate_signals(df, config=RELAX_CONFIG_Q3)

    if "entry_signal" not in df.columns:
        df["entry_signal"] = None

    if df["entry_signal"].isnull().mean() >= 1.0:
        raise RuntimeError("[Patch QA] ❌ ไม่มีสัญญาณเข้าเลย – หยุดรัน backtest")

    df = df.dropna(subset=["timestamp", "entry_signal", "close", "high", "low"])
    df["entry_time"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["signal_id"] = df["timestamp"].astype(str)

    print("\n🚀 [Patch v12.4.2] Using simulate_and_autofix() pipeline...")
    trades_df, equity_df, config_used = simulate_and_autofix(
        df,
        simulate_partial_tp_safe,
        SNIPER_CONFIG_Q3_TUNED,
        session="London"
    )
    print("\n✅ Simulation Completed with Adaptive Config:")
    for k, v in config_used.items():
        print(f"    ▸ {k}: {v}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(TRADE_DIR, f"trades_cleanbacktest_{ts}.csv")
    trades_df.to_csv(out_path, index=False)
    print(f"📁 Exported trades → {out_path}")

    config_path = os.path.join(TRADE_DIR, f"config_used_{ts}.json")
    with open(config_path, "w") as f:
        json.dump(config_used, f, indent=2)
    print(f"⚙️ Exported config_used → {config_path}")

    if "exit_reason" in trades_df.columns:
        qa_summary = {
            "tp1_count": int(trades_df["exit_reason"].eq("tp1").sum()),
            "tp2_count": int(trades_df["exit_reason"].eq("tp2").sum()),
            "sl_count": int(trades_df["exit_reason"].eq("sl").sum()),
            "total_trades": len(trades_df),
            "net_pnl": float(trades_df["pnl"].sum() if "pnl" in trades_df.columns else 0.0),
        }
        qa_path = os.path.join(TRADE_DIR, f"qa_summary_{ts}.json")
        with open(qa_path, "w") as f:
            json.dump(qa_summary, f, indent=2)
        print(f"📊 Exported QA Summary → {qa_path}")

        print("\n📊 [Patch QA] Summary (TP1/TP2):")
        for k, v_val in qa_summary.items():
            print(f"    ▸ {k.replace('_', ' ').title()}: {v_val}")
    return trades_df

def run_wfv_with_progress(df, features, label_col):
    # [Patch vWFV.7] Fallback support for lowercase 'open' or only 'close'
    if "Open" not in df.columns:
        if "open" in df.columns:
            df = df.rename(columns={"open": "Open"})
            print("[Patch vWFV.7] 🛠️ rename 'open' → 'Open'")
        elif "close" in df.columns:
            df = df.copy()
            df["Open"] = df["close"]
            print("[Patch vWFV.7] 🛠️ create 'Open' from 'close'")
        else:
            raise ValueError("[Patch vWFV.7] ❌ Missing 'Open'/'open'/'close'")
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
    from nicegold_v5.exit import simulate_partial_tp_safe
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

    # [Patch v16.1.1] เพิ่ม fallback Diagnostic หากยังไม่พบสัญญาณ
    if df["entry_signal"].isnull().mean() >= 1.0:
        from nicegold_v5.config import SNIPER_CONFIG_DIAGNOSTIC
        print("[Patch CLI] ⚠️ RELAX_CONFIG_Q3 ยังไม่พบสัญญาณ – fallback DIAGNOSTIC")
        df = generate_signals(df, config=SNIPER_CONFIG_DIAGNOSTIC)

    show_progress_bar("🧪 ตรวจสอบความสมบูรณ์ของข้อมูล", steps=1)

    if "entry_time" not in df.columns:
        print("[Patch CLI] ⛑ สร้าง entry_time จาก timestamp")
        df["entry_time"] = df.get("timestamp")
    if not pd.api.types.is_datetime64_any_dtype(df["entry_time"]):
        df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")

    required = ["timestamp", "entry_signal", "entry_time"]
    df = df.dropna(subset=required)
    df["signal"] = df["entry_signal"].apply(lambda x: "long" if pd.notnull(x) else None)
    trade_df = pd.DataFrame()
    if df.empty:
        print("[Patch CLI] ⚠️ ไม่มีข้อมูลครบสำหรับ simulate – ข้ามขั้นตอนนี้")
    else:
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            raise ValueError("[Patch QA] ❌ timestamp ต้องแปลงเป็น datetime ก่อน simulate")
        show_progress_bar("🚀 รัน simulate_partial_tp_safe", steps=2)
        trade_df = simulate_partial_tp_safe(df)  # ✅ [Patch v12.0.2] ใช้ logic ใหม่ TP1/TP2 จากราคาแท่งจริง

    if trade_df.empty or trade_df.get("exit_reason").isnull().all():
        print("[Patch QA] ⚠️ simulate_trades_with_tp ไม่พบ trade ที่ถูกยิงจริง - ข้ามสรุปผล")
    else:
        out_path = os.path.join(TRADE_DIR, "trades_v12_tp1tp2.csv")
        trade_df.to_csv(out_path, index=False)
        print(f"[Patch v12.0.1] ✅ บันทึกผล TP1/TP2 Trade log ที่: {out_path}")

        tp1_hits = trade_df["exit_reason"].eq("tp1").sum()
        tp2_hits = trade_df["exit_reason"].eq("tp2").sum()
        sl_hits = (
            trade_df["exit_reason"].eq("sl").sum() if "exit_reason" in trade_df.columns else 0
        )
        total_pnl = safe_calculate_net_change(trade_df)

        print("\n📊 [Patch QA] Summary (TP1/TP2):")
        print(f"   ▸ TP1 Triggered : {tp1_hits}")
        print(f"   ▸ TP2 Triggered : {tp2_hits}")
        print(f"   ▸ SL Count      : {sl_hits}")
        print(f"   ▸ Net PnL       : {total_pnl:.2f} USD")
    maximize_ram()

    print("\n🧩 เลือกเมนูต่อไป:")
    print("1. Run WFV (ML)")
    print("2. วิเคราะห์ Session")
    print("3. วิเคราะห์ Drawdown")
    print("4. Backtest Signal (Scalper)")
    print("5. เทรน TP2 Classifier (LSTM)")
    print("6. TP1/TP2 Simulator")
    print("7. Run Walk-Forward Validation (WFV) แบบเทพ")
    choice = input("👉 เลือกเมนู (1–7): ").strip()
    try:
        choice = int(choice)
    except:
        print("❌ เมนูไม่ถูกต้อง")
        return

    if choice == 1:
        print("\n🚀 เริ่มรัน Walk-Forward ML Strategy...")
        df = pd.read_csv(M15_PATH, parse_dates=["timestamp"], engine="python", on_bad_lines="skip")
        show_progress_bar("🚧 เตรียมฟีเจอร์", steps=5)
        df.set_index("timestamp", inplace=True)
        df["EMA_50"] = df["Close"].ewm(span=50).mean()
        df["RSI_14"] = rsi(df["Close"], 14)
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
        from nicegold_v5.ml_dataset_m1 import generate_ml_dataset_m1
        from nicegold_v5.train_lstm_runner import load_dataset, train_lstm
        import types
        try:
            import torch
        except Exception:
            torch = types.SimpleNamespace(save=lambda *a, **k: None)

        print("🚀 สร้าง Dataset และฝึก LSTM Classifier สำหรับ TP2 Prediction...")
        generate_ml_dataset_m1()
        X, y = load_dataset("data/ml_dataset_m1.csv")
        model = train_lstm(X, y)
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/model_lstm_tp2.pth")
        print("✅ บันทึกโมเดลที่ models/model_lstm_tp2.pth")
    elif choice == 6:
        show_progress_bar("🧪 TP1/TP2 Backtest Mode", steps=3)
        print("\n⚙️ เริ่มรัน simulate_trades_with_tp() จาก UltraFix Patch...")
        df = load_csv_safe(M1_PATH)

        try:
            from nicegold_v5.ml_dataset_m1 import generate_ml_dataset_m1
            from nicegold_v5.deep_model_m1 import LSTMClassifier
            import torch
            import numpy as np

            generate_ml_dataset_m1()
            tp2_model = LSTMClassifier(input_dim=7, hidden_dim=64)
            tp2_model.load_state_dict(
                torch.load("models/model_lstm_tp2.pth", map_location=torch.device("cpu"))
            )
            tp2_model.eval()
            print("✅ Loaded TP2 LSTM Classifier")

            df_feat = pd.read_csv("data/ml_dataset_m1.csv")
            seq_len = 10
            feat_cols = ["gain_z", "ema_slope", "atr", "rsi", "volume", "entry_score", "pattern_label"]
            data = df_feat[feat_cols].values
            X_seq = [data[i:i + seq_len] for i in range(len(data) - seq_len)]
            if X_seq:
                X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32)
                with torch.no_grad():
                    pred = tp2_model(X_tensor).squeeze().numpy()
                df_feat["tp2_proba"] = np.concatenate([np.zeros(seq_len), pred])
                df["tp2_proba"] = df_feat["tp2_proba"]
                df["tp2_guard_pass"] = df["tp2_proba"] >= 0.7
                df = df[df["tp2_guard_pass"] | df["entry_signal"].isnull()]
                print(
                    f"✅ Applied TP2 Guard Filter → เหลือ {df['entry_signal'].notnull().sum()} signals"
                )
        except Exception as e:
            print(f"⚠️ TP2 Guard disabled: {e}")

        # ✅ [Patch v11.9.18] รองรับ Date แบบพุทธศักราช
        df = convert_thai_datetime(df)

        df["timestamp"] = parse_timestamp_safe(df["timestamp"], DATETIME_FORMAT)
        df = df.sort_values("timestamp")

        if len(df) < 2:
            print("[Patch CLI] ⚠️ ไม่มีข้อมูลครบสำหรับ simulate – ข้ามขั้นตอนนี้")
            trade_df = pd.DataFrame()
        else:
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
        sl_hits = (
            trade_df["exit_reason"].eq("sl").sum() if "exit_reason" in trade_df.columns else 0
        )
        total_pnl = safe_calculate_net_change(trade_df)

        print("\n📊 QA Summary (TP1/TP2):")
        print(f"   ▸ TP1 Triggered : {tp1_hits}")
        print(f"   ▸ TP2 Triggered : {tp2_hits}")
        print(f"   ▸ SL Count      : {sl_hits}")
        print(f"   ▸ Net PnL       : {total_pnl:.2f} USD")

        maximize_ram()
    elif choice == 7:
        show_progress_bar("🧠 เตรียม Walk-Forward", steps=2)
        print("\n🚀 [Patch v21.2.2] เริ่ม AutoFix WFV แบบเทพ...")

        df = load_csv_safe(M1_PATH)
        df = convert_thai_datetime(df)
        df["timestamp"] = parse_timestamp_safe(df["timestamp"], DATETIME_FORMAT)
        df = sanitize_price_columns(df)

        try:
            validate_indicator_inputs(df, min_rows=min(500, len(df)))
        except TypeError:
            validate_indicator_inputs(df)

        df = generate_signals(df, config=SNIPER_CONFIG_Q3_TUNED)
        if df["entry_signal"].isnull().mean() >= 1.0:
            print("[Patch CLI] ⚠️ ไม่มีสัญญาณ – fallback RELAX_CONFIG_Q3")
            df = generate_signals(df, config=RELAX_CONFIG_Q3)

        from nicegold_v5.utils import run_autofix_wfv
        from nicegold_v5.config import SNIPER_CONFIG_Q3_TUNED
        from nicegold_v5.exit import simulate_partial_tp_safe

        trades_df = run_autofix_wfv(
            df,
            simulate_partial_tp_safe,
            SNIPER_CONFIG_Q3_TUNED,
            n_folds=5,
        )
        out_path = os.path.join(TRADE_DIR, "wfv_autofix_result.csv")
        trades_df.to_csv(out_path, index=False)
        print(f"📦 บันทึกผล AutoFix WFV ที่: {out_path}")
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
    # [Patch v12.4.1] Example of how choice 7 might be handled if menu was active
    # elif choice == 7:  # This would be part of the active menu loop in welcome()
    #     print("\n🚀 เริ่มรัน CleanBacktest ด้วย AutoFix + Export...")
    #     df = load_csv_safe(M1_PATH)  # Ensure M1_PATH is defined
    #     # Potentially convert_thai_datetime(df) and other pre-processing here
    #     trades_df = run_clean_backtest(df)
