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
from nicegold_v5.config import (
    SNIPER_CONFIG_PROFIT,
    SNIPER_CONFIG_Q3_TUNED,
    RELAX_CONFIG_Q3,
    SESSION_CONFIG,
    COMPOUND_MILESTONES,
    KILL_SWITCH_DD,
    RECOVERY_SL_TRIGGER,
    RECOVERY_LOT_MULT,
    ensure_order_side_enabled,
)
from nicegold_v5.optuna_tuner import start_optimization
# [Patch v28.2.1] QA Robustness Integration (Enterprise Grade)
from nicegold_v5.qa import run_qa_guard, auto_qa_after_backtest, force_entry_stress_test  # [Patch v28.2.1] QA Robustness Integration (Enterprise Grade)
from nicegold_v5.utils import (
    safe_calculate_net_change,
    convert_thai_datetime,
    parse_timestamp_safe,
    get_resource_plan,
)
from nicegold_v5.fix_engine import simulate_and_autofix  # [Patch v12.3.9] Added import
# User-provided custom instructions
# *สนทนาภาษาไทยเท่านั้น

# --- Advanced Risk Management (Patch C) ---
# ค่า DD Limit อัปเกรดจาก config (Patch HEDGEFUND-NEXT)
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

# [Patch HEDGEFUND-NEXT] OMS Compound/KillSwitch
def update_compound_lot(equity, last_milestone, base_lot=0.01):
    for milestone in sorted(COMPOUND_MILESTONES):
        if equity >= milestone:
            last_milestone = milestone
    lot = max(base_lot, round(0.01 * (last_milestone / 100), 2))
    return lot, last_milestone


def kill_switch_hedge(equity_curve, dd_limit=KILL_SWITCH_DD):
    peak = equity_curve[0]
    for eq in equity_curve:
        dd = (peak - eq) / peak * 100
        if dd >= dd_limit:
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


def run_smart_fast_qa():
    """Run a minimal set of tests for ultra-fast QA."""
    import subprocess
    print("\n🤖 Running Smart Fast QA tests...")
    try:
        subprocess.run([
            "pytest",
            "-q",
            "nicegold_v5/tests/test_core_all.py",
        ], check=True)
        print("✅ Smart Fast QA Passed")
    except Exception as e:
        print(f"❌ Smart Fast QA Failed: {e}")

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
    """Run backtest with session-based config and OMS risk management."""
    df = df.copy()
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

    try:
        df = sanitize_price_columns(df)
        validate_indicator_inputs(df, min_rows=min(500, len(df)))
    except TypeError:
        validate_indicator_inputs(df)

    df["session"] = df["timestamp"].dt.hour.apply(lambda h: "Asia" if h < 8 else "London" if h < 15 else "NY")
    all_trades = []
    for sess_name, cfg in SESSION_CONFIG.items():
        df_sess = df[df["session"] == sess_name].copy()
        if df_sess.empty:
            continue
        print(f"\n[Patch HEDGEFUND-NEXT] Running {sess_name} session with config: {cfg}")
        ensure_order_side_enabled(cfg)
        df_sess = generate_signals(df_sess, config=cfg, test_mode=True)
        if df_sess["entry_signal"].isnull().mean() >= 1.0:
            continue
        trades_df, _, _ = simulate_and_autofix(df_sess, simulate_partial_tp_safe, cfg, session=sess_name)
        trades_df["session"] = sess_name
        all_trades.append(trades_df)

    if not all_trades:
        raise RuntimeError("[Patch HEDGEFUND-NEXT] ❌ ไม่มีสัญญาณเข้าเลย – หยุดรัน backtest")

    df_trades = pd.concat(all_trades, ignore_index=True)

    capital = 100.0
    last_milestone = 100
    equity_curve = [capital]
    sl_streak = 0
    recovery_mode = False
    processed = []
    for _, row in df_trades.iterrows():
        lot, last_milestone = update_compound_lot(capital, last_milestone)
        if recovery_mode:
            lot = round(lot * RECOVERY_LOT_MULT, 2)
        pnl = float(row.get("pnl", 0.0))
        capital += pnl
        exit_reason = row.get("exit_reason", "TP" if pnl >= 0 else "SL")
        processed.append({
            "entry_time": row.get("entry_time"),
            "exit_time": row.get("exit_time"),
            "type": row.get("type"),
            "lot": lot,
            "pnl": capital,
            "exit_reason": exit_reason,
            "session": row.get("session"),
            "risk_mode": "recovery" if recovery_mode else "normal",
        })
        if pnl < 0:
            sl_streak += 1
        else:
            sl_streak = 0
        if sl_streak >= RECOVERY_SL_TRIGGER:
            recovery_mode = True
            sl_streak = 0
        else:
            recovery_mode = False
        equity_curve.append(capital)
        if kill_switch_hedge(equity_curve):
            break

    trades_df_final = pd.DataFrame(processed)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(TRADE_DIR, f"trades_hedgefund_next_{ts}.csv")
    trades_df_final.to_csv(out_path, index=False)
    print(f"📁 Exported trades → {out_path}")
    return trades_df_final

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


def autopipeline(mode="default", train_epochs=1):
    """Run full ML + AutoFix WFV pipeline automatically."""
    print("\n🚀 เริ่ม NICEGOLD AutoPipeline")
    maximize_ram()

    try:
        from nicegold_v5.ml_dataset_m1 import generate_ml_dataset_m1
        from nicegold_v5.train_lstm_runner import load_dataset, train_lstm
        import torch
    except Exception:
        torch = None
        generate_ml_dataset_m1 = None
        load_dataset = None
        train_lstm = None
        print("⚠️ PyTorch ไม่พร้อมใช้งาน - ข้ามขั้นตอน LSTM")

    plan = get_resource_plan()
    device = plan["device"]
    DEVICE = torch.device(device) if torch else None
    print("\n🧠 AI Resource Plan Summary:")
    print(f"   ▸ GPU       : {plan['gpu']}")
    print(f"   ▸ RAM       : {plan['ram']:.2f} GB")
    print(f"   ▸ VRAM      : {plan['vram']:.2f} GB")
    print(f"   ▸ CUDA Core : {plan['cuda_cores']}")
    print(f"   ▸ Threads   : {plan['threads']}")
    print(f"   ▸ Precision : {plan['precision']}")
    print(f"   ▸ Batch     : {plan['batch_size']}")
    print(f"   ▸ ModelDim  : {plan['model_dim']}")
    print(f"   ▸ Epochs    : {plan['train_epochs']}")
    print(f"   ▸ Optimizer : {plan['optimizer']}")
    print("✅ บันทึก resource_plan.json แล้วที่ logs/")

    batch_size = plan["batch_size"]
    model_dim = plan["model_dim"]
    n_folds = plan["n_folds"]
    lr = plan.get("lr", 0.001)
    opt = plan["optimizer"]
    train_epochs = plan.get("train_epochs", train_epochs)

    # Load and prepare CSV for all pipeline modes
    df = load_csv_safe(M1_PATH)
    df = convert_thai_datetime(df)
    df["timestamp"] = parse_timestamp_safe(df["timestamp"], DATETIME_FORMAT)
    df = sanitize_price_columns(df)
    try:
        validate_indicator_inputs(df, min_rows=min(500, len(df)))
    except TypeError:
        validate_indicator_inputs(df)
    ensure_order_side_enabled(SNIPER_CONFIG_Q3_TUNED)
    test_mode = mode in ["QA", "test", "WFV", "diagnose"]
    df = generate_signals(df, config=SNIPER_CONFIG_Q3_TUNED)  # เอา test_mode ออกถ้าไม่ได้ใช้จริง
    if df["entry_signal"].isnull().mean() >= 1.0:
        print("[AutoPipeline] ⚠️ ไม่มีสัญญาณ – fallback RELAX_CONFIG_Q3")
        ensure_order_side_enabled(RELAX_CONFIG_Q3)
        df = generate_signals(df, config=RELAX_CONFIG_Q3, test_mode=test_mode)

    if mode == "ai_master" and torch is not None:
        print("\n🧠 [AI Master Pipeline] เริ่มทำงานแบบครบวงจร (SHAP + Optuna + Guard + WFV)")
        generate_ml_dataset_m1(csv_path=M1_PATH, out_path="data/ml_dataset_m1.csv")
        X, y = load_dataset("data/ml_dataset_m1.csv")
        print(f"[Debug] y unique: {y.unique()}, value_counts: {np.unique(y, return_counts=True)}")
        model = train_lstm(
            X,
            y,
            hidden_dim=model_dim,
            epochs=train_epochs,
            lr=lr,
            batch_size=batch_size,
            optimizer_name=opt,
        )
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/model_lstm_tp2.pth")
        print("✅ บันทึกโมเดล LSTM แล้ว")

        try:
            import shap

            explainer = shap.DeepExplainer(model, X[:100])
            shap_vals = explainer.shap_values(X[:100])[0]
            shap_mean = np.abs(shap_vals).mean(axis=0)
            top_k_idx = np.argsort(shap_mean)[::-1][:5]
            all_features = [
                "gain_z",
                "ema_slope",
                "atr",
                "rsi",
                "volume",
                "entry_score",
                "pattern_label",
            ]
            top_features = [all_features[i] for i in top_k_idx]
            print("📊 [Patch v24.1.0] SHAP Top Features:", top_features)
            shap.summary_plot(shap_vals, X[:100], feature_names=top_features, show=False)
            import matplotlib.pyplot as plt
            plt.savefig("logs/shap_summary.png")
            with open("logs/shap_top_features.json", "w") as f:
                json.dump(top_features, f, indent=2)
        except Exception as e:
            print(f"⚠️ [Patch v24.1.0] SHAP skipped: {e}")
            top_features = [
                "gain_z",
                "ema_slope",
                "atr",
                "rsi",
                "volume",
                "entry_score",
                "pattern_label",
            ]

        # [Patch v24.1.0] ✅ เทรน MetaClassifier จาก trade log
        try:
            from sklearn.ensemble import RandomForestClassifier
            import joblib
            trades_meta = pd.read_csv("logs/trades_v12_tp1tp2.csv")
            trades_meta = trades_meta.dropna(subset=["exit_reason", "entry_score"])
            trades_meta["target"] = trades_meta["exit_reason"].eq("tp2").astype(int)
            features = ["gain_z", "ema_slope", "atr", "rsi", "entry_score"]
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(trades_meta[features], trades_meta["target"])
            os.makedirs("models", exist_ok=True)
            joblib.dump(clf, "models/meta_exit.pkl")
            print("✅ [MetaClassifier] บันทึก model → models/meta_exit.pkl")
        except Exception as e:
            print(f"⚠️ [MetaClassifier] Training Failed: {e}")

        # [Patch v24.1.0] ✅ เทรน RL Agent จาก trade log
        try:
            from nicegold_v5.rl_agent import RLScalper
            trades_rl = trades_meta.copy()
            agent = RLScalper()
            for i in range(len(trades_rl) - 1):
                s = int(trades_rl.iloc[i]["gain_z"] > 0)
                ns = int(trades_rl.iloc[i + 1]["gain_z"] > 0)
                reward = (
                    1
                    if trades_rl.iloc[i]["exit_reason"] == "tp2"
                    else -1
                    if "sl" in trades_rl.iloc[i]["exit_reason"]
                    else 0.5
                )
                action = 0 if trades_rl.iloc[i]["entry_signal"] == "buy" else 1
                agent.update(s, action, reward, ns)
            import pickle

            with open("models/rl_agent.pkl", "wb") as f:
                pickle.dump(agent, f)
            print("✅ [RL Agent] บันทึก model → models/rl_agent.pkl")
        except Exception as e:
            print(f"⚠️ [RL Trainer] Failed: {e}")

        df_feat = pd.read_csv("data/ml_dataset_m1.csv")
        df_feat = df_feat.dropna(subset=top_features)
        df_feat = df_feat[top_features + ["timestamp", "tp2_hit"]]
        study = start_optimization(df, n_trials=100)
        print("✅ [Patch v22.7.2] Optuna เสร็จสิ้น – บันทึก best trial")
        with open("logs/optuna_best_config.json", "w") as f:
            json.dump(study.best_trial.params, f, indent=2)

        df = load_csv_safe(M1_PATH)
        df = convert_thai_datetime(df)
        df["timestamp"] = parse_timestamp_safe(df["timestamp"], DATETIME_FORMAT)
        df = sanitize_price_columns(df)
        ensure_order_side_enabled(study.best_trial.params)
        df = generate_signals(df, config=study.best_trial.params, test_mode=True)

        seq_len = 10
        data = df_feat[top_features].values
        seqs = [data[i : i + seq_len] for i in range(len(data) - seq_len)]
        device2 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # [Patch v25.1.0] Fix: Ensure 'timestamp' dtype matches before merge (datetime64[ns])
        print("[Patch v25.1.0] ตรวจสอบ dtype ก่อน merge: df.timestamp =", df["timestamp"].dtype, "df_feat.timestamp =", df_feat["timestamp"].dtype)
        if df["timestamp"].dtype != "datetime64[ns]":
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            print("[Patch v25.1.0] → df['timestamp'] แปลงเป็น datetime64[ns]")
        if df_feat["timestamp"].dtype != "datetime64[ns]":
            df_feat["timestamp"] = pd.to_datetime(df_feat["timestamp"], errors="coerce")
            print("[Patch v25.1.0] → df_feat['timestamp'] แปลงเป็น datetime64[ns]")

        # [Patch v25.0.0] Batch inference LSTM ป้องกัน OOM
        def predict_lstm_in_batches(model, X_data, batch_size=1024, device=None):
            model.eval()
            preds = []
            n = X_data.shape[0]
            for i in range(0, n, batch_size):
                X_batch = X_data[i:i+batch_size]
                with torch.no_grad():
                    batch_pred = model(X_batch.to(device)).squeeze().cpu().numpy()
                    preds.append(batch_pred)
            return np.concatenate(preds)

        X_tensor = torch.tensor(np.array(seqs), dtype=torch.float32).to(device2)
        preds = predict_lstm_in_batches(model, X_tensor, batch_size=1024, device=device2)
        df_feat["tp2_proba"] = np.concatenate([np.zeros(seq_len), preds])

        # [Patch v25.1.0] log: dtype after fix
        print("[Patch v25.1.0] dtype หลังแปลง: df.timestamp =", df["timestamp"].dtype, "df_feat.timestamp =", df_feat["timestamp"].dtype)
        df = df.merge(df_feat[["timestamp", "tp2_proba"]], on="timestamp", how="left")
        # [Patch v27.0.0] Adaptive threshold (auto fallback)
        for THRESHOLD in [0.5, 0.4, 0.3, 0.0]:
            df["tp2_guard_pass"] = df["tp2_proba"] >= THRESHOLD
            n_signal = df[df["tp2_guard_pass"] & df["entry_signal"].notnull()].shape[0]
            print(f"[Patch v27.0.0] TP2 Guard @ {THRESHOLD:.2f} → {n_signal} signals")
            if n_signal > 20:
                break
        if n_signal == 0:
            print("[Patch v27.0.0] 🚨 No signals after TP2 Guard – Fallback: ปิด guard ทุกไม้ (DEV)")
            df["tp2_guard_pass"] = True
        df = df[df["tp2_guard_pass"] | df["entry_signal"].isnull()]
        print(f"[Patch v27.0.0] TP2 Guard Filter (final) → เหลือ {df['entry_signal'].notnull().sum()} signals")

        from nicegold_v5.utils import run_autofix_wfv

        ensure_order_side_enabled(SNIPER_CONFIG_Q3_TUNED)
        trades_df = run_autofix_wfv(df, simulate_partial_tp_safe, SNIPER_CONFIG_Q3_TUNED)
        if trades_df.empty or trades_df["pnl"].abs().sum() == 0:
            print("[Patch v27.0.0] 🚨 WFV ได้ 0 trades ทั้งหมด! Fallback: relax entry_signal + ปิด guard")
            df["tp2_guard_pass"] = True
            df = df[df["entry_signal"].notnull()]
            relax_config = {
                "gain_z_thresh": -0.2,
                "ema_slope_min": -0.01,
                "atr_thresh": 0.1,
                "sniper_risk_score_min": 2.0,
                "tp_rr_ratio": 4.0,
                "volume_ratio": 0.4,
                "disable_buy": False,
                "disable_sell": False,
            }
            df = generate_signals(df, config=relax_config, test_mode=True)
            trades_df = run_autofix_wfv(df, simulate_partial_tp_safe, relax_config)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(TRADE_DIR, f"trades_ai_master_{ts}.csv")
        trades_df.to_csv(out_path, index=False)
        print(f"📦 [Patch v22.7.2] Exported trades → {out_path}")
        if trades_df.empty:
            print("[Patch v27.0.0] ❌ ยังไม่มี trade หลัง fallback – ตรวจสอบ label/entry logic")
        return trades_df

    # [Patch v24.0.0] ✅ เพิ่ม AI Fusion Mode: LSTM + SHAP + Meta + RL Fallback
    if mode == "fusion_ai" and torch is not None:
        print("\n🚀 [Fusion AI] เริ่ม Pipeline แบบ AI Fusion Strategy Engine")
        from nicegold_v5.ml_dataset_m1 import generate_ml_dataset_m1
        from nicegold_v5.train_lstm_runner import load_dataset, train_lstm
        import shap

        generate_ml_dataset_m1(csv_path=M1_PATH, out_path="data/ml_dataset_m1.csv")
        X, y = load_dataset("data/ml_dataset_m1.csv")
        print(f"[Debug] y unique: {y.unique()}, value_counts: {np.unique(y, return_counts=True)}")
        model = train_lstm(X, y, epochs=train_epochs)
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/model_lstm_tp2.pth")
        print("✅ บันทึกโมเดล TP2 LSTM แล้ว")

        explainer = shap.DeepExplainer(model, X[:100])
        shap_vals = explainer.shap_values(X[:100])[0]
        shap_mean = np.abs(shap_vals).mean(axis=0)
        all_features = ["gain_z", "ema_slope", "atr", "rsi", "volume", "entry_score", "pattern_label"]
        top_k_idx = np.argsort(shap_mean)[::-1][:5]
        top_features = [all_features[i] for i in top_k_idx]
        print("📊 [SHAP] Top Features:", top_features)
        with open("logs/shap_top_features.json", "w") as f:
            json.dump(top_features, f)

        df_feat = pd.read_csv("data/ml_dataset_m1.csv")
        df_feat["tp2_proba"] = np.concatenate([np.zeros(10), model(X).detach().cpu().numpy().flatten()])
        df_feat = df_feat[["timestamp", "tp2_proba"] + top_features]

        df_raw = load_csv_safe(M1_PATH)
        df_raw = sanitize_price_columns(df_raw)
        ensure_order_side_enabled(SNIPER_CONFIG_Q3_TUNED)
        df_raw = generate_signals(df_raw, config=SNIPER_CONFIG_Q3_TUNED, test_mode=True)
        df = df_raw.merge(df_feat, on="timestamp", how="left")
        # [Patch v27.0.0] Adaptive threshold (auto fallback)
        for THRESHOLD in [0.5, 0.4, 0.3, 0.0]:
            df["tp2_guard_pass"] = df["tp2_proba"] >= THRESHOLD
            n_signal = df[df["tp2_guard_pass"] & df["entry_signal"].notnull()].shape[0]
            print(f"[Patch v27.0.0] TP2 Guard @ {THRESHOLD:.2f} → {n_signal} signals")
            if n_signal > 20:
                break
        if n_signal == 0:
            print("[Patch v27.0.0] 🚨 No signals after TP2 Guard – Fallback: ปิด guard ทุกไม้ (DEV)")
            df["tp2_guard_pass"] = True
        df = df[df["tp2_guard_pass"] | df["entry_signal"].isnull()]
        print(f"[Patch v27.0.0] TP2 Guard Filter (final) → เหลือ {df['entry_signal'].notnull().sum()} signals")

        try:
            from nicegold_v5.meta_classifier import MetaClassifier
            meta = MetaClassifier("model/meta_exit.pkl")
            df_meta = df[df["entry_signal"].notnull()].copy()
            df_meta["meta_decision"] = meta.predict(df_meta)
            df = df_meta[df_meta["meta_decision"] == 1]
            print(f"✅ MetaClassifier เหลือ {len(df)} signals")
        except Exception:
            print("⚠️ MetaClassifier ไม่พร้อม → ข้าม")

        if df.empty:
            from nicegold_v5.rl_agent import RLScalper
            agent = RLScalper()
            df_rl = load_csv_safe(M1_PATH)
            agent.train(df_rl)
            print("✅ RL Fallback ยิงไม้แบบ Q-Learning")
            df_rl["entry_signal"] = df_rl.apply(lambda r: "buy" if agent.act(int(r["close"] > r["open"])) == 0 else "sell", axis=1)
            df = df_rl

        from nicegold_v5.utils import run_autofix_wfv
        ensure_order_side_enabled(SNIPER_CONFIG_Q3_TUNED)
        trades_df = run_autofix_wfv(df, simulate_partial_tp_safe, SNIPER_CONFIG_Q3_TUNED, n_folds=5)
        out_path = os.path.join(TRADE_DIR, "trades_fusion_ai_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv")
        trades_df.to_csv(out_path, index=False)
        print(f"📦 Exported FusionAI trades → {out_path}")
        maximize_ram()
        return trades_df

    if torch is not None and mode in ["full", "ultra"]:
        print(
            f"⚙️ [mode={mode}] Training LSTM {train_epochs} epochs hidden_dim={model_dim} batch_size={batch_size} lr={lr} optimizer={opt}"
        )
        # Step 2: Generate ML Dataset safely (after timestamp is confirmed)
        try:
            generate_ml_dataset_m1(csv_path=M1_PATH, out_path="data/ml_dataset_m1.csv")
        except Exception as e:
            print("❌ ไม่สามารถสร้าง dataset ได้:", e)
            return

        # Step 3: Train LSTM Classifier
        X, y = load_dataset("data/ml_dataset_m1.csv")
        print(f"[Debug] y unique: {y.unique()}, value_counts: {np.unique(y, return_counts=True)}")
        model = train_lstm(
            X,
            y,
            hidden_dim=model_dim,
            epochs=train_epochs,
            lr=lr,
            batch_size=batch_size,
            optimizer_name=opt,
        )
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/model_lstm_tp2.pth")
        print("✅ บันทึกโมเดลที่ models/model_lstm_tp2.pth")

        df_feat = pd.read_csv("data/ml_dataset_m1.csv")
        df_feat["timestamp"] = parse_timestamp_safe(df_feat["timestamp"], DATETIME_FORMAT)
        df_feat["timestamp"] = pd.to_datetime(df_feat["timestamp"], errors="coerce")
        df_feat = df_feat.dropna(subset=["timestamp"])
        feat_cols = [
            "gain_z",
            "ema_slope",
            "atr",
            "rsi",
            "volume",
            "entry_score",
            "pattern_label",
        ]
        data = df_feat[feat_cols].values
        seq_len = 10
        seqs = [data[i : i + seq_len] for i in range(len(data) - seq_len)]
        if seqs:
            X_tensor = torch.tensor(np.array(seqs), dtype=torch.float32).to(DEVICE)
            model = model.to(DEVICE)

            # [Patch v25.0.0] ใช้ batch inference เพื่อป้องกัน OOM
            def predict_lstm_in_batches(model, X_data, batch_size=1024, device=None):
                model.eval()
                preds = []
                n = X_data.shape[0]
                for i in range(0, n, batch_size):
                    X_batch = X_data[i:i+batch_size]
                    with torch.no_grad():
                        batch_pred = model(X_batch.to(device)).squeeze().cpu().numpy()
                        preds.append(batch_pred)
                return np.concatenate(preds)

            pred = predict_lstm_in_batches(model, X_tensor, batch_size=1024, device=DEVICE)
            df_feat["tp2_proba"] = np.concatenate([np.zeros(seq_len), pred])
        else:
            df_feat["tp2_proba"] = 0.0

        df = df.merge(df_feat[["timestamp", "tp2_proba"]], on="timestamp", how="left")
        # [Patch v27.0.0] Adaptive threshold (auto fallback)
        for THRESHOLD in [0.5, 0.4, 0.3, 0.0]:
            df["tp2_guard_pass"] = df["tp2_proba"] >= THRESHOLD
            n_signal = df[df["tp2_guard_pass"] & df["entry_signal"].notnull()].shape[0]
            print(f"[Patch v27.0.0] TP2 Guard @ {THRESHOLD:.2f} → {n_signal} signals")
            if n_signal > 20:
                break
        if n_signal == 0:
            print("[Patch v27.0.0] 🚨 No signals after TP2 Guard – Fallback: ปิด guard ทุกไม้ (DEV)")
            df["tp2_guard_pass"] = True
        df = df[df["tp2_guard_pass"] | df["entry_signal"].isnull()]
        print(f"[Patch v27.0.0] TP2 Guard Filter (final) → เหลือ {df['entry_signal'].notnull().sum()} signals")
    else:
        n_folds = plan["n_folds"]

    from nicegold_v5.utils import run_autofix_wfv
    trades_df = run_autofix_wfv(
        df,
        simulate_partial_tp_safe,
        ensure_order_side_enabled(SNIPER_CONFIG_Q3_TUNED),
        n_folds=n_folds,
    )
    if trades_df.empty or trades_df["pnl"].abs().sum() == 0:
        print("[Patch v27.0.0] 🚨 WFV ได้ 0 trades ทั้งหมด! Fallback: relax entry_signal + ปิด guard")
        df["tp2_guard_pass"] = True
        df = df[df["entry_signal"].notnull()]
        relax_config = {
            "gain_z_thresh": -0.2,
            "ema_slope_min": -0.01,
            "atr_thresh": 0.1,
            "sniper_risk_score_min": 2.0,
            "tp_rr_ratio": 4.0,
            "volume_ratio": 0.4,
            "disable_buy": False,
            "disable_sell": False,
        }
        df = generate_signals(df, config=relax_config, test_mode=True)
        trades_df = run_autofix_wfv(df, simulate_partial_tp_safe, relax_config)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(TRADE_DIR, f"trades_autopipeline_{ts}.csv")
    trades_df.to_csv(out_path, index=False)
    print(f"📦 Exported AutoPipeline trades → {out_path}")
    if trades_df.empty:
        print("[Patch v27.0.0] ❌ ยังไม่มี trade หลัง fallback – ตรวจสอบ label/entry logic")
    maximize_ram()
    return trades_df

def welcome():
    print("\n🟡 NICEGOLD Assistant พร้อมให้บริการแล้ว (L4 GPU + QA Guard)")
    maximize_ram()

    print("\n🧩 เลือกเมนู:")
    print("1. 🚀 Full AutoPipeline Mode (ML+Optuna+LSTM+SHAP+RL+WFV)")
    print("2. 🟢 Smart Fast QA (Ultra-Fast Logic Test)")
    choice = input("👉 เลือกเมนู (1–2): ").strip()
    try:
        choice = int(choice)
    except Exception:
        print("❌ เมนูไม่ถูกต้อง")
        return

    if choice == 1:
        print("\n🚀 [Full AutoPipeline] โหมด ML+Optuna+LSTM+SHAP+RL+WFV ครบทุกฟีเจอร์")
        autopipeline(mode="ai_master", train_epochs=50)
        maximize_ram()
        return
    elif choice == 2:
        print("\n🟢 [Smart Fast QA Mode] รันทดสอบ ultra-fast logic...")
        run_smart_fast_qa()
        maximize_ram()
        return
    else:
        print("❌ เลือกเมนูไม่ถูกต้อง")
        maximize_ram()
        return

def run_production_wfv():
    """[Patch v28.2.1] Production WFV + Auto QA"""
    trades, equity = run_walkforward_backtest()
    auto_qa_after_backtest(trades, equity, label="ProductionWFV")


def run_qa_robustness():
    """[Patch v28.2.1] Full QA Robustness: Overfit, Noise, Leak, Bias"""
    trades, equity = run_walkforward_backtest()
    features = pd.read_csv("data/ml_dataset_m1.csv") if os.path.exists("data/ml_dataset_m1.csv") else None
    if features is not None:
        run_qa_guard(trades, features)
    auto_qa_after_backtest(trades, equity, label="QA_Robustness")


def run_forceentry_stress():
    """[Patch v28.2.1] ForceEntry Stress Test"""
    df = pd.read_csv("XAUUSD_M1.csv")
    config = {"version": "v28.2.1", "run": "ForceEntryStress"}
    force_entry_stress_test(df, config)


def export_audit():
    """[Patch v28.2.1] Audit Export Example (Enterprise Grade)"""
    from nicegold_v5.utils import export_audit_report
    path = "/content/drive/MyDrive/NICEGOLD/logs/qa"
    export_audit_report(
        config={"example": True},
        metrics={"result": "pass"},
        run_type="manual_export",
        version="v28.2.1",
        fold="manual",
        outdir=path,
    )
    print(f"🔒 [Audit Export] QA report example → {path}")


def main_menu():
    print("\n📊 NICEGOLD CLI MENU [QA/Production Enterprise]")
    print("[1] Production (WFV)   [2] QA Robustness")
    print("(Report/export จะ export อัตโนมัติหลัง run)")
    choice = input("เลือกเมนู (1–2): ")
    if choice == "1":
        run_production_wfv()
    elif choice == "2":
        run_qa_robustness()
    else:
        print("❌ เมนูไม่ถูกต้อง [เลือก 1 หรือ 2]")


if __name__ == "__main__":
    main_menu()
