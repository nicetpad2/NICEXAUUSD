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
from nicegold_v5.optuna_tuner import start_optimization
from nicegold_v5.qa import run_qa_guard, auto_qa_after_backtest
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
    df = generate_signals(df, config=SNIPER_CONFIG_Q3_TUNED)
    if df["entry_signal"].isnull().mean() >= 1.0:
        print("[AutoPipeline] ⚠️ ไม่มีสัญญาณ – fallback RELAX_CONFIG_Q3")
        df = generate_signals(df, config=RELAX_CONFIG_Q3)

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
        df = generate_signals(df, config=study.best_trial.params)

        seq_len = 10
        data = df_feat[top_features].values
        seqs = [data[i : i + seq_len] for i in range(len(data) - seq_len)]
        device2 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(np.array(seqs), dtype=torch.float32).to(device2)
            preds = model(X_tensor).squeeze().cpu().numpy()
        df_feat["tp2_proba"] = np.concatenate([np.zeros(seq_len), preds])

        df = df.merge(df_feat[["timestamp", "tp2_proba"]], on="timestamp", how="left")
        df["tp2_guard_pass"] = df["tp2_proba"] >= 0.7
        print(
            f"✅ [Patch v22.7.2] TP2 Guard Filter → เหลือ {df['entry_signal'].notnull().sum()} signals"
        )
        df = df[df["tp2_guard_pass"] | df["entry_signal"].isnull()]

        from nicegold_v5.utils import run_autofix_wfv

        trades_df = run_autofix_wfv(df, simulate_partial_tp_safe, SNIPER_CONFIG_Q3_TUNED)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(TRADE_DIR, f"trades_ai_master_{ts}.csv")
        trades_df.to_csv(out_path, index=False)
        print(f"📦 [Patch v22.7.2] Exported trades → {out_path}")
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
        df_raw = generate_signals(df_raw, config=SNIPER_CONFIG_Q3_TUNED)
        df = df_raw.merge(df_feat, on="timestamp", how="left")
        df["tp2_guard_pass"] = df["tp2_proba"] >= 0.7
        df = df[df["tp2_guard_pass"] | df["entry_signal"].isnull()]
        print(f"✅ TP2 Guard Filter → เหลือ {df['entry_signal'].notnull().sum()} signals")

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
            model.eval()
            with torch.no_grad():
                pred = model(X_tensor).squeeze().cpu().numpy()
            df_feat["tp2_proba"] = np.concatenate([np.zeros(seq_len), pred])
        else:
            df_feat["tp2_proba"] = 0.0

        df = df.merge(df_feat[["timestamp", "tp2_proba"]], on="timestamp", how="left")
        df["tp2_guard_pass"] = df["tp2_proba"] >= 0.7
        df = df[df["tp2_guard_pass"] | df["entry_signal"].isnull()]
        print(f"✅ TP2 Guard Filter → เหลือ {df['entry_signal'].notnull().sum()} signals")
    else:
        n_folds = plan["n_folds"]

    from nicegold_v5.utils import run_autofix_wfv
    trades_df = run_autofix_wfv(
        df,
        simulate_partial_tp_safe,
        SNIPER_CONFIG_Q3_TUNED,
        n_folds=n_folds,
    )
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(TRADE_DIR, f"trades_autopipeline_{ts}.csv")
    trades_df.to_csv(out_path, index=False)
    print(f"📦 Exported AutoPipeline trades → {out_path}")
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

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "clean":
        print("📥 Loading CSV...")
        df = load_csv_safe(M1_PATH)
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
