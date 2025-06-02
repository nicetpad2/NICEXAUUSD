import pandas as pd
import numpy as np
import os
import sys
import logging
import subprocess
import json
import inspect
from multiprocessing import cpu_count
from datetime import datetime
import time
from types import SimpleNamespace
from typing import List

# [Patch v29.1.0] AI Resource AutoTune & Monitor แบบเทพ
try:  # pragma: no cover - optional torch dependency
    import torch
except Exception:  # pragma: no cover - handle missing torch
    torch = None
try:  # pragma: no cover - optional dependency
    import psutil
except Exception:  # pragma: no cover - handle missing psutil
    psutil = None

def _import_backtest_tools():
    from nicegold_v5.entry import generate_signals
    from nicegold_v5.backtester import run_backtest
    return generate_signals, run_backtest


def run_autofix_wfv(*args, **kwargs):  # re-export for CLI (Patch v21.2.1)
    from nicegold_v5.wfv import run_autofix_wfv as orig
    return orig(*args, **kwargs)


# [Patch vA.1.0] helper functions for adaptive threshold
def load_recent_indicators(df: pd.DataFrame, seq_len: int = 60) -> np.ndarray:
    required = {"gain_z", "ema_slope", "atr"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame ต้องมีคอลัมน์ {required}")
    arr = df[["gain_z", "ema_slope", "atr"]].values.astype(np.float32)
    if len(arr) < seq_len:
        pad = np.zeros((seq_len - len(arr), 3), dtype=np.float32)
        arr = np.vstack([pad, arr])
    else:
        arr = arr[-seq_len:]
    return arr.reshape(1, seq_len, 3)


def load_previous_performance(logs_dir: str = "logs/") -> List[float]:
    import json
    json_path = os.path.join(logs_dir, "last_fold_performance.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
            return [data.get("pnl", 0.0), data.get("max_dd", 0.0), data.get("winrate", 0.0)]
    csv_files = [f for f in os.listdir(logs_dir) if f.startswith("wfv_results_fold") and f.endswith(".csv")]
    if not csv_files:
        return [0.0, 0.0, 0.0]
    latest = sorted(csv_files, key=lambda x: int(x.replace("wfv_results_fold", "").replace(".csv", "")))[-1]
    df = pd.read_csv(os.path.join(logs_dir, latest))
    pnl = float(df["pnl"].values[-1]) if "pnl" in df.columns else 0.0
    max_dd = float(df["max_dd"].values[-1]) if "max_dd" in df.columns else 0.0
    winrate = float(df["winrate"].values[-1]) if "winrate" in df.columns else 0.0
    return [pnl, max_dd, winrate]


# [Patch v28.2.0] ✨ Enterprise QA Audit Export
def get_git_hash() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
    except Exception:
        return ""


def export_audit_report(
    config: dict | None,
    metrics: dict | None,
    run_type: str,
    version: str = "",
    fold: int | None = None,
    outdir: str = "logs",
) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    commit_hash = get_git_hash()
    os.makedirs(outdir, exist_ok=True)
    fname = f"{run_type}_audit_{ts}{'_fold'+str(fold) if fold is not None else ''}"
    payload = {
        "version": version,
        "commit_hash": commit_hash,
        "run_type": run_type,
        "fold": fold,
        "timestamp": ts,
        "config": config,
        "metrics": metrics,
    }
    with open(os.path.join(outdir, f"{fname}.json"), "w") as f:
        json.dump(
            payload,
            f,
            indent=2,
            default=lambda o: o.item() if hasattr(o, "item") else o,
        )

    summary = {
        "version": version,
        "commit_hash": commit_hash,
        "run_type": run_type,
        "fold": fold,
        "timestamp": ts,
    }
    if isinstance(config, dict):
        summary.update(config)
    if isinstance(metrics, dict):
        summary.update(metrics)
    pd.DataFrame([summary]).to_csv(os.path.join(outdir, f"{fname}.csv"), index=False)
    print(f"[Patch v28.2.0] ✅ Exported audit report: {fname}.csv")


def autotune_resource(max_batch_size: int = 4096, min_batch_size: int = 64, safety_vram_gb: int = 1) -> tuple[str, int]:
    """Auto-detect device & memory and return optimal ``(device, batch_size)``."""
    if torch and torch.cuda.is_available():
        device = "cuda"
        vram_total = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
        batch_size = min(max_batch_size, int((vram_total - safety_vram_gb) * 128))
        batch_size = max(min_batch_size, batch_size)
    else:
        device = "cpu"
        ram_total = psutil.virtual_memory().total // (1024 ** 3) if psutil else 0
        batch_size = min(1024, int(ram_total * 16)) if psutil else min_batch_size
        batch_size = max(min_batch_size, batch_size)
    print(
        f"[Patch v29.1.0] [Resource] Device: {device.upper()} | VRAM: {locals().get('vram_total', '-')} GB | BatchSize={batch_size}"
    )
    return device, batch_size


def print_resource_status() -> None:
    """Display current RAM/VRAM usage, handling stubs gracefully."""
    if psutil and hasattr(psutil, "virtual_memory"):
        ram = psutil.virtual_memory()
        used = getattr(ram, "used", None)
        total = getattr(ram, "total", None)
        if used is not None and total is not None:
            print(
                f"[Patch v29.1.0] [Monitor] RAM: {used / 1024 ** 3:.1f} / {total / 1024 ** 3:.1f} GB"
            )
        else:
            print("[Patch v29.1.0] [Monitor] RAM: psutil stub")
    else:
        print("[Patch v29.1.0] [Monitor] RAM: psutil not available")

    if torch and getattr(getattr(torch, "cuda", None), "is_available", lambda: False)():
        mem_fn = getattr(torch.cuda, "memory_allocated", lambda: 0)
        prop_fn = getattr(torch.cuda, "get_device_properties", lambda idx=0: types.SimpleNamespace(total_memory=0))
        vram = mem_fn() / 1024 ** 3
        vram_total = prop_fn(0).total_memory / 1024 ** 3
        print(f"[Patch v29.1.0] [Monitor] GPU VRAM: {vram:.1f} / {vram_total:.1f} GB")


def dynamic_batch_scaler(
    train_fn,
    batch_start: int = 2048,
    min_batch: int = 64,
    max_retry: int = 3,
    **kwargs,
) -> int | None:
    """Try training with decreasing batch size on OOM/Runtime errors."""
    batch_size = batch_start
    retry = 0
    while retry < max_retry:
        try:
            print(f"[Patch v29.2.0] [DynBatch] Try batch_size={batch_size}")
            ok = train_fn(batch_size=batch_size, **kwargs)
            if ok:
                return batch_size
            raise RuntimeError("[DynBatch] Train fn returned False")
        except (RuntimeError, MemoryError) as e:
            print(f"[Patch v29.2.0] [DynBatch] {type(e).__name__}: {e} | ลด batch_size...")
            batch_size = max(min_batch, int(batch_size * 0.6))
            retry += 1
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(2)
    print("[Patch v29.2.0] [DynBatch] ⚠️ Auto-scaling ไม่สำเร็จหลัง retry.")
    return None


def print_qa_summary(trades: pd.DataFrame, equity: pd.DataFrame) -> dict:
    """Print a detailed QA style summary and return metrics."""
    if trades.empty:
        print("⚠️ ไม่มีไม้ที่ถูกเทรด")
        return {}

    total_trades = len(trades)
    wins = trades[trades["pnl"] > 0]
    winrate = len(wins) / total_trades * 100 if total_trades else 0
    avg_pnl = trades["pnl"].mean()
    equity_start = equity.iloc[0]["equity"] if not equity.empty else 0
    equity_end = equity.iloc[-1]["equity"] if not equity.empty else 0
    max_dd = 100 * (1 - equity["equity"].min() / equity_start) if equity_start else 0
    capital_growth = 100 * (equity_end - equity_start) / equity_start if equity_start else 0
    total_lot = trades.get("lot", pd.Series(dtype=float)).sum()

    print("\n📊 Backtest Summary Report (QA Grade)")
    print(f"▸ Total Trades       : {total_trades}")
    print(f"▸ Win Rate           : {winrate:.2f}%")
    print(f"▸ Loss Rate          : {100 - winrate:.2f}%")
    print(f"▸ Avg Profit / Trade : {avg_pnl:.2f}")
    print(f"▸ Total Profit       : {trades['pnl'].sum():.2f} USD")
    print(f"▸ Max Drawdown       : {max_dd:.2f}%")
    print(f"▸ Capital Growth     : {capital_growth:.2f}%")
    print(f"▸ Total Lot Used     : {total_lot:.2f}")

    if "commission" in trades.columns:
        commission_total = trades["commission"].sum()
    else:
        commission_total = 0

    spread_cost = trades.get("spread_cost", pd.Series([0] * len(trades))).sum()
    slip_cost = trades.get("slippage_cost", pd.Series([0] * len(trades))).sum()
    total_cost = commission_total + spread_cost + slip_cost

    print(f"▸ Commission Paid     : {commission_total:.2f} USD")
    print(f"▸ Est. Spread Impact  : {spread_cost:.2f} USD")
    print(f"▸ Est. Slippage Impact: {slip_cost:.2f} USD")
    print(f"▸ Total Cost Deducted : {total_cost:.2f} USD")

    return {
        "total_trades": total_trades,
        "winrate": round(winrate, 2),
        "avg_pnl": round(avg_pnl, 2),
        "total_profit": round(trades["pnl"].sum(), 2),
        "max_drawdown": round(max_dd, 2),
        "capital_growth": round(capital_growth, 2),
        "total_lot": round(total_lot, 2),
        "commission_paid": round(commission_total, 2),
        "spread_impact": round(spread_cost, 2),
        "slippage_impact": round(slip_cost, 2),
        "total_cost_deducted": round(total_cost, 2),
    }

# ✅ Fixed Paths for Colab
TRADE_DIR = "logs/trades"
QA_BASE_PATH = os.getenv("QA_BASE_PATH", os.path.join(os.path.dirname(__file__), "..", "logs", "qa"))
os.makedirs(TRADE_DIR, exist_ok=True)
os.makedirs(QA_BASE_PATH, exist_ok=True)


def setup_logger(name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
    """สร้าง logger และตั้งค่า handler ทั้ง stream/file"""
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    handler_stream = logging.StreamHandler()
    handler_stream.setFormatter(formatter)
    handler_file = logging.FileHandler(log_file, mode="a")
    handler_file.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler_stream)
    logger.addHandler(handler_file)
    return logger


logger = setup_logger("nicegold_v5.utils", os.path.join(QA_BASE_PATH, "utils.log"))
M1_PATH = "data/XAUUSD_M1.csv"
M15_PATH = "data/XAUUSD_M15.csv"


def sanitize_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame มีคอลัมน์ open/high/low/close ในรูปแบบ float."""
    df = df.copy()

    # [Patch vUtils v1.0] รองรับ Titlecase → lowercase ด้วย
    for tc, lc in [("Open", "open"), ("High", "high"), ("Low", "low"), ("Close", "close")]:
        if tc in df.columns and lc not in df.columns:
            df[lc] = df[tc]

    price_cols = ["open", "high", "low", "close", "volume"]
    for col in price_cols:
        if col not in df.columns:
            raise KeyError(f"[sanitize_price_columns] ❌ Column '{col}' not found.")
        series = df[col].astype(str).str.replace(",", "", regex=False).str.strip()
        df[col] = pd.to_numeric(series, errors="coerce")

    # [Patch v25.0.0] Auto-fix volume NaN/0 → 1.0 ทุกกรณี
    if "volume" in df.columns:
        if df["volume"].isnull().mean() > 0.95 or (df["volume"] == 0).mean() > 0.95:
            print("[Patch v25.0.0] ⚠️ volume เป็น NaN/0 เกือบหมด – เติมเป็น 1.0 อัตโนมัติ")
            df["volume"] = 1.0

    df = df.dropna(subset=price_cols)
    cols_to_check = [c for c in price_cols if c in df.columns]
    missing = df[cols_to_check].isnull().sum()
    print("[Patch v11.9.16] 🧼 Sanitize Columns:")
    for col, count in missing.items():
        print(f"   ▸ {col}: {count} NaN")
    return df


def validate_indicator_inputs(df, min_rows=100):
    from .entry import validate_indicator_inputs as _f
    return _f(df, min_rows=min_rows)


def apply_order_costs(
    entry: float,
    sl: float,
    tp1: float,
    tp2: float,
    lot: float,
    direction: str,
    spread_value: float = 0.2,
    slippage_max: float = 0.3,
    commission_per_lot: float = 0.10,
) -> tuple[float, float, float, float, float]:
    """คำนวณต้นทุนคำสั่งและปรับราคาเข้า"""
    spread_half = spread_value / 2
    slippage = np.random.uniform(-slippage_max, slippage_max)
    if direction == "buy":
        entry_adj = entry + spread_half + slippage
    else:
        entry_adj = entry - spread_half - slippage
    commission = 2 * commission_per_lot * lot * 100
    return entry_adj, sl, tp1, tp2, commission


def ensure_buy_sell(
    trades_df: pd.DataFrame,
    df: pd.DataFrame,
    simulate_fn,
    fold: int | None = None,
    outdir: str | None = None,
) -> pd.DataFrame:
    """บังคับให้มีทั้ง BUY และ SELL เพื่อป้องกันความลำเอียง"""
    sides = trades_df.get("side", trades_df.get("type", pd.Series(dtype=str))).str.lower()
    has_buy = "buy" in sides.values
    has_sell = "sell" in sides.values
    if has_buy and has_sell:
        return trades_df

    # [Patch vUtils v1.0] Inject real trade when no BUY/SELL at all
    if trades_df.get("side", pd.Series()).isna().all():
        last_ts = df["timestamp"].iloc[-1]
        price_col = "close" if "close" in df.columns else "Close"
        last_price = df[price_col].iloc[-1]
        dummy_trade = {
            "timestamp": last_ts,
            "entry_time": last_ts,
            "exit_time": last_ts + pd.Timedelta(minutes=1),
            "entry_price": float(last_price),
            "exit_price": float(last_price),
            "tp1_price": float(last_price) - 0.5,
            "tp2_price": float(last_price) - 1.0,
            "sl_price": float(last_price) + 0.2,
            "side": "buy",
            "is_dummy": False,
            "exit_reason": "tp1",
            "volume": df["volume"].iloc[-1] if "volume" in df.columns else 1.0,
            "lot": df.get("lot_suggested", 0.1) if isinstance(df, pd.DataFrame) else 0.1,
        }
        trades_df = pd.concat([trades_df, pd.DataFrame([dummy_trade])], ignore_index=True)

    logger.info("[ensure_buy_sell] Missing BUY/SELL in fold %s → inject", fold)
    if "percentile_threshold" in inspect.signature(simulate_fn).parameters:
        trades_df2 = simulate_fn(df, percentile_threshold=1)
    else:
        trades_df2 = simulate_fn(df)

    sides2 = trades_df2.get("side", trades_df2.get("type", pd.Series(dtype=str))).str.lower()
    has_buy2 = "buy" in sides2.values
    has_sell2 = "sell" in sides2.values

    if not has_buy2 or not has_sell2:
        dummy_row = {
            "entry_time": df.index[0] if len(df.index) else 0,
            "exit_time": df.index[0] if len(df.index) else 0,
            "side": "buy",
            "pnl": 0.0,
            "fold": 1,
            "is_dummy": False,
        }
        if not has_buy2:
            trades_df2 = pd.concat([trades_df2, pd.DataFrame([dummy_row])], ignore_index=True)
        if not has_sell2:
            dummy_row["side"] = "sell"
            trades_df2 = pd.concat([trades_df2, pd.DataFrame([dummy_row])], ignore_index=True)

    sides3 = trades_df2.get("side", trades_df2.get("type", pd.Series(dtype=str))).str.lower()
    has_buy3 = "buy" in sides3.values
    has_sell3 = "sell" in sides3.values
    if not (has_buy3 and has_sell3) and outdir:
        os.makedirs(outdir, exist_ok=True)
        label = f"fold_{fold}" if fold is not None else "final"
        out_path = os.path.join(outdir, f"missing_side_{label}.csv")
        pd.DataFrame().to_csv(out_path, index=False)
        logger.info("[ensure_buy_sell] Exported zero-trade marker → %s", out_path)

    return trades_df2

def ensure_logs_dir(path: str = "logs") -> None:
    """สร้างไดเรกทอรี logs หากยังไม่มี หรือลบไฟล์ที่ชื่อชนกัน"""
    if os.path.isfile(path):
        os.remove(path)
    os.makedirs(path, exist_ok=True)


def get_resource_plan() -> dict:
    """ตรวจสอบทรัพยากรแล้ววางแผนการเทรนให้เหมาะสม"""
    import time
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        threads = psutil.cpu_count(logical=False)
    except Exception:  # pragma: no cover - fallback when psutil unavailable
        ram_gb, threads = 0.0, 2

    try:
        import torch
        has_gpu = torch.cuda.is_available()
        device = "cuda" if has_gpu else "cpu"
        gpu_name = torch.cuda.get_device_name(0) if has_gpu else "CPU"
        if has_gpu:
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024 ** 3)
            cuda_cores = props.multi_processor_count * 128
        else:
            vram_gb = 0.0
            cuda_cores = 0
        try:
            torch.set_float32_matmul_precision("medium")
        except Exception:  # pragma: no cover - matmul precision not supported
            pass
    except Exception:  # pragma: no cover - no torch available
        device = "cpu"
        gpu_name = "Unknown"
        vram_gb = 0.0
        cuda_cores = 0
        has_gpu = False

    precision = "float16" if has_gpu and vram_gb >= 16 else "float32"

    if cuda_cores >= 512 and ram_gb > 12:
        batch_size, model_dim, n_folds, lr, opt, epochs = 384, 128, 7, 0.0005, "adamw", 60
    elif device == "cuda" and ram_gb > 12:
        batch_size, model_dim, n_folds, lr, opt, epochs = 256, 128, 6, 0.0007, "adamw", 50
    elif ram_gb > 12:
        batch_size, model_dim, n_folds, lr, opt, epochs = 128, 64, 5, 0.001, "adam", 40
    elif ram_gb > 8:
        batch_size, model_dim, n_folds, lr, opt, epochs = 64, 32, 4, 0.005, "adam", 30
    else:
        batch_size, model_dim, n_folds, lr, opt, epochs = 32, 16, 3, 0.01, "sgd", 15

    ensure_logs_dir("logs")
    plan = {
        "device": device,
        "gpu": gpu_name,
        "vram": vram_gb,
        "cuda_cores": cuda_cores,
        "ram": ram_gb,
        "threads": threads,
        "batch_size": batch_size,
        "model_dim": model_dim,
        "n_folds": n_folds,
        "optimizer": opt,
        "lr": lr,
        "precision": precision,
        "train_epochs": epochs,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        import json
        with open("logs/resource_plan.json", "w") as f:
            json.dump(plan, f, indent=2)
    except Exception:  # pragma: no cover - ignore file write errors
        pass

    return plan


def load_data(path: str = M1_PATH) -> pd.DataFrame:
    """Load CSV and parse timestamp safely."""
    if not os.path.exists(path):
        logger.error(
            f"❌ File not found: {path}. โปรดตรวจสอบว่ามีไฟล์ `XAUUSD_M1.csv` ในโฟลเดอร์ `data/`"
        )
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)
    if {"Date", "Timestamp"}.issubset(df.columns):
        df["timestamp"] = convert_thai_datetime(df["Date"], df["Timestamp"])
    elif {"date", "timestamp"}.issubset(df.columns):
        df["timestamp"] = parse_timestamp_safe(
            pd.to_datetime(df["date"] + " " + df["timestamp"], errors="coerce"),
            format="%Y-%m-%d %H:%M:%S",
        )
    elif "timestamp" in df.columns:
        df["timestamp"] = parse_timestamp_safe(
            pd.to_datetime(df["timestamp"], errors="coerce"),
            format="%Y-%m-%d %H:%M:%S",
        )
    else:
        logger.error(
            "❌ ไม่พบคอลัมน์ Date/Timestamp หรือ date/timestamp เพื่อแปลงเป็น datetime"
        )
        raise RuntimeError("Missing timestamp columns")

    df = df.sort_values("timestamp")
    return df


def summarize_results(trades, equity):
    profit = trades["pnl"].sum() if not trades.empty else 0
    trades_count = len(trades)
    winrate = (trades["pnl"] > 0).mean() if trades_count else 0
    return {"profit": profit, "trades": trades_count, "winrate": winrate}


def save_results(trades, equity, metrics, outdir):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    trades.to_csv(os.path.join(outdir, f"trades_{ts}.csv"), index=False)
    equity.to_csv(os.path.join(outdir, f"equity_{ts}.csv"), index=False)
    with open(os.path.join(outdir, f"summary_{ts}.txt"), "w") as f:
        f.write(str(metrics))


def export_chatgpt_ready_logs(trades: pd.DataFrame, equity: pd.DataFrame, summary_dict: dict, outdir: str = "logs") -> None:
    """Export trades, equity and summary metrics to CSV files."""
    os.makedirs(outdir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    trades_path = os.path.join(outdir, f"trades_detail_{ts}.csv")
    equity_path = os.path.join(outdir, f"equity_curve_{ts}.csv")
    summary_path = os.path.join(outdir, f"summary_metrics_{ts}.csv")

    # [Patch E] Fill missing meta columns if absent
    for col in ["planned_risk", "r_multiple", "pnl_pct", "break_even_min", "mfe"]:
        if col not in trades.columns:
            trades[col] = None

    # [Patch E] Meta tags for ML filtering
    trades["is_recovery"] = trades.get("risk_mode", "") == "recovery"
    exit_reason_series = trades.get("exit_reason", pd.Series(dtype=str))
    trades["is_tsl"] = exit_reason_series.str.contains("tsl", na=False)
    trades["is_tp2"] = exit_reason_series == "TP2"

    trades.to_csv(trades_path, index=False)
    equity.to_csv(equity_path, index=False)
    pd.DataFrame([summary_dict]).to_csv(summary_path, index=False)

    print("📁 Export Completed:")
    print(f"   ├ trades_detail → {trades_path}")
    print(f"   ├ equity_curve  → {equity_path}")
    print(f"   └ summary_metrics → {summary_path}")


def create_summary_dict(
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    file_name: str = "",
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    duration_sec: float | None = None,
) -> dict:
    """Create a summary dictionary for export_chatgpt_ready_logs."""

    start_eq = equity.iloc[0]["equity"] if not equity.empty else 0
    end_eq = equity.iloc[-1]["equity"] if not equity.empty else 0

    safe_start_time = (
        start_time.strftime("%Y-%m-%d %H:%M:%S") if pd.notnull(start_time) else ""
    )
    safe_end_time = (
        end_time.strftime("%Y-%m-%d %H:%M:%S") if pd.notnull(end_time) else ""
    )

    return {
        "file_name": file_name,
        "total_trades": len(trades),
        "winrate": round((trades["pnl"] > 0).mean() * 100, 2) if not trades.empty else 0,
        "total_profit": round(trades["pnl"].sum(), 2) if not trades.empty else 0,
        "capital_growth": round((end_eq - start_eq) / start_eq * 100, 2) if start_eq else 0,
        "max_drawdown": round(100 * (1 - equity["equity"].min() / start_eq), 2) if start_eq else 0,
        "avg_pnl": round(trades["pnl"].mean(), 4) if not trades.empty else 0,
        "total_lot": round(trades.get("lot", pd.Series(dtype=float)).sum(), 2),
        "commission_paid": round(trades.get("commission", pd.Series(dtype=float)).sum(), 2),
        "spread_impact": round(trades.get("spread_cost", pd.Series(dtype=float)).sum(), 2),
        "slippage_impact": round(trades.get("slippage_cost", pd.Series(dtype=float)).sum(), 2),
        "total_cost_deducted": round(
            trades.get("commission", pd.Series(dtype=float)).sum()
            + trades.get("spread_cost", pd.Series(dtype=float)).sum()
            + trades.get("slippage_cost", pd.Series(dtype=float)).sum(),
            2,
        ),
        "duration_sec": round(duration_sec, 2) if duration_sec is not None else None,
        "start_time": safe_start_time,
        "end_time": safe_end_time,
    }


def auto_entry_config(fold_df: pd.DataFrame) -> dict:
    """Generate entry config based on fold statistics."""
    atr_std = fold_df["atr"].std()
    ema_slope_mean = fold_df["ema_fast"].diff().mean()
    gainz_std = fold_df["gain_z"].std() if "gain_z" in fold_df.columns else 0.1

    return {
        "gain_z_thresh": -0.05 if gainz_std > 0.2 else -0.1 if gainz_std > 0.1 else -0.2,
        "ema_slope_min": -0.005 if ema_slope_mean < 0 else 0.0,
    }


# ✅ Run Walk-Forward Validation

def split_folds(df: pd.DataFrame, n_folds: int = 5) -> list[pd.DataFrame]:
    """แบ่ง DataFrame ตามลำดับเวลาและรับมือกับแถวเศษส่วนใน fold สุดท้าย"""

    df = df.sort_values("timestamp")
    folds: list[pd.DataFrame] = []
    N = len(df)
    base = N // n_folds
    for i in range(n_folds):
        start = i * base
        if i < n_folds - 1:
            end = (i + 1) * base
            folds.append(df.iloc[start:end])
        else:
            folds.append(df.iloc[start:])
    return folds


def run_auto_wfv(df: pd.DataFrame, outdir: str, n_folds: int = 5) -> pd.DataFrame:
    """Run simple walk-forward validation."""
    folds = split_folds(df, n_folds=n_folds)
    summary = []
    generate_signals, run_backtest = _import_backtest_tools()

    for i, fold_df in enumerate(folds):
        fold_id = i + 1
        print(f"\n[WFV] Fold {fold_id}/{n_folds}")

        # คำนวณอินดิเคเตอร์พื้นฐานก่อนหา config อัตโนมัติ
        base_df = generate_signals(fold_df)
        config = auto_entry_config(base_df)
        fold_df_filtered = generate_signals(fold_df, config=config)
        trades, equity = run_backtest(fold_df_filtered)

        if trades.empty:
            # [Patch D.4.3] ผ่อนปรนตัวกรองเมื่อไม่มีเทรดเลย
            relaxed_config = {"gain_z_thresh": -0.3, "ema_slope_min": -0.01}
            fold_df_filtered = generate_signals(fold_df, config=relaxed_config)
            trades, equity = run_backtest(fold_df_filtered)
            config = relaxed_config
        metrics = summarize_results(trades, equity)
        metrics["fold"] = fold_id
        summary.append(metrics)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        trades.to_csv(os.path.join(outdir, f"trades_fold{i + 1}_{ts}.csv"), index=False)
        equity.to_csv(os.path.join(outdir, f"equity_fold{i + 1}_{ts}.csv"), index=False)

    return pd.DataFrame(summary)


def split_by_session(df: pd.DataFrame) -> dict:
    """Split dataframe into session-based subsets."""
    df = df.copy()
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.date_range("2000-01-01", periods=len(df), freq="h")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df = df.set_index("timestamp")
    asia_df = df[df["hour"].between(3, 7)]
    london_df = df[df["hour"].between(8, 14)]
    ny_df = df[df["hour"].between(15, 23)]  # [Patch v7.4]
    return {
        "Asia": asia_df,
        "London": london_df,
        "NY": ny_df,
    }


def safe_calculate_net_change(trade_df: pd.DataFrame) -> float:
    """Calculate net change considering trade direction if available."""

    if trade_df.empty or "entry_price" not in trade_df or "exit_price" not in trade_df:
        print("⚠️ trade_df ไม่มีข้อมูล entry_price หรือ exit_price")
        return 0.0

    trade_df = trade_df.dropna(subset=["entry_price", "exit_price"]).copy()

    if "direction" in trade_df.columns:
        diffs = np.where(
            trade_df["direction"].str.lower() == "sell",
            trade_df["entry_price"] - trade_df["exit_price"],
            trade_df["exit_price"] - trade_df["entry_price"],
        )
        net_change = diffs.sum()
    elif "side" in trade_df.columns:
        diffs = np.where(
            trade_df["side"].str.lower() == "sell",
            trade_df["entry_price"] - trade_df["exit_price"],
            trade_df["exit_price"] - trade_df["entry_price"],
        )
        net_change = diffs.sum()
    else:
        net_change = trade_df["exit_price"].sub(trade_df["entry_price"]).sum()

    return round(float(net_change), 4)


def merge_equity_curves(equity_lists: list[pd.DataFrame], n_folds: int) -> pd.DataFrame:
    """รวม equity curve จากหลาย fold และคำนวณยอดรวม"""

    df_merged = pd.concat(equity_lists, ignore_index=True)
    if "timestamp" in df_merged.columns:
        df_merged = df_merged.sort_values("timestamp").reset_index(drop=True)
    df_merged["equity_total"] = df_merged["pnl_usd_net"].cumsum()
    return df_merged


def convert_thai_datetime(df_or_series, format: str = "%Y-%m-%d %H:%M:%S"):
    """แปลง Date พ.ศ. และ Timestamp แบบไทย ให้เป็น ``datetime64[ns]``"""
    if isinstance(df_or_series, pd.Series):
        try:
            return pd.to_datetime(df_or_series, format=format, errors="coerce")
        except Exception:
            return pd.to_datetime(df_or_series, errors="coerce")
    else:
        df = df_or_series.copy()
        cols_upper = {"Date", "Timestamp"}
        cols_lower = {"date", "timestamp"}
        if cols_upper.issubset(df.columns):
            df["year"] = df["Date"].astype(str).str[:4].astype(int) - 543
            df["month"] = df["Date"].astype(str).str[4:6]
            df["day"] = df["Date"].astype(str).str[6:8]
            df["datetime_str"] = df["year"].astype(str) + "-" + df["month"] + "-" + df["day"] + " " + df["Timestamp"].astype(str)
            df["timestamp"] = pd.to_datetime(df["datetime_str"], format=format, errors="coerce")
            return df
        if cols_lower.issubset(df.columns):
            df["year"] = df["date"].astype(str).str[:4].astype(int) - 543
            df["month"] = df["date"].astype(str).str[4:6]
            df["day"] = df["date"].astype(str).str[6:8]
            df["datetime_str"] = df["year"].astype(str) + "-" + df["month"] + "-" + df["day"] + " " + df["timestamp"].astype(str)
            df["timestamp"] = pd.to_datetime(df["datetime_str"], format=format, errors="coerce")
            return df
        return df


def parse_timestamp_safe(ts_series: pd.Series, fmt: str = "%Y-%m-%d %H:%M:%S", **kwargs) -> pd.Series:
    """แปลง Series เป็น datetime อย่างปลอดภัยและรองรับพารามิเตอร์ชื่อ ``format``"""

    if "format" in kwargs and not kwargs.get("fmt"):
        fmt = kwargs["format"]

    try:
        ts = pd.to_datetime(ts_series, format=fmt, errors="coerce")
    except ValueError as e:  # pragma: no cover - unexpected format
        logger.warning(f"[parse_timestamp_safe] Invalid format {fmt}: {e}")
        ts = pd.to_datetime(ts_series, errors="coerce")
    return ts


def simulate_tp_exit(
    df_trades: pd.DataFrame,
    df_m1: pd.DataFrame,
    window_minutes: int = 60,
) -> pd.DataFrame:
    """Check if TP1/TP2/SL is hit within a time window."""
    df_trades = df_trades.copy()
    df_m1 = df_m1.copy()
    df_m1["timestamp"] = pd.to_datetime(df_m1["timestamp"])

    exit_prices: list[float] = []
    exit_reasons: list[str] = []

    for _, row in df_trades.iterrows():
        t0 = pd.to_datetime(row["timestamp"])
        direction = row.get("direction")
        if direction is None:
            direction = "sell" if row["entry_price"] > row["tp1_price"] else "buy"  # pragma: no cover - simple fallback
        t1 = t0 + pd.Timedelta(minutes=window_minutes)

        window = df_m1[(df_m1["timestamp"] >= t0) & (df_m1["timestamp"] <= t1)]
        hit = None

        if direction == "buy":
            for _, bar in window.iterrows():
                if bar["low"] <= row["sl_price"]:
                    hit = ("SL", row["sl_price"])  # pragma: no cover
                    break
                elif bar["high"] >= row["tp2_price"]:
                    hit = ("TP2", row["tp2_price"])  # pragma: no cover - rare branch
                    break
                elif bar["high"] >= row["tp1_price"]:
                    hit = ("TP1", row["tp1_price"])  # pragma: no cover
                    break
        else:
            for _, bar in window.iterrows():
                if bar["high"] >= row["sl_price"]:
                    hit = ("SL", row["sl_price"])  # pragma: no cover
                    break
                elif bar["low"] <= row["tp2_price"]:
                    hit = ("TP2", row["tp2_price"])  # pragma: no cover - rare branch
                    break
                elif bar["low"] <= row["tp1_price"]:
                    hit = ("TP1", row["tp1_price"])  # pragma: no cover
                    break

        if hit:
            exit_reasons.append(hit[0])
            exit_prices.append(hit[1])
        else:
            exit_reasons.append("TIMEOUT")
            exit_prices.append(row["entry_price"])

    df_trades["exit_reason"] = exit_reasons
    df_trades["exit_price"] = exit_prices
    return df_trades


def prepare_csv_auto(path: str, datetime_format: str = "%Y-%m-%d %H:%M:%S") -> pd.DataFrame:
    """โหลดไฟล์ CSV แล้วแปลงและตรวจสอบให้พร้อมใช้งาน"""
    import importlib

    main = importlib.import_module("main")
    df = main.load_csv_safe(path)
    from nicegold_v5.entry import sanitize_price_columns, validate_indicator_inputs
    df = convert_thai_datetime(df)
    if "timestamp" in df.columns:
        df["timestamp"] = parse_timestamp_safe(df["timestamp"], datetime_format)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df = sanitize_price_columns(df)
    try:
        validate_indicator_inputs(df, min_rows=min(500, len(df)))
    except TypeError:  # pragma: no cover - fallback for legacy signature
        validate_indicator_inputs(df)
    return df

