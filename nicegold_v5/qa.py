import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
import logging
from nicegold_v5.utils import export_audit_report
from nicegold_v5.entry import validate_indicator_inputs, simulate_partial_tp_safe
from nicegold_v5.utils import ensure_buy_sell  # [Patch QA-FIX v28.2.5] Forward for QA

# module logger
logger = logging.getLogger("nicegold_v5.qa")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# --- QA Guard Functions ---

def detect_overfit_bias(trades: pd.DataFrame) -> dict:
    """Return simple statistics to gauge overfitting."""
    if trades.empty:
        return {
            "winrate": 0.0,
            "avg_pnl": 0.0,
            "pnl_std": 0.0,
            "pnl_zscore": 0.0,
            "overfit_score": 0.0,
        }
    winrate = trades["exit_reason"].value_counts().get("tp2", 0) / max(len(trades), 1)
    pnl_std = trades["pnl"].std(ddof=0)
    avg_pnl = trades["pnl"].mean()
    pnl_zscore = avg_pnl / (pnl_std + 1e-9)
    score = round((pnl_zscore * winrate) * 100, 2)
    return {
        "winrate": round(winrate * 100, 2),
        "avg_pnl": round(avg_pnl, 2),
        "pnl_std": round(pnl_std, 2),
        "pnl_zscore": round(pnl_zscore, 2),
        "overfit_score": score,
    }


def detect_noise_exit(trades: pd.DataFrame) -> pd.DataFrame:
    """Find trades exited too quickly or with low MFE while losing."""
    if 'mfe' not in trades.columns or 'duration_min' not in trades.columns:
        return pd.DataFrame()
    suspected = trades.copy()
    suspected = suspected[(suspected['mfe'] < 2.0) | (suspected['duration_min'] < 2)]
    suspected = suspected[suspected['pnl'] < 0]
    cols = ['_id', 'entry_time', 'pnl', 'mfe', 'duration_min', 'exit_reason']
    return suspected[[c for c in cols if c in suspected.columns]]


def detect_leakage_columns(df: pd.DataFrame) -> list:
    """Detect columns that likely leak future information."""
    return [
        c for c in df.columns
        if 'future' in c or 'next_' in c or c.endswith('_lead') or c == 'target'
    ]


def run_qa_guard(trades: pd.DataFrame, df_features: pd.DataFrame) -> None:
    """Print QA guard information about overfitting, noise and leakage."""
    print("\n🔎 [Patch G] QA Guard – วิเคราะห์ Overfitting / Noise / Data Leak")
    print("\n📊 Overfitting Score:")
    score = detect_overfit_bias(trades)
    score_clean = {k: float(v) for k, v in score.items()}
    print(score_clean)

    print("\n⚠️ Noise Exit Suspicion:")
    noisy = detect_noise_exit(trades)
    print(noisy.head(5) if not noisy.empty else "✅ ไม่พบ exit น่าสงสัย")

    print("\n🧯 Leak Columns:")
    leaks = detect_leakage_columns(df_features)
    if leaks:
        print("❌ พบคอลัมน์ต้องสงสัย:", leaks)
    else:
        print("✅ ไม่พบคอลัมน์ที่มีลักษณะ Data Leakage")


# --- Stability Layer ---

def summarize_fold(trades: pd.DataFrame, fold_name: str = "Fold") -> dict:
    """Return statistics summary for a fold."""
    if "exit_reason" not in trades.columns:
        logger.warning("[summarize_fold] Missing exit_reason")
        trades["exit_reason"] = ""
    wins = trades[trades["pnl"] > 0]
    return {
        "fold": fold_name,
        "trades": len(trades),
        "winrate": round(len(wins) / len(trades) * 100, 2) if len(trades) > 0 else 0,
        "avg_pnl": round(trades["pnl"].mean(), 2) if not trades.empty else 0,
        "max_loss": round(trades["pnl"].min(), 2) if not trades.empty else 0,
        "sl_count": trades.get("exit_reason", pd.Series(dtype=str)).fillna("").str.contains("sl").sum(),
        "be_count": trades.get("exit_reason", pd.Series(dtype=str)).fillna("").str.contains("be").sum(),
        "tsl_count": trades.get("exit_reason", pd.Series(dtype=str)).fillna("").str.contains("tsl").sum(),
        "tp1_count": (trades.get("exit_reason", pd.Series(dtype=str)).fillna("") == "tp1").sum(),
        "tp2_count": (trades.get("exit_reason", pd.Series(dtype=str)).fillna("") == "tp2").sum(),
    }


def compute_fold_bias(trades: pd.DataFrame) -> float:
    """Compute bias score for a fold."""
    if trades.empty:
        return -999.0
    pnl_std = trades['pnl'].std()
    avg_pnl = trades['pnl'].mean()
    winrate = (trades['pnl'] > 0).mean()
    zscore = avg_pnl / (pnl_std + 1e-9)
    bias_score = round(zscore * winrate * 100, 2)
    return bias_score


def analyze_drawdown(equity_df: pd.DataFrame) -> dict:
    """Return maximum drawdown percentage from equity curve."""
    if 'equity' not in equity_df.columns:
        return {'max_drawdown_pct': 'N/A'}
    equity = equity_df['equity'].ffill()
    peak = equity.expanding().max()
    dd = (equity - peak) / peak
    max_dd = dd.min()
    return {'max_drawdown_pct': round(abs(max_dd) * 100, 2)}


# --- Export & Auto QA ---

def export_fold_qa(
    fold_name: str,
    stats: dict,
    bias_score: float,
    drawdown: dict,
    outdir: str = "logs/qa",
):
    """Export fold QA summary and related metrics as JSON."""
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"fold_qa_{fold_name.lower()}.json")
    payload = {
        "fold": fold_name,
        "qa_summary": stats,
        "overfit_bias_score": bias_score,
        "drawdown": drawdown,
    }
    with open(outpath, "w") as f:
        json.dump(payload, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else o)
    print(f"📁 Exported QA summary → {outpath}")


def detect_fold_drift(trades: pd.DataFrame) -> dict:
    """Detect drift in a single fold of trades."""
    if "pnl" not in trades.columns:
        return {"pnl_mean": 0.0, "pnl_std": 0.0, "pct_std": 0.0}
    pnl = trades["pnl"]
    mean = pnl.mean() if not pnl.empty else 0
    std = pnl.std(ddof=0) if not pnl.empty else 0
    if mean == 0:
        pct_std = 0.0
    else:
        pct_std = std / abs(mean)
    return {"pnl_mean": mean, "pnl_std": std, "pct_std": pct_std}


QA_BASE_PATH = "logs/qa"
os.makedirs(QA_BASE_PATH, exist_ok=True)


def auto_qa_after_backtest(trades: pd.DataFrame, equity: pd.DataFrame, label: str = "Run"):
    """Automatically export QA summary after backtest with timestamp-enhanced label."""
    if trades.empty:
        print("❌ ไม่มีข้อมูล trades สำหรับ QA")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label_full = f"{label}_{timestamp}"

    stats = summarize_fold(trades, label_full)
    bias = compute_fold_bias(trades)
    dd = analyze_drawdown(equity)
    export_fold_qa(label_full, stats, bias, dd, outdir=QA_BASE_PATH)

    save_csv_path = os.path.join(QA_BASE_PATH, f"fold_qa_{label_full.lower()}.csv")
    pd.DataFrame([stats | {"bias_score": bias} | dd]).to_csv(save_csv_path, index=False)
    print(f"✅ QA Auto Export → {save_csv_path}")

    export_audit_report(
        config={},
        metrics=stats | {"bias_score": bias} | dd,
        run_type="QA",
        version="v32.1.0",
        fold=None,
        outdir=QA_BASE_PATH,
    )


def force_entry_stress_test(
    df: pd.DataFrame,
    config: dict,
    audit_prefix: str = "QA_FORCEENTRY",
    log_dir: str = "logs/",
) -> dict:
    """Run ForceEntry Stress Test by opening a trade on every bar."""
    os.makedirs(log_dir, exist_ok=True)
    df = df.copy()
    validate_indicator_inputs(df, min_rows=min(10, len(df)))
    df["entry_signal"] = "force_buy"
    trades_df = simulate_partial_tp_safe(df)

    equity = trades_df["pnl"].cumsum()
    peak = equity.cummax()
    trades_df["drawdown"] = (
        ((peak - equity) / peak.replace(0, np.nan))
        .fillna(0.0)
    )

    audit = {
        "run_type": "QA_FORCEENTRY",
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M"),
        "config": config,
        "metrics": {
            "total_trades": len(trades_df),
            "max_drawdown": float(trades_df["drawdown"].max()) if not trades_df.empty else 0.0,
            "total_pnl": float(trades_df["pnl"].sum()) if not trades_df.empty else 0.0,
            "sl_count": int((trades_df.get("exit_reason") == "sl").sum()),
        },
        "version": config.get("version", "dev"),
        "commit_hash": config.get("commit_hash", "manual"),
    }

    csv_path = os.path.join(log_dir, f"{audit_prefix}_{audit['timestamp']}.csv")
    json_path = os.path.join(log_dir, f"{audit_prefix}_{audit['timestamp']}.json")
    trades_df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"[QA] Exported audit files → {csv_path}, {json_path}")
    return audit
