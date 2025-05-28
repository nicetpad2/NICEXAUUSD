import os
import json
from datetime import datetime
import pandas as pd
import numpy as np

# --- QA Guard Functions ---

def detect_overfit_bias(trades: pd.DataFrame) -> dict:
    """Return simple statistics to gauge overfitting."""
    winrate = (trades['pnl'] > 0).mean() if not trades.empty else 0.0
    pnl_std = trades['pnl'].std() if not trades.empty else 0.0
    avg_pnl = trades['pnl'].mean() if not trades.empty else 0.0
    pnl_zscore = avg_pnl / (pnl_std + 1e-9)
    score = round((pnl_zscore * winrate) * 100, 2)
    return {
        'winrate': round(winrate * 100, 2),
        'avg_pnl': round(avg_pnl, 2),
        'pnl_std': round(pnl_std, 2),
        'pnl_zscore': round(pnl_zscore, 2),
        'overfit_score': score,
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
    wins = trades[trades['pnl'] > 0]
    losses = trades[trades['pnl'] < 0]
    return {
        'fold': fold_name,
        'trades': len(trades),
        'winrate': round(len(wins) / len(trades) * 100, 2) if len(trades) > 0 else 0,
        'avg_pnl': round(trades['pnl'].mean(), 2) if not trades.empty else 0,
        'max_loss': round(trades['pnl'].min(), 2) if not trades.empty else 0,
        'sl_count': trades.get('exit_reason', pd.Series(dtype=str)).fillna('').str.contains('sl').sum(),
        'be_count': trades.get('exit_reason', pd.Series(dtype=str)).fillna('').str.contains('be').sum(),
        'tsl_count': trades.get('exit_reason', pd.Series(dtype=str)).fillna('').str.contains('tsl').sum(),
        'tp1_count': (trades.get('exit_reason', pd.Series(dtype=str)).fillna('') == 'tp1').sum(),
        'tp2_count': (trades.get('exit_reason', pd.Series(dtype=str)).fillna('') == 'tp2').sum(),
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

def export_fold_qa(fold_name: str, stats: dict, bias_score: float, drawdown: dict, outdir: str = "logs"):
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


def detect_fold_drift(fold_stats: list[dict]) -> pd.DataFrame:
    """Detect metric variations across folds."""
    df = pd.DataFrame(fold_stats)
    if len(df) < 2:
        return pd.DataFrame()
    metrics = ["winrate", "avg_pnl", "sl_count"]
    drifts = []
    for m in metrics:
        std = df[m].std()
        mean = df[m].mean()
        pct_std = round((std / abs(mean)) * 100, 2) if mean != 0 else 0.0
        drifts.append({"metric": m, "std_dev": round(std, 2), "%_variation": pct_std})
    return pd.DataFrame(drifts)


QA_BASE_PATH = "/content/drive/MyDrive/NICEGOLD/logs/qa"
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
