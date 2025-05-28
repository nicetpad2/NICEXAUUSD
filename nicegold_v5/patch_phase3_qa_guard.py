import os
import json
from datetime import datetime
import pandas as pd
import numpy as np


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
    print(detect_overfit_bias(trades))

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
