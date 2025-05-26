import pandas as pd
import os
from datetime import datetime
from nicegold_v5.entry import generate_signals
from nicegold_v5.backtester import run_backtest


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
TRADE_DIR = "/content/drive/MyDrive/NICEGOLD/logs"
os.makedirs(TRADE_DIR, exist_ok=True)
M1_PATH = "/content/drive/MyDrive/NICEGOLD/XAUUSD_M1.csv"
M15_PATH = "/content/drive/MyDrive/NICEGOLD/XAUUSD_M15.csv"


def load_data(path):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )
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
    """Split dataframe into equal sequential folds."""
    fold_size = len(df) // n_folds
    return [
        df.iloc[i * fold_size:(i + 1) * fold_size].reset_index(drop=True)
        for i in range(n_folds)
    ]


def run_auto_wfv(df: pd.DataFrame, outdir: str, n_folds: int = 5) -> pd.DataFrame:
    """Run simple walk-forward validation."""
    folds = split_folds(df, n_folds=n_folds)
    summary = []

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
        df["timestamp"] = pd.date_range("2000-01-01", periods=len(df), freq="H")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df = df.set_index("timestamp")
    asia_df = df[df["hour"].between(3, 7)]
    london_df = df[df["hour"].between(8, 15)]
    ny_df = df[df["hour"].between(16, 22)]
    return {
        "Asia": asia_df,
        "London": london_df,
        "NY": ny_df,
    }
