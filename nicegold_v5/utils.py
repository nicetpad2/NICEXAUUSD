import pandas as pd
import os
from datetime import datetime
from nicegold_v5.entry import generate_signals
from nicegold_v5.backtester import run_backtest

# ✅ Fixed Paths for Colab
TRADE_DIR = "/content/drive/MyDrive/NICEGOLD/logs"
os.makedirs(TRADE_DIR, exist_ok=True)
M1_PATH = "/content/drive/MyDrive/NICEGOLD/XAUUSD_M1.csv"
M15_PATH = "/content/drive/MyDrive/NICEGOLD/XAUUSD_M15.csv"


def load_data(path):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
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
        print(f"\n[WFV] Fold {i + 1}/{n_folds}")
        fold_df = generate_signals(fold_df)
        trades, equity = run_backtest(fold_df)
        metrics = summarize_results(trades, equity)
        metrics["fold"] = i + 1
        summary.append(metrics)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        trades.to_csv(os.path.join(outdir, f"trades_fold{i + 1}_{ts}.csv"), index=False)
        equity.to_csv(os.path.join(outdir, f"equity_fold{i + 1}_{ts}.csv"), index=False)

    return pd.DataFrame(summary)
