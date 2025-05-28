import os
import pandas as pd
from nicegold_v5.entry import generate_signals_v11_scalper_m1
from nicegold_v5.config import SNIPER_CONFIG_Q3_TUNED
from nicegold_v5.backtester import run_backtest
from nicegold_v5.utils import print_qa_summary, export_chatgpt_ready_logs

TRADE_DIR = "/content/drive/MyDrive/NICEGOLD/logs"
M1_PATH = "/content/drive/MyDrive/NICEGOLD/XAUUSD_M1.csv"
os.makedirs(TRADE_DIR, exist_ok=True)


def strip_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that could leak future information."""
    leak_cols = [
        c for c in df.columns
        if "future" in c or "next_" in c or c.endswith("_lead") or c == "target"
    ]
    return df.drop(columns=leak_cols, errors="ignore")


def run_clean_exit_backtest():
    """Run backtest using real exit logic without future leakage."""
    df = pd.read_csv(M1_PATH)
    df.columns = [c.lower() for c in df.columns]
    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp")

    df = generate_signals_v11_scalper_m1(df, config=SNIPER_CONFIG_Q3_TUNED)
    df["entry_time"] = df["timestamp"]
    df["signal_id"] = df["timestamp"].astype(str)
    df = strip_leakage_columns(df)

    # [Patch F.1] Enable Break-even and Trailing SL Logic
    df["use_be"] = True
    df["use_tsl"] = True
    df["tp1_rr_ratio"] = 2.0  # TP1 used to trigger BE
    df["use_dynamic_tsl"] = True  # TSL adjusts dynamically based on RR threshold

    if "exit_reason" in df.columns:
        df.drop(columns=["exit_reason"], inplace=True)

    print("\U0001F680 Running Backtest with Clean Exit + BE/TSL Protection...")
    trades, equity = run_backtest(df)
    print_qa_summary(trades, equity)

    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_chatgpt_ready_logs(
        trades,
        equity,
        {
            "file_name": os.path.basename(M1_PATH),
            "total_trades": len(trades),
            "total_profit": trades["pnl"].sum(),
        },
        outdir=TRADE_DIR,
    )
    print(f"\U0001F4E6 Export Completed: trades_detail_{ts}.csv")
    return trades


if __name__ == "__main__":
    run_clean_exit_backtest()
