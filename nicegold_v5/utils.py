import pandas as pd
import os
from datetime import datetime


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
