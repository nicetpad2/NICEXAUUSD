import argparse
import os

from nicegold_v5.utils import load_data, summarize_results, save_results
from nicegold_v5.entry import generate_signals
from nicegold_v5.backtester import run_backtest


def main(data_path: str, outdir: str = "results"):
    df = load_data(data_path)
    df = generate_signals(df)
    trades, equity = run_backtest(df)
    metrics = summarize_results(trades, equity)
    os.makedirs(outdir, exist_ok=True)
    save_results(trades, equity, metrics, outdir)
    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()
    main(args.data_path, args.outdir)
