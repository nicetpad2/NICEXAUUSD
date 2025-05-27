import pandas as pd
from nicegold_v5.entry import generate_signals_v8_0
from nicegold_v5.config import SNIPER_CONFIG_DIAGNOSTIC
from nicegold_v5.backtester import run_backtest


def run_diagnostic_backtest(df: pd.DataFrame, batch_size: int = 50000):
    """Run backtest in batches using diagnostic sniper config."""
    df_signals = generate_signals_v8_0(df, config=SNIPER_CONFIG_DIAGNOSTIC)

    batches = [
        df_signals.iloc[i : i + batch_size].reset_index(drop=True)
        for i in range(0, len(df_signals), batch_size)
    ]

    all_trades, all_equity = [], []
    for i, batch_df in enumerate(batches):
        print(f"\U0001F4E6 Running Batch {i+1}/{len(batches)}")
        trades, equity = run_backtest(batch_df)
        if not trades.empty:
            all_trades.append(trades)
            all_equity.append(equity)

    if not all_trades:
        return pd.DataFrame(), pd.DataFrame()

    combined_trades = pd.concat(all_trades, ignore_index=True)
    combined_equity = pd.concat(all_equity, ignore_index=True)
    return combined_trades, combined_equity
