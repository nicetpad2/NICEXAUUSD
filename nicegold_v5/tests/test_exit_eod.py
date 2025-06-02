import pandas as pd
from nicegold_v5.exit import simulate_partial_tp_safe


def test_simulate_partial_tp_safe_eod_exit():
    ts = pd.date_range('2025-01-01', periods=3, freq='min')
    df = pd.DataFrame({
        'timestamp': ts,
        'close': [100.0, 100.5, 100.7],
        'high': [100.0, 100.5, 100.7],
        'low': [100.0, 100.4, 100.6],
        'atr': [1.0, 1.0, 1.0],
        'entry_signal': [None, 'buy', None],
        'session': ['Asia', 'Asia', 'Asia'],
    })
    trades = simulate_partial_tp_safe(df)
    assert not trades.empty
    last_trade = trades.iloc[-1]
    assert last_trade['exit_reason'] == 'eod_exit'
    assert last_trade['exit_price'] == df.iloc[-1]['close']
