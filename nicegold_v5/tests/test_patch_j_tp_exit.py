import pandas as pd
from nicegold_v5.patch_j_tp1tp2_simulation import simulate_tp_exit


def test_simulate_tp_exit():
    trades = pd.DataFrame([
        {
            'timestamp': pd.Timestamp('2025-01-01 00:00:00'),
            'entry_price': 100.0,
            'tp1_price': 101.0,
            'tp2_price': 102.0,
            'sl_price': 99.0,
            'direction': 'buy',
        },
        {
            'timestamp': pd.Timestamp('2025-01-01 00:10:00'),
            'entry_price': 100.0,
            'tp1_price': 99.0,
            'tp2_price': 98.0,
            'sl_price': 101.0,
            'direction': 'sell',
        },
    ])

    m1 = pd.DataFrame([
        {'timestamp': '2025-01-01 00:00:00', 'high': 100.5, 'low': 99.5},
        {'timestamp': '2025-01-01 00:01:00', 'high': 101.5, 'low': 100.0},
        {'timestamp': '2025-01-01 00:10:00', 'high': 100.5, 'low': 99.5},
        {'timestamp': '2025-01-01 00:11:00', 'high': 100.8, 'low': 97.5},
    ])

    result = simulate_tp_exit(trades, m1, window_minutes=2)

    assert result.loc[0, 'exit_reason'] == 'TP1'
    assert result.loc[1, 'exit_reason'] == 'TP2'
