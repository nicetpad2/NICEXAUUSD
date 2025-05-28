import pandas as pd
from nicegold_v5.entry import (
    apply_tp_logic,
    generate_entry_signal,
    session_filter,
    trade_log_fields,
)


def test_apply_tp_logic():
    tp1, tp2 = apply_tp_logic(100, 'buy', rr1=1.5, rr2=3.0, sl_distance=5)
    assert tp1 == 107.5
    assert tp2 == 115.0


def test_generate_entry_signal():
    logs = []
    row = {
        'rsi': 25,
        'pattern': 'inside_bar',
        'timestamp': pd.Timestamp('2025-01-01'),
        'close': 100.0,
        'session': 'London',
    }
    signal = generate_entry_signal(row, logs)
    assert signal == 'RSI_InsideBar'
    assert logs and logs[0]['signal'] == 'RSI_InsideBar'


def test_session_filter():
    row_block = {'session': 'NY', 'ny_sl_count': 4}
    row_allow = {'session': 'NY', 'ny_sl_count': 1}
    assert not session_filter(row_block)
    assert session_filter(row_allow)


def test_trade_log_fields():
    required = {'tp1_price', 'tp2_price', 'mfe', 'duration_min'}
    assert required.issubset(set(trade_log_fields))


def test_simulate_trades_with_tp():
    from nicegold_v5.entry import simulate_trades_with_tp

    df = pd.DataFrame([
        {
            'timestamp': pd.Timestamp('2025-01-01 00:00:00'),
            'close': 100.0,
            'signal': 'long',
            'session': 'London',
            'rsi': 25,
            'pattern': 'inside_bar',
        }
    ])

    trades, logs = simulate_trades_with_tp(df)
    assert len(trades) == 1
    assert len(logs) == 1
    trade = trades[0]
    assert trade['tp1_price'] > trade['entry_price']
    assert trade['tp2_price'] > trade['tp1_price']
