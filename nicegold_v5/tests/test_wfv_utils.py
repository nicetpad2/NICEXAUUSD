import pandas as pd
from datetime import datetime, timedelta

from nicegold_v5.wfv import (
    calculate_position_size,
    entry_decision,
    exceeded_order_duration,
    session_label,
    merge_equity_curves,
    streak_summary,
    apply_order_costs,
    COMMISSION_PER_LOT,
)


def test_calculate_position_size_normal():
    lot = calculate_position_size(10000, 50)
    assert lot == 0.2


def test_calculate_position_size_zero():
    assert calculate_position_size(10000, 0) == 0.0


def test_entry_decision():
    assert entry_decision(0.6, 0.5)
    assert not entry_decision(0.4, 0.5)


def test_exceeded_order_duration():
    now = datetime.now()
    assert not exceeded_order_duration(now - timedelta(minutes=100), now)
    assert exceeded_order_duration(now - timedelta(minutes=121), now)


def test_session_label():
    assert session_label(pd.Timestamp('2024-01-01 01:00')) == 'Asia'
    assert session_label(pd.Timestamp('2024-01-01 10:00')) == 'London'
    assert session_label(pd.Timestamp('2024-01-01 16:00')) == 'NY'
    assert session_label(pd.Timestamp('2024-01-01 23:00')) == 'Off'


def test_merge_equity_curves_and_streak_summary(monkeypatch):
    monkeypatch.setattr('numpy.random.uniform', lambda a, b: 0.0)
    df1 = pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=2, freq='h'),
        'pnl': [1, -0.5],
        'win_streak': [1, 0],
        'loss_streak': [0, 1],
        'drawdown': [0.1, 0.2],
    })
    df2 = pd.DataFrame({
        'time': pd.date_range('2024-01-01 02:00', periods=2, freq='h'),
        'pnl': [0.5, 1],
        'win_streak': [2, 3],
        'loss_streak': [0, 0],
        'drawdown': [0.05, 0.1],
    })
    merged = merge_equity_curves(df1, df2)
    assert 'equity_total' in merged.columns
    assert merged.index.name == 'time'
    summary = streak_summary(pd.concat([df1, df2], ignore_index=True))
    assert summary['max_win_streak'] == 3
    assert summary['max_loss_streak'] == 1
    assert summary['max_drawdown'] == 0.2


def test_apply_order_costs(monkeypatch):
    monkeypatch.setattr('numpy.random.uniform', lambda a, b: 0.05)
    entry_adj, sl, tp1, tp2, commission = apply_order_costs(100, 90, 110, 120, 0.1, 'buy')
    assert entry_adj > 100
    assert commission == 2 * COMMISSION_PER_LOT * 0.1 * 100
