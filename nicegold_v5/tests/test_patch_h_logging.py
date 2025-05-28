import pandas as pd
from nicegold_v5.backtester import (
    calculate_duration,
    calculate_mfe,
    calculate_planned_risk,
)


def test_calculate_duration():
    start = pd.Timestamp('2025-01-01 00:00:00')
    end = pd.Timestamp('2025-01-01 00:05:30')
    assert calculate_duration(start, end) == 5.5


def test_calculate_planned_risk():
    risk = calculate_planned_risk(100.0, 99.0, 0.02)
    assert risk == 0.2


def test_calculate_mfe():
    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=5, freq='min'),
        'high': [1, 2, 3, 4, 5],
        'low': [0, 1, 2, 3, 4],
    })
    start = df['timestamp'].iloc[1]
    end = df['timestamp'].iloc[3]
    mfe_buy = calculate_mfe(start, end, df, 1.5, 'buy')
    mfe_sell = calculate_mfe(start, end, df, 3.5, 'sell')
    assert mfe_buy == 2.5
    assert mfe_sell == 2.5
