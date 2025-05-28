import pandas as pd
import pytest
from nicegold_v5.entry import generate_signals_v8_0


def test_entry_blocked_reason_length_check(monkeypatch):
    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=3, freq='min'),
        'open': [1, 1, 1],
        'high': [1, 1, 1],
        'low': [1, 1, 1],
        'close': [1, 1, 1],
        'volume': [1, 1, 1],
    })

    original_reset_index = pd.Series.reset_index

    def fake_reset_index(self, level=None, drop=False, name=None, inplace=False):
        result = original_reset_index(self, level=level, drop=drop, name=name, inplace=False)
        return result.iloc[:-1]

    monkeypatch.setattr(pd.Series, "reset_index", fake_reset_index)
    with pytest.raises(ValueError):
        generate_signals_v8_0(df)

