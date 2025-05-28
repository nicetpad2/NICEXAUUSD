import pandas as pd
from nicegold_v5.entry import generate_signals_v8_0


def test_entry_tier_handles_small_df():
    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=1, freq='min'),
        'open': [1],
        'high': [1],
        'low': [1],
        'close': [1],
        'volume': [1],
    })
    out = generate_signals_v8_0(df)
    assert 'entry_tier' in out.columns
    assert out['entry_tier'].iloc[0] == 'C'
