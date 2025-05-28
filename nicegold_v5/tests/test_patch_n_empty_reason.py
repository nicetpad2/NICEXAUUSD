import pandas as pd
from nicegold_v5.entry import generate_signals_v8_0


def test_entry_blocked_reason_empty_df():
    df = pd.DataFrame({
        'timestamp': pd.to_datetime([]),
        'open': [],
        'high': [],
        'low': [],
        'close': [],
        'volume': [],
    })
    out = generate_signals_v8_0(df)
    assert 'entry_blocked_reason' in out.columns
    assert len(out) == 0
