import pandas as pd
from nicegold_v5.entry import sanitize_price_columns


def test_sanitize_conversion_and_logging(capsys):
    df = pd.DataFrame({
        'close': ['1', '2'],
        'high': ['3', '4'],
        'low': ['5', '6'],
        'open': ['7', '8'],
        'volume': ['9', '10'],
    })
    out = sanitize_price_columns(df)
    assert out['close'].tolist() == [1.0, 2.0]
    assert pd.api.types.is_numeric_dtype(out['high'])
    log = capsys.readouterr().out
    assert 'Sanitize Columns' in log


def test_sanitize_nan_count(capsys):
    df = pd.DataFrame({
        'close': ['a', '1'],
        'high': ['b', '2'],
        'low': ['c', '3'],
        'open': ['d', '4'],
        'volume': ['e', '5'],
    })
    out = sanitize_price_columns(df)
    assert out['close'].isna().sum() == 1
    log = capsys.readouterr().out
    assert 'close: 1 NaN' in log

