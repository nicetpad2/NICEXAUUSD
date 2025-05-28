import pandas as pd
import warnings
import importlib

def test_no_datetime_warning():
    main = importlib.import_module('main')
    df = pd.DataFrame({'timestamp': ['2025-01-01 00:00:00', '2025-01-01 01:00:00']})
    with warnings.catch_warnings(record=True) as w:
        df['timestamp'] = pd.to_datetime(
            df['timestamp'], format=main.DATETIME_FORMAT, errors='coerce'
        )
    assert len(w) == 0
