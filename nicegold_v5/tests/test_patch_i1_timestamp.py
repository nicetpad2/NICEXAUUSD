import pandas as pd
from nicegold_v5.utils import convert_thai_datetime

def test_convert_thai_datetime():
    df = pd.DataFrame({'Date': ['25660416'], 'Timestamp': ['22:00:00']})
    result = convert_thai_datetime(df)
    assert 'timestamp' in result.columns
    assert result.loc[0, 'timestamp'] == pd.Timestamp('2023-04-16 22:00:00')
