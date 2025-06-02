import pandas as pd
import nicegold_v5.wfv as wfv


def test_build_trade_log_missing_index():
    df = pd.DataFrame({'Open': [100, 101]}, index=pd.date_range('2025-01-01', periods=2, freq='h'))
    position = {
        'entry_time': pd.Timestamp('2025-01-01 03:00:00'),
        'entry': 100.0,
        'sl': 99.0,
        'tp': 101.0,
        'lot': 0.1,
        'side': 'buy',
        'commission': 0.0,
    }
    ts = pd.Timestamp('2025-01-01 04:00:00')
    trade = wfv.build_trade_log(position, ts, 102.0, False, False, 1000.0, 0.0, df)
    assert trade['mfe'] == 0.0
