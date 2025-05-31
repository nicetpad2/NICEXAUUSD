import importlib
import pandas as pd
import numpy as np

import nicegold_v5.utils as utils
import nicegold_v5.wfv as wfv


def test_wfv_auto_entry_and_split():
    df = pd.DataFrame({
        'atr': np.linspace(1, 2, 24),
        'ema_fast': np.linspace(1, 3, 24),
        'gain_z': np.linspace(0, 0.2, 24),
        'timestamp': pd.date_range('2025-01-01', periods=24, freq='h'),
    })
    cfg = wfv.auto_entry_config(df)
    assert 'gain_z_thresh' in cfg
    sessions = wfv.split_by_session(df)
    assert set(sessions.keys()) == {'Asia', 'London', 'NY'}


def test_build_trade_log_empty_window():
    position = {
        'entry_time': pd.Timestamp('2025-01-01 00:00:00'),
        'entry': 100.0,
        'sl': 99.0,
        'tp': 101.0,
        'lot': 0.1,
        'side': 'buy',
        'commission': 0.0,
    }
    ts = pd.Timestamp('2025-01-01 01:00:00')
    df_empty = pd.DataFrame({'Open': [], 'Close': []})
    trade = wfv.build_trade_log(position, ts, 100.0, False, False, 1000.0, 0.0, df_empty)
    assert trade['break_even_min'] is None
    assert trade['mfe'] == 0.0


def test_run_walkforward_backtest_skip(monkeypatch, capsys):
    df = pd.DataFrame({
        'Open': np.linspace(100, 101, 4),
        'feat1': [1, 2, 3, 4],
        'feat2': [5, 6, 7, 8],
        'label': [1, 1, 1, 1],
        'ATR_14': 1.0,
        'ATR_14_MA50': 1.0,
        'EMA_50_slope': 0.1,
    }, index=pd.date_range('2025-01-01', periods=4, freq='h'))
    trades = wfv.run_walkforward_backtest(df, ['feat1', 'feat2'], 'label', n_folds=2)
    out = capsys.readouterr().out
    assert "insufficient class" in out
    assert trades.empty


def test_prepare_csv_auto_validate_error(monkeypatch):
    utils_mod = importlib.reload(importlib.import_module('nicegold_v5.utils'))
    main = importlib.reload(importlib.import_module('main'))
    df = pd.DataFrame({
        'timestamp': ['2025-01-01 00:00:00'],
        'open': ['1'], 'high': ['1'], 'low': ['1'], 'close': ['1'], 'volume': ['1']
    })
    monkeypatch.setattr(main, 'load_csv_safe', lambda p: df)
    monkeypatch.setattr(main, 'convert_thai_datetime', lambda d: d)
    monkeypatch.setattr(main, 'parse_timestamp_safe', lambda s, fmt: pd.to_datetime(s))
    monkeypatch.setattr(main, 'sanitize_price_columns', lambda d: d)
    def raise_error(*args, **kwargs):
        raise TypeError('fail')
    monkeypatch.setattr(main, 'validate_indicator_inputs', raise_error)
    out = utils_mod.prepare_csv_auto('dummy.csv')
    assert not out.empty


def test_safe_calculate_net_change_default():
    df = pd.DataFrame([
        {'entry_price': 100.0, 'exit_price': 105.0},
        {'entry_price': 200.0, 'exit_price': 195.0},
    ])
    assert utils.safe_calculate_net_change(df) == 0.0


def test_convert_thai_datetime_error(monkeypatch, capsys):
    df = pd.DataFrame({'Date': ['25660416'], 'Timestamp': ['22:00:00']})
    monkeypatch.setattr(pd, 'to_datetime', lambda *a, **k: (_ for _ in ()).throw(ValueError('bad')))
    result = utils.convert_thai_datetime(df)
    out = capsys.readouterr().out
    assert '❌' in out
    assert 'timestamp' not in result
