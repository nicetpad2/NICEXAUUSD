import importlib
import pandas as pd
import pytest


def test_run_production_wfv(monkeypatch):
    main = importlib.import_module('main')
    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=2, freq='min'),
        'open': [1, 2],
        'high': [1, 2],
        'low': [1, 2],
        'close': [1, 2],
        'gain_z': [0.0, 0.0],
        'ema_slope': [0.0, 0.0],
        'atr': [1.0, 1.0],
        'rsi': [50, 50],
        'volume': [100, 100],
        'entry_score': [0.1, 0.2],
        'pattern_label': [1, 0],
        'tp2_hit': [0, 1],
    })
    monkeypatch.setattr(main, 'load_csv_safe', lambda p: df)
    monkeypatch.setattr(main, 'convert_thai_datetime', lambda d: d)
    monkeypatch.setattr(main, 'parse_timestamp_safe', lambda s, fmt: s)
    monkeypatch.setattr(main, 'sanitize_price_columns', lambda d: d)
    monkeypatch.setattr(main, 'validate_indicator_inputs', lambda d, min_rows=None: None)
    monkeypatch.setattr(main, 'check_exit_reason_variety', lambda df: True)
    monkeypatch.setattr('nicegold_v5.entry.generate_signals_v8_0', lambda d, config=None: d.assign(entry_signal=['buy']*len(d)))

    called = {}
    def fake_run(df_in, features, label_col, **kw):
        called['args'] = (features, label_col)
        called['cols'] = list(df_in.columns)
        called['index_type'] = isinstance(df_in.index, pd.DatetimeIndex)
        return pd.DataFrame({
            'pnl': [1.0] * 5,
            'side': ['buy'] * 5,
            'exit_reason': ['tp1'] * 5,
            'is_dummy': [False] * 5,
        })
    monkeypatch.setattr(main, 'run_walkforward_backtest', fake_run)
    monkeypatch.setattr(main, 'auto_qa_after_backtest', lambda *a, **k: None)

    main.run_production_wfv()

    assert called['args'][1] == 'tp2_hit'
    assert 'Open' in called['cols']


def test_run_production_wfv_close_fallback(monkeypatch):
    main = importlib.import_module('main')
    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=2, freq='min'),
        'high': [1, 2],
        'low': [1, 2],
        'close': [1, 2],
        'gain_z': [0.0, 0.0],
        'ema_slope': [0.0, 0.0],
        'atr': [1.0, 1.0],
        'rsi': [50, 50],
        'volume': [100, 100],
        'entry_score': [0.1, 0.2],
        'pattern_label': [1, 0],
        'tp2_hit': [0, 1],
    })
    monkeypatch.setattr(main, 'load_csv_safe', lambda p: df)
    monkeypatch.setattr(main, 'convert_thai_datetime', lambda d: d)
    monkeypatch.setattr(main, 'parse_timestamp_safe', lambda s, fmt: s)
    monkeypatch.setattr(main, 'sanitize_price_columns', lambda d: d)
    monkeypatch.setattr(main, 'validate_indicator_inputs', lambda d, min_rows=None: None)
    monkeypatch.setattr(main, 'check_exit_reason_variety', lambda df: True)

    called = {}
    def fake_run(df_in, features, label_col, **kw):
        called['args'] = (features, label_col)
        called['cols'] = list(df_in.columns)
        called['index_type'] = isinstance(df_in.index, pd.DatetimeIndex)
        return pd.DataFrame({
            'pnl': [1.0] * 5,
            'side': ['buy'] * 5,
            'exit_reason': ['tp1'] * 5,
            'is_dummy': [False] * 5,
        })
    monkeypatch.setattr(main, 'run_walkforward_backtest', fake_run)
    monkeypatch.setattr(main, 'auto_qa_after_backtest', lambda *a, **k: None)

    main.run_production_wfv()

    assert called['args'][1] == 'tp2_hit'
    assert 'Open' in called['cols']


def test_run_production_wfv_no_open_close(monkeypatch):
    main = importlib.import_module('main')
    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=2, freq='min'),
        'high': [1, 2],
        'low': [1, 2],
        'gain_z': [0.0, 0.0],
        'ema_slope': [0.0, 0.0],
        'atr': [1.0, 1.0],
        'rsi': [50, 50],
        'volume': [100, 100],
        'entry_score': [0.1, 0.2],
        'pattern_label': [1, 0],
        'tp2_hit': [0, 1],
    })
    monkeypatch.setattr(main, 'load_csv_safe', lambda p: df)
    monkeypatch.setattr(main, 'convert_thai_datetime', lambda d: d)
    monkeypatch.setattr(main, 'parse_timestamp_safe', lambda s, fmt: s)
    monkeypatch.setattr(main, 'sanitize_price_columns', lambda d: d)
    monkeypatch.setattr(main, 'validate_indicator_inputs', lambda d, min_rows=None: None)
    monkeypatch.setattr(main, 'check_exit_reason_variety', lambda df: True)
    called = {}
    monkeypatch.setattr(main, 'auto_qa_after_backtest', lambda *a, **k: None)
    monkeypatch.setattr(main, 'run_walkforward_backtest', lambda *a, **k: pd.DataFrame({'pnl':[0.0]*5, 'side':['buy']*5, 'exit_reason':['tp2']*5, 'is_dummy':[False]*5}))

    main.run_production_wfv()


def test_run_production_wfv_auto_dataset(monkeypatch):
    main = importlib.import_module('main')
    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=2, freq='min'),
        'open': [1, 2],
        'high': [1, 2],
        'low': [1, 2],
        'close': [1, 2],
        'gain_z': [0.0, 0.0],
        'ema_slope': [0.0, 0.0],
        'atr': [1.0, 1.0],
        'rsi': [50, 50],
        'volume': [100, 100],
        'entry_score': [0.1, 0.2],
        'pattern_label': [1, 0],
    })
    monkeypatch.setattr(main, 'load_csv_safe', lambda p: df)
    monkeypatch.setattr(main, 'convert_thai_datetime', lambda d: d)
    monkeypatch.setattr(main, 'parse_timestamp_safe', lambda s, fmt: s)
    monkeypatch.setattr(main, 'sanitize_price_columns', lambda d: d)
    monkeypatch.setattr(main, 'validate_indicator_inputs', lambda d, min_rows=None: None)
    monkeypatch.setattr(main, 'check_exit_reason_variety', lambda df: True)

    gen_called = {}
    monkeypatch.setattr('nicegold_v5.ml_dataset_m1.generate_ml_dataset_m1', lambda *a, **k: gen_called.setdefault('called', True))
    monkeypatch.setattr(pd, 'read_csv', lambda p: df.assign(tp2_hit=[0, 1]))
    monkeypatch.setattr(main, 'run_walkforward_backtest', lambda *a, **k: pd.DataFrame({'pnl':[0.0]*5, 'side':['buy']*5, 'exit_reason':['tp1']*5, 'is_dummy':[False]*5}))
    monkeypatch.setattr(main, 'auto_qa_after_backtest', lambda *a, **k: None)

    main.run_production_wfv()

    assert gen_called.get('called')

def test_run_production_wfv_insufficient_trades(monkeypatch):
    main = importlib.import_module('main')
    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=2, freq='min'),
        'open': [1, 2],
        'high': [1, 2],
        'low': [1, 2],
        'close': [1, 2],
        'gain_z': [0.0, 0.0],
        'ema_slope': [0.0, 0.0],
        'atr': [1.0, 1.0],
        'rsi': [50, 50],
        'volume': [100, 100],
        'entry_score': [0.1, 0.2],
        'pattern_label': [1, 0],
    })
    monkeypatch.setattr(main, 'load_csv_safe', lambda p: df)
    monkeypatch.setattr(main, 'convert_thai_datetime', lambda d: d)
    monkeypatch.setattr(main, 'parse_timestamp_safe', lambda s, fmt: s)
    monkeypatch.setattr(main, 'sanitize_price_columns', lambda d: d)
    monkeypatch.setattr(main, 'validate_indicator_inputs', lambda d, min_rows=None: None)
    monkeypatch.setattr(main, 'check_exit_reason_variety', lambda df: True)

    def raise_runtime(*a, **k):
        raise RuntimeError('[Inject Variety] ❌ ไม่พอ TP1/TP2/SL >=5 ใน production')

    monkeypatch.setattr('nicegold_v5.ml_dataset_m1.generate_ml_dataset_m1', raise_runtime)
    called = {}

    def fake_run(df_in, features, label_col, **kw):
        called['cols'] = list(df_in.columns)
        return pd.DataFrame({'pnl': [0.0], 'side': ['buy'], 'exit_reason': ['tp1'], 'is_dummy': [False]})

    monkeypatch.setattr(main, 'run_walkforward_backtest', fake_run)
    monkeypatch.setattr(main, 'auto_qa_after_backtest', lambda *a, **k: None)

    main.run_production_wfv()

    assert 'tp2_hit' in called['cols']


def test_run_production_wfv_fallback_relax(monkeypatch):
    main = importlib.import_module('main')
    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=1, freq='min'),
        'open': [1],
        'high': [1],
        'low': [1],
        'close': [1],
        'gain_z': [0.0],
        'ema_slope': [0.0],
        'atr': [1.0],
        'rsi': [50],
        'entry_score': [0.1],
        'pattern_label': [1],
        'tp2_hit': [1],
    })
    monkeypatch.setattr(main, 'load_csv_safe', lambda p: df)
    monkeypatch.setattr(main, 'convert_thai_datetime', lambda d: d)
    monkeypatch.setattr(main, 'parse_timestamp_safe', lambda s, fmt: s)
    monkeypatch.setattr(main, 'sanitize_price_columns', lambda d: d)
    monkeypatch.setattr(main, 'validate_indicator_inputs', lambda d, min_rows=None: None)
    monkeypatch.setattr(main, 'check_exit_reason_variety', lambda df: True)

    call = {'count': 0}

    def fake_gen(data, config=None):
        call['count'] += 1
        out = data.copy()
        if call['count'] == 1:
            out['entry_signal'] = [None]
        else:
            out['entry_signal'] = ['buy']
        return out

    monkeypatch.setattr('nicegold_v5.entry.generate_signals_v8_0', fake_gen)

    run_called = {}

    def fake_run(df_in, *a, **k):
        run_called['entries'] = df_in['entry_signal'].notnull().sum()
        return pd.DataFrame({'pnl': [0.0], 'is_dummy': [False]})

    monkeypatch.setattr(main, 'run_walkforward_backtest', fake_run)
    monkeypatch.setattr(main, 'auto_qa_after_backtest', lambda *a, **k: None)

    main.run_production_wfv()

    assert call['count'] == 2
    assert run_called['entries'] == 1


def test_run_production_wfv_empty_trades(monkeypatch):
    main = importlib.import_module('main')
    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=2, freq='min'),
        'open': [1, 2],
        'high': [1, 2],
        'low': [1, 2],
        'close': [1, 2],
        'gain_z': [0.0, 0.0],
        'ema_slope': [0.0, 0.0],
        'atr': [1.0, 1.0],
        'rsi': [50, 50],
        'volume': [100, 100],
        'entry_score': [0.1, 0.2],
        'pattern_label': [1, 0],
        'tp2_hit': [0, 1],
    })
    monkeypatch.setattr(main, 'load_csv_safe', lambda p: df)
    monkeypatch.setattr(main, 'convert_thai_datetime', lambda d: d)
    monkeypatch.setattr(main, 'parse_timestamp_safe', lambda s, fmt: s)
    monkeypatch.setattr(main, 'sanitize_price_columns', lambda d: d)
    monkeypatch.setattr(main, 'validate_indicator_inputs', lambda d, min_rows=None: None)
    monkeypatch.setattr(main, 'check_exit_reason_variety', lambda df: True)
    monkeypatch.setattr('nicegold_v5.entry.generate_signals_v8_0', lambda d, config=None: d.assign(entry_signal=['buy']*len(d)))

    monkeypatch.setattr(main, 'run_walkforward_backtest', lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(main, 'auto_qa_after_backtest', lambda *a, **k: None)

    out = main.run_production_wfv()

    assert out.empty
