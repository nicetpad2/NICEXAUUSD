import importlib
import pandas as pd
import pytest


def test_autorun_simulate(monkeypatch, capsys, tmp_path):
    main = importlib.import_module('main')
    monkeypatch.setattr('builtins.input', lambda _: '6')
    monkeypatch.setattr(main, "TRADE_DIR", str(tmp_path))
    monkeypatch.setattr(main, "load_csv_safe", lambda path: pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=2, freq='h'),
        'entry_signal': ['long', 'short'],
        'entry_time': pd.date_range('2024-01-01', periods=2, freq='h'),
        'close': [1, 1],
        'high': [1, 1],
        'low': [1, 1],
        'volume': [1, 1],
    }))
    monkeypatch.setattr('nicegold_v5.entry.validate_indicator_inputs', lambda df, required_cols=None, min_rows=500: None)
    monkeypatch.setattr(main, 'validate_indicator_inputs', lambda df: None)
    monkeypatch.setattr(main, 'generate_signals', lambda df: df)
    monkeypatch.setattr(main, 'generate_signals', lambda df: df)
    monkeypatch.setattr(
        'nicegold_v5.exit.simulate_partial_tp_safe',
        lambda df: pd.DataFrame([{'exit_reason': 'tp1'}])
    )
    monkeypatch.setattr(main, 'safe_calculate_net_change', lambda df: 5.0)
    main.welcome()
    output = capsys.readouterr().out
    assert 'Summary (TP1/TP2)' in output


def test_cli_train_lstm(monkeypatch, capsys, tmp_path):
    import types
    import sys
    main = importlib.import_module('main')
    monkeypatch.setattr('builtins.input', lambda _: '5')
    monkeypatch.setattr(main, 'TRADE_DIR', str(tmp_path))

    monkeypatch.setattr(main, 'load_csv_safe', lambda path: pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=2, freq='h'),
        'entry_signal': ['long', 'short'],
        'entry_time': pd.date_range('2024-01-01', periods=2, freq='h'),
        'close': [1, 1],
        'high': [1, 1],
        'low': [1, 1],
        'volume': [1, 1],
    }))
    monkeypatch.setattr('nicegold_v5.entry.validate_indicator_inputs', lambda df, required_cols=None, min_rows=500: None)
    monkeypatch.setattr(main, 'validate_indicator_inputs', lambda df: None)
    monkeypatch.setattr(main, 'generate_signals', lambda df: df)
    monkeypatch.setattr('nicegold_v5.exit.simulate_partial_tp_safe', lambda df: pd.DataFrame([{'exit_reason': 'tp1'}]))
    monkeypatch.setattr(main, 'safe_calculate_net_change', lambda df: 0.0)

    called = {}

    data_mod = types.SimpleNamespace(DataLoader=object, TensorDataset=object)
    utils_mod = types.ModuleType('torch.utils')
    utils_mod.data = data_mod
    dummy_torch = types.ModuleType('torch')
    dummy_torch.__path__ = []
    dummy_torch.save = lambda state, path: called.setdefault('saved', path)
    dummy_torch.load = lambda *a, **k: {}
    dummy_torch.nn = types.ModuleType('torch.nn')
    dummy_torch.nn.Module = object
    dummy_torch.nn.LSTM = object
    dummy_torch.nn.Linear = object
    dummy_torch.nn.Sigmoid = lambda: None
    dummy_torch.optim = types.ModuleType('torch.optim')
    dummy_torch.utils = utils_mod
    monkeypatch.setitem(sys.modules, 'torch', dummy_torch)
    monkeypatch.setitem(sys.modules, 'torch.nn', dummy_torch.nn)
    monkeypatch.setitem(sys.modules, 'torch.optim', dummy_torch.optim)
    monkeypatch.setitem(sys.modules, 'torch.utils', utils_mod)
    monkeypatch.setitem(sys.modules, 'torch.utils.data', data_mod)

    monkeypatch.setattr('nicegold_v5.ml_dataset_m1.generate_ml_dataset_m1', lambda: called.setdefault('gen', True))
    monkeypatch.setattr('nicegold_v5.train_lstm_runner.load_dataset', lambda p: (None, None))

    class DummyModel:
        def state_dict(self):
            return {}

    monkeypatch.setattr('nicegold_v5.train_lstm_runner.train_lstm', lambda X, y: DummyModel())

    main.welcome()
    output = capsys.readouterr().out
    assert 'บันทึกโมเดล' in output
    assert called.get('gen') is True
    assert called.get('saved') == 'models/model_lstm_tp2.pth'


def test_autorun_string_timestamp(monkeypatch, capsys, tmp_path):
    main = importlib.import_module('main')
    monkeypatch.setattr('builtins.input', lambda _: '6')
    monkeypatch.setattr(main, "TRADE_DIR", str(tmp_path))
    monkeypatch.setattr(main, "load_csv_safe", lambda path: pd.DataFrame({
        'timestamp': ['2024-01-01 00:00:00', '2024-01-01 01:00:00'],
        'entry_signal': ['long', 'short'],
        'entry_time': ['2024-01-01 00:00:00', '2024-01-01 01:00:00'],
        'close': [1, 1],
        'high': [1, 1],
        'low': [1, 1],
        'volume': [1, 1],
    }))

    def fake_simulate(df):
        assert pd.api.types.is_datetime64_any_dtype(df['timestamp'])
        return pd.DataFrame([{'exit_reason': 'tp2'}])
    monkeypatch.setattr('nicegold_v5.entry.validate_indicator_inputs', lambda df, required_cols=None, min_rows=500: None)
    monkeypatch.setattr(main, 'validate_indicator_inputs', lambda df: None)
    monkeypatch.setattr(main, 'generate_signals', lambda df: df)

    monkeypatch.setattr('nicegold_v5.exit.simulate_partial_tp_safe', fake_simulate)
    monkeypatch.setattr(main, 'safe_calculate_net_change', lambda df: 7.0)
    main.welcome()
    output = capsys.readouterr().out
    assert 'TP2 Triggered' in output


def test_autorun_missing_entry_time(monkeypatch, capsys, tmp_path):
    main = importlib.import_module('main')
    monkeypatch.setattr('builtins.input', lambda _: '6')
    monkeypatch.setattr(main, "TRADE_DIR", str(tmp_path))
    monkeypatch.setattr(main, "load_csv_safe", lambda path: pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1, freq='h'),
        'entry_signal': ['long'],
        'close': [1],
        'high': [1],
        'low': [1],
        'volume': [1],
    }))
    monkeypatch.setattr('nicegold_v5.entry.validate_indicator_inputs', lambda df, required_cols=None, min_rows=500: None)
    monkeypatch.setattr(main, 'validate_indicator_inputs', lambda df: None)
    monkeypatch.setattr('nicegold_v5.entry.generate_signals', lambda df, config=None: df)
    monkeypatch.setattr(main, 'generate_signals', lambda df: df)
    main.welcome()
    output = capsys.readouterr().out
    assert 'ไม่มีข้อมูลครบสำหรับ simulate' in output


def test_autorun_relax_fallback(monkeypatch, capsys, tmp_path):
    main = importlib.import_module('main')
    monkeypatch.setattr('builtins.input', lambda _: '6')
    monkeypatch.setattr(main, "TRADE_DIR", str(tmp_path))

    # DataFrame without entry_signal column to trigger auto-generation
    monkeypatch.setattr(
        main,
        "load_csv_safe",
        lambda path: pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=2, freq='h'),
            'entry_time': pd.date_range('2024-01-01', periods=2, freq='h'),
            'close': [1, 1],
            'high': [1, 1],
            'low': [1, 1],
            'volume': [1, 1],
        })
    )
    monkeypatch.setattr('nicegold_v5.entry.validate_indicator_inputs', lambda df, required_cols=None, min_rows=500: None)

    monkeypatch.setattr(main, 'validate_indicator_inputs', lambda df: None)
    def fake_generate(df, config=None):
        if config == main.SNIPER_CONFIG_Q3_TUNED:
            return df.assign(entry_signal=[None, None])
        elif config == main.RELAX_CONFIG_Q3:
            return df.assign(entry_signal=['long', 'short'])
        elif config is None:
            return df.assign(entry_signal=['x', 'y'])
        else:
            raise AssertionError("Unexpected config")

    monkeypatch.setattr(main, 'generate_signals', fake_generate)
    monkeypatch.setattr('nicegold_v5.entry.generate_signals', fake_generate)
    monkeypatch.setattr(
        'nicegold_v5.exit.simulate_partial_tp_safe',
        lambda df: pd.DataFrame([{'exit_reason': 'tp1'}])
    )
    monkeypatch.setattr(main, 'safe_calculate_net_change', lambda df: 1.0)

    main.welcome()
    output = capsys.readouterr().out
    assert 'Summary (TP1/TP2)' in output


def test_autorun_diagnostic_fallback(monkeypatch, capsys, tmp_path):
    main = importlib.import_module('main')
    monkeypatch.setattr('builtins.input', lambda _: '6')
    monkeypatch.setattr(main, "TRADE_DIR", str(tmp_path))

    monkeypatch.setattr(
        main,
        "load_csv_safe",
        lambda path: pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=2, freq='h'),
            'entry_time': pd.date_range('2024-01-01', periods=2, freq='h'),
            'close': [1, 1],
            'high': [1, 1],
            'low': [1, 1],
            'volume': [1, 1],
        })
    )
    monkeypatch.setattr("nicegold_v5.entry.validate_indicator_inputs", lambda df, required_cols=None, min_rows=500: None)
    monkeypatch.setattr(main, 'validate_indicator_inputs', lambda df: None)

    from nicegold_v5 import config as cfg

    def fake_generate(df, config=None):
        if config == main.SNIPER_CONFIG_Q3_TUNED:
            return df.assign(entry_signal=[None, None])
        elif config == main.RELAX_CONFIG_Q3:
            return df.assign(entry_signal=[None, None])
        elif config == cfg.SNIPER_CONFIG_DIAGNOSTIC:
            return df.assign(entry_signal=['long', 'short'], gain_z=[0.1, 0.2], atr=[1, 1], volume_ma=[100, 100])
        elif config is None:
            return df.assign(entry_signal=['x', 'y'])
        else:
            raise AssertionError("Unexpected config")

    monkeypatch.setattr(main, 'generate_signals', fake_generate)
    monkeypatch.setattr('nicegold_v5.entry.generate_signals', fake_generate)
    monkeypatch.setattr(
        'nicegold_v5.exit.simulate_partial_tp_safe',
        lambda df: pd.DataFrame([{'exit_reason': 'tp1'}])
    )
    monkeypatch.setattr(main, 'safe_calculate_net_change', lambda df: 2.0)

    main.welcome()
    output = capsys.readouterr().out
    assert 'Summary (TP1/TP2)' in output


def test_cli_autofix_wfv(monkeypatch, capsys, tmp_path):
    import importlib
    main = importlib.import_module('main')
    monkeypatch.setattr('builtins.input', lambda _: '7')
    monkeypatch.setattr(main, 'TRADE_DIR', str(tmp_path))
    monkeypatch.setattr(main, 'maximize_ram', lambda: None)

    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=2, freq='h'),
        'open': [1, 1],
        'high': [1, 1],
        'low': [1, 1],
        'close': [1, 1],
        'volume': [1, 1],
    })
    monkeypatch.setattr(main, 'load_csv_safe', lambda path: df)
    monkeypatch.setattr(main, 'convert_thai_datetime', lambda d: d)
    monkeypatch.setattr(main, 'parse_timestamp_safe', lambda s, fmt: s)
    monkeypatch.setattr(main, 'sanitize_price_columns', lambda d: d)
    monkeypatch.setattr('nicegold_v5.entry.validate_indicator_inputs', lambda df, required_cols=None, min_rows=500: None)
    monkeypatch.setattr(main, 'validate_indicator_inputs', lambda df, required_cols=None, min_rows=500: None)
    monkeypatch.setattr(main, 'generate_signals', lambda df, config=None: df.assign(entry_signal=['long'] * len(df)))

    called = {}

    def fake_run(df_arg, sim_fn, cfg, n_folds=5):
        called['n_folds'] = n_folds
        return pd.DataFrame({'fold': [1], 'pnl': [0.0]})

    monkeypatch.setattr('nicegold_v5.utils.run_autofix_wfv', fake_run)
    monkeypatch.setattr('nicegold_v5.exit.simulate_partial_tp_safe', lambda d: pd.DataFrame({'exit_reason': ['tp1']}))

    main.welcome()
    output = capsys.readouterr().out
    assert 'AutoFix WFV' in output
    assert called.get('n_folds') == 5

