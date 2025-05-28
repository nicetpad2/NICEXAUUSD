import importlib
import pandas as pd


def test_autorun_simulate(monkeypatch, capsys, tmp_path):
    main = importlib.import_module('main')
    monkeypatch.setattr(main, "TRADE_DIR", str(tmp_path))
    monkeypatch.setattr(main, "load_csv_safe", lambda path: pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=2, freq='h'),
        'close': [100, 101]
    }))
    monkeypatch.setattr(
        'nicegold_v5.entry.simulate_trades_with_tp',
        lambda df: ([{'exit_reason': 'tp1'}], [])
    )
    monkeypatch.setattr(main, 'safe_calculate_net_change', lambda df: 5.0)
    main.welcome()
    output = capsys.readouterr().out
    assert 'QA Summary (TP1/TP2)' in output
    assert 'TP1 Triggered' in output


def test_autorun_string_timestamp(monkeypatch, capsys, tmp_path):
    main = importlib.import_module('main')
    monkeypatch.setattr(main, "TRADE_DIR", str(tmp_path))
    monkeypatch.setattr(main, "load_csv_safe", lambda path: pd.DataFrame({
        'timestamp': ['2024-01-01 00:00:00', '2024-01-01 01:00:00'],
        'close': [100, 101]
    }))

    def fake_simulate(df):
        assert pd.api.types.is_datetime64_any_dtype(df['timestamp'])
        return ([{'exit_reason': 'tp2'}], [])

    monkeypatch.setattr('nicegold_v5.entry.simulate_trades_with_tp', fake_simulate)
    monkeypatch.setattr(main, 'safe_calculate_net_change', lambda df: 7.0)
    main.welcome()
    output = capsys.readouterr().out
    assert 'TP2 Triggered' in output

