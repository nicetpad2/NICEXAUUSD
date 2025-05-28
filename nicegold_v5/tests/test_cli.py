import importlib
import pandas as pd
import pytest


def test_autorun_simulate(monkeypatch, capsys, tmp_path):
    main = importlib.import_module('main')
    monkeypatch.setattr(main, "TRADE_DIR", str(tmp_path))
    monkeypatch.setattr(main, "load_csv_safe", lambda path: pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=2, freq='h'),
        'entry_signal': ['long', 'short'],
        'entry_time': pd.date_range('2024-01-01', periods=2, freq='h')
    }))
    monkeypatch.setattr(
        'nicegold_v5.entry.simulate_trades_with_tp',
        lambda df: ([{'exit_reason': 'tp1'}], [])
    )
    monkeypatch.setattr(main, 'safe_calculate_net_change', lambda df: 5.0)
    main.welcome()
    output = capsys.readouterr().out
    assert 'Summary (TP1/TP2)' in output
    assert 'TP1 Triggered' in output


def test_autorun_string_timestamp(monkeypatch, capsys, tmp_path):
    main = importlib.import_module('main')
    monkeypatch.setattr(main, "TRADE_DIR", str(tmp_path))
    monkeypatch.setattr(main, "load_csv_safe", lambda path: pd.DataFrame({
        'timestamp': ['2024-01-01 00:00:00', '2024-01-01 01:00:00'],
        'entry_signal': ['long', 'short'],
        'entry_time': ['2024-01-01 00:00:00', '2024-01-01 01:00:00']
    }))

    def fake_simulate(df):
        assert pd.api.types.is_datetime64_any_dtype(df['timestamp'])
        return ([{'exit_reason': 'tp2'}], [])

    monkeypatch.setattr('nicegold_v5.entry.simulate_trades_with_tp', fake_simulate)
    monkeypatch.setattr(main, 'safe_calculate_net_change', lambda df: 7.0)
    main.welcome()
    output = capsys.readouterr().out
    assert 'TP2 Triggered' in output


def test_autorun_missing_entry_time(monkeypatch, tmp_path):
    main = importlib.import_module('main')
    monkeypatch.setattr(main, "TRADE_DIR", str(tmp_path))
    monkeypatch.setattr(main, "load_csv_safe", lambda path: pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1, freq='h'),
        'entry_signal': ['long']
    }))

    monkeypatch.setattr('nicegold_v5.entry.generate_signals', lambda df, config=None: df)
    with pytest.raises(ValueError):
        main.welcome()


def test_autorun_relax_fallback(monkeypatch, capsys, tmp_path):
    main = importlib.import_module('main')
    monkeypatch.setattr(main, "TRADE_DIR", str(tmp_path))

    # DataFrame without entry_signal column to trigger auto-generation
    monkeypatch.setattr(
        main,
        "load_csv_safe",
        lambda path: pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=2, freq='h'),
            'entry_time': pd.date_range('2024-01-01', periods=2, freq='h'),
        })
    )

    def fake_generate(df, config=None):
        if config == main.SNIPER_CONFIG_Q3_TUNED:
            return df.assign(entry_signal=[None, None])
        elif config == main.RELAX_CONFIG_Q3:
            return df.assign(entry_signal=['long', 'short'])
        else:
            raise AssertionError("Unexpected config")

    monkeypatch.setattr(main, 'generate_signals', fake_generate)
    monkeypatch.setattr('nicegold_v5.entry.generate_signals', fake_generate)
    monkeypatch.setattr(
        'nicegold_v5.entry.simulate_trades_with_tp',
        lambda df: ([{'exit_reason': 'tp1'}], [])
    )
    monkeypatch.setattr(main, 'safe_calculate_net_change', lambda df: 1.0)

    main.welcome()
    output = capsys.readouterr().out
    assert 'fallback to relaxed strategy' in output

