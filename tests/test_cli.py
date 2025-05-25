import builtins
import importlib
import pandas as pd


def test_welcome_menu_exit(monkeypatch, capsys):
    # Always exit menu by selecting option 5
    monkeypatch.setattr(builtins, "input", lambda prompt='': '5')
    main = importlib.import_module('main')
    main.welcome()
    output = capsys.readouterr().out
    assert "NICEGOLD Assistant" in output
    assert "ออกจากระบบ" in output


def test_welcome_manual_backtest(monkeypatch, capsys, tmp_path):
    inputs = iter(['4'])

    monkeypatch.setattr(builtins, "input", lambda prompt='': next(inputs))

    main = importlib.import_module('main')
    monkeypatch.setattr(main, "TRADE_DIR", str(tmp_path))
    monkeypatch.setattr(main, "load_csv_safe", lambda path: pd.DataFrame({
        'date': [25670101]*20,
        'timestamp': ['00:00:00']*20,
        'open': [1]*20,
        'high': [1]*20,
        'low': [1]*20,
        'close': [1]*20
    }))
    monkeypatch.setattr(main, "run_walkforward_backtest", lambda df, features, label: pd.DataFrame({'time': [pd.Timestamp('2024-01-01')], 'pnl': [1]}))
    main.welcome()
    output = capsys.readouterr().out
    assert "Backtest จาก Signal" in output
    assert "บันทึกผล Backtest" in output
