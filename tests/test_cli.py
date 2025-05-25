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
        'timestamp': pd.date_range('2024-01-01', periods=10, freq='h'),
        'open': [1]*10,
        'high': [1]*10,
        'low': [1]*10,
        'close': [1]*10
    }))

    monkeypatch.setattr('nicegold_v5.entry.generate_signals', lambda df: df.assign(entry_signal='buy'))

    def fake_run_backtest(df):
        print("✅ เสร็จแล้ว: Trades = 1 | Profit = 1.00")
        return pd.DataFrame({'pnl': [1]}), pd.DataFrame()

    monkeypatch.setattr('nicegold_v5.backtester.run_backtest', fake_run_backtest)
    main.welcome()
    output = capsys.readouterr().out
    assert "Backtest จาก Signal" in output
    assert "เสร็จแล้ว" in output
