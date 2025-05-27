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

    monkeypatch.setattr(
        'nicegold_v5.entry.generate_signals_v8_0',
        lambda df: df.assign(entry_signal='buy'),
    )

    def fake_run_backtest(df):
        return (
            pd.DataFrame({'pnl': [1]}),
            pd.DataFrame({'timestamp': [pd.Timestamp('2024-01-01')], 'equity': [100]})
        )

    monkeypatch.setattr('nicegold_v5.backtester.run_backtest', fake_run_backtest)
    monkeypatch.setattr('nicegold_v5.utils.print_qa_summary', lambda *args, **kwargs: {})
    monkeypatch.setattr('nicegold_v5.utils.create_summary_dict', lambda *args, **kwargs: {})
    monkeypatch.setattr('nicegold_v5.utils.export_chatgpt_ready_logs', lambda *args, **kwargs: print('Export Completed'))
    main.welcome()
    output = capsys.readouterr().out
    assert "Backtest จาก Signal" in output
    assert "Export Completed" in output
