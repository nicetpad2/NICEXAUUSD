import sys
import types
import importlib
import pandas as pd


def test_run_csv_integrity_check():
    main = importlib.import_module('main')
    assert main.run_csv_integrity_check()


def test_run_smart_fast_qa(monkeypatch, capsys):
    main = importlib.reload(importlib.import_module('main'))
    calls = {}
    def fake_run(cmd, check):
        calls['cmd'] = cmd
    monkeypatch.setattr('subprocess.run', fake_run)
    main.run_smart_fast_qa()
    out = capsys.readouterr().out
    assert 'Fast QA' in out
    assert calls.get('cmd')


def test_maximize_ram(monkeypatch, capsys):
    main = importlib.reload(importlib.import_module('main'))
    monkeypatch.setattr(main, 'MAX_RAM_MODE', True)
    dummy_psutil = types.SimpleNamespace(virtual_memory=lambda: types.SimpleNamespace(total=8*1024**3))
    monkeypatch.setitem(sys.modules, 'psutil', dummy_psutil)
    main.maximize_ram()
    assert 'MAX_RAM_MODE: ON' in capsys.readouterr().out


def test_check_exit_reason_variety_pass():
    main = importlib.import_module('main')
    df = pd.DataFrame({'exit_reason': ['tp1', 'tp2', 'sl']})
    assert main.check_exit_reason_variety(df)


def test_check_exit_reason_variety_uppercase():
    main = importlib.import_module('main')
    df = pd.DataFrame({'exit_reason': ['TP', 'SL']})
    assert main.check_exit_reason_variety(df)


def test_check_exit_reason_variety_fail(capsys):
    main = importlib.import_module('main')
    df = pd.DataFrame({'exit_reason': ['tp1']})
    assert main.check_exit_reason_variety(df)
    out = capsys.readouterr().out
    assert 'OK' in out
