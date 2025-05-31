import sys
import types
import importlib


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
