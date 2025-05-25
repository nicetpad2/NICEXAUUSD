import builtins
import importlib


def test_welcome_menu_exit(monkeypatch, capsys):
    # Always exit menu by selecting option 5
    monkeypatch.setattr(builtins, "input", lambda prompt='': '5')
    main = importlib.import_module('main')
    main.welcome()
    output = capsys.readouterr().out
    assert "NICEGOLD Assistant" in output
    assert "ออกจากระบบ" in output
