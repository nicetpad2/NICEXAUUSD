import importlib
from nicegold_v5 import config


def test_yaml_defaults_loaded():
    assert config.SESSION_CONFIG['Asia']['start'] == '01:00'
    assert config.SNIPER_CONFIG_Q3_TUNED['tp1_pct'] == 0.7
    assert config.DEFAULT_RR1 == 1.2


def test_env_override(monkeypatch):
    monkeypatch.setenv('NICEGOLD_ENV', 'qa')
    cfg = importlib.reload(config)
    assert cfg.ULTRA_OVERRIDE_QA['force_entry'] is True
    monkeypatch.delenv('NICEGOLD_ENV', raising=False)
    importlib.reload(config)
