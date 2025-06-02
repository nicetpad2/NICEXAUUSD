import importlib
from nicegold_v5 import config


def test_yaml_defaults_loaded():
    assert config.SESSION_CONFIG['Asia']['start'] == '01:00'
    assert config.SNIPER_CONFIG_Q3_TUNED['tp1_pct'] == 0.7
    assert config.DEFAULT_RR1 == 1.2
    # ตรวจสอบคีย์ใหม่ใน SNIPER_CONFIG_Q3_TUNED
    assert config.SNIPER_CONFIG_Q3_TUNED['gain_z_thresh'] == 0.5
    assert config.SNIPER_CONFIG_Q3_TUNED['atr_multiplier'] == 1.0
    assert config.SNIPER_CONFIG_Q3_TUNED['rsi_oversold'] == 30
    assert config.SNIPER_CONFIG_Q3_TUNED['rsi_overbought'] == 70
    # ตรวจสอบคีย์ใหม่ใน RELAX_CONFIG_Q3
    assert config.RELAX_CONFIG_Q3['gain_z_thresh'] == 0.3
    assert config.RELAX_CONFIG_Q3['atr_multiplier'] == 0.8
    assert config.RELAX_CONFIG_Q3['rsi_oversold'] == 35
    assert config.RELAX_CONFIG_Q3['rsi_overbought'] == 65


def test_env_override(monkeypatch):
    monkeypatch.setenv('NICEGOLD_ENV', 'qa')
    cfg = importlib.reload(config)
    assert cfg.ULTRA_OVERRIDE_QA['force_entry'] is True
    monkeypatch.delenv('NICEGOLD_ENV', raising=False)
    importlib.reload(config)
