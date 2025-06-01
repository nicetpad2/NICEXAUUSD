import pytest
import pandas as pd
from nicegold_v5.tests.test_core_all import sample_df
from nicegold_v5.entry import generate_signals, generate_signals_v12_0


def test_force_entry_injection():
    df = sample_df()
    cfg = {
        "force_entry": True,
        "force_entry_ratio": 0.2,
        "force_entry_side": "buy",
    }
    out = generate_signals(df, config=cfg, test_mode=True)
    assert out["entry_signal"].notnull().sum() >= 10


def test_force_entry_production_block():
    df = sample_df()
    cfg = {"force_entry": True}
    with pytest.raises(RuntimeError):
        generate_signals(df, config=cfg, test_mode=False)


def test_force_entry_v12_injection(monkeypatch):
    df = sample_df()
    monkeypatch.setattr('nicegold_v5.entry.sanitize_price_columns', lambda d: d)
    monkeypatch.setattr('nicegold_v5.entry.validate_indicator_inputs', lambda d: None)
    cfg = {
        "force_entry": True,
        "force_entry_ratio": 0.2,
        "force_entry_side": "buy",
    }
    out = generate_signals_v12_0(df, config=cfg, test_mode=True)
    assert out["entry_signal"].notnull().sum() >= 10

