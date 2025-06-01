import pandas as pd
import importlib

from nicegold_v5.ml_dataset_m1 import generate_ml_dataset_m1


def test_generate_ml_dataset_import_fallback(tmp_path, monkeypatch):
    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=20, freq='min'),
        'open': 1,
        'high': 1,
        'low': 1,
        'close': 1,
        'volume': 1,
    })
    csv_path = tmp_path / 'XAUUSD_M1.csv'
    df.to_csv(csv_path, index=False)
    monkeypatch.setattr(importlib, 'import_module', lambda name: (_ for _ in ()).throw(ImportError()))
    monkeypatch.setattr('nicegold_v5.entry.generate_signals', lambda d, config=None, **kw: d)
    monkeypatch.setattr('nicegold_v5.exit.simulate_partial_tp_safe', lambda d: pd.DataFrame({'entry_time': d['timestamp'].iloc[:1], 'exit_reason':['tp2']}))
    monkeypatch.chdir(tmp_path)
    out_csv = tmp_path / 'out.csv'
    generate_ml_dataset_m1(None, str(out_csv))
    assert out_csv.exists()


def test_generate_ml_dataset_oversample(tmp_path, monkeypatch):
    rows = 600
    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=rows, freq='min'),
        'open': 1,
        'high': 1,
        'low': 1,
        'close': 1,
        'volume': 1,
    })
    csv_path = tmp_path / 'XAUUSD_M1.csv'
    df.to_csv(csv_path, index=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('nicegold_v5.entry.generate_signals', lambda d, config=None, **kw: d)
    monkeypatch.setattr('nicegold_v5.exit.simulate_partial_tp_safe', lambda d: pd.DataFrame({'entry_time': d['entry_time'].iloc[[0]], 'exit_reason':['tp2']}))
    out_csv = tmp_path / 'data' / 'ml_dataset_m1.csv'
    generate_ml_dataset_m1(str(csv_path), str(out_csv))
    out_df = pd.read_csv(out_csv)
    assert out_df['tp2_hit'].sum() >= int(0.02 * len(out_df))
