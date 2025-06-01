import pandas as pd
import importlib

from nicegold_v5.ml_dataset_m1 import generate_ml_dataset_m1


def test_generate_ml_dataset_import_fallback(tmp_path, monkeypatch):
    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=200, freq='min'),
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
    monkeypatch.setattr('nicegold_v5.exit.simulate_partial_tp_safe', lambda d, percentile_threshold=75: pd.DataFrame({'entry_time': d.index[:1], 'exit_reason': ['tp2']}))
    monkeypatch.chdir(tmp_path)
    out_csv = tmp_path / 'out.csv'
    generate_ml_dataset_m1(None, str(out_csv), mode="qa")
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
    monkeypatch.setattr('nicegold_v5.exit.simulate_partial_tp_safe', lambda d, percentile_threshold=75: pd.DataFrame({'entry_time': d.index[:1], 'exit_reason': ['tp2']}))
    out_csv = tmp_path / 'data' / 'ml_dataset_m1.csv'
    generate_ml_dataset_m1(str(csv_path), str(out_csv), mode="qa")
    out_df = pd.read_csv(out_csv)
    assert out_df['tp2_hit'].sum() >= int(0.02 * len(out_df))


def test_generate_ml_dataset_prod_fallback(tmp_path, monkeypatch):
    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=200, freq='min'),
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
    monkeypatch.setattr('nicegold_v5.exit.simulate_partial_tp_safe', lambda d, percentile_threshold=75: pd.DataFrame({'entry_time': [], 'exit_reason': []}))
    monkeypatch.setattr('nicegold_v5.wfv.ensure_buy_sell', lambda trades_df, df, fn: trades_df)
    out_csv = tmp_path / 'out' / 'ml_dataset_m1.csv'
    generate_ml_dataset_m1(str(csv_path), str(out_csv), mode='production')
    out_df = pd.read_csv(out_csv)
    assert out_df['tp2_hit'].sum() == 0


def test_generate_ml_dataset_force_near_tp2(tmp_path, monkeypatch):
    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=200, freq='min'),
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

    def fake_simulate(d, percentile_threshold=75):
        start = pd.Timestamp('2025-01-01 02:00:00')
        return pd.DataFrame({
            'entry_time': pd.date_range(start, periods=12, freq='min'),
            'exit_reason': ['tp2'] * 8 + ['sl'] * 4,
            'tp2_price': [2] * 12,
            'entry_price': [1] * 12,
            'mfe': [1] * 8 + [0.95] * 4,
        })

    monkeypatch.setattr('nicegold_v5.exit.simulate_partial_tp_safe', fake_simulate)
    monkeypatch.setattr('nicegold_v5.wfv.ensure_buy_sell', lambda trades_df, df, fn: trades_df)
    out_csv = tmp_path / 'force' / 'ml_dataset_m1.csv'
    generate_ml_dataset_m1(str(csv_path), str(out_csv), mode='production')
    out_df = pd.read_csv(out_csv)
    assert out_df['tp2_hit'].sum() == 10


def test_generate_ml_dataset_entry_time_zero(tmp_path, monkeypatch):
    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=200, freq='min'),
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
    monkeypatch.setattr('nicegold_v5.exit.simulate_partial_tp_safe', lambda d, percentile_threshold=75: pd.DataFrame({'entry_time': ['0'], 'exit_reason': ['tp2']}))
    monkeypatch.setattr('nicegold_v5.wfv.ensure_buy_sell', lambda trades_df, df, fn: trades_df)
    out_csv = tmp_path / 'out_zero' / 'ml_dataset_m1.csv'
    generate_ml_dataset_m1(str(csv_path), str(out_csv), mode='production')
    out_df = pd.read_csv(out_csv)
    assert len(out_df) > 0
