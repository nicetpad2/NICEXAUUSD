import pandas as pd
import numpy as np
import pytest
torch = pytest.importorskip('torch')

from nicegold_v5.ml_dataset_m1 import generate_ml_dataset_m1
from nicegold_v5.train_lstm_runner import load_dataset, train_lstm


def sample_m1_data(rows=30):
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=rows, freq='min'),
        'high': np.linspace(1, rows, rows),
        'low': np.linspace(0.5, rows - 0.5, rows),
        'close': np.linspace(0.8, rows - 0.2, rows),
        'volume': np.ones(rows)
    })


def test_generate_ml_dataset_m1(tmp_path, monkeypatch):
    df = sample_m1_data()
    csv_path = tmp_path / 'XAUUSD_M1.csv'
    df.to_csv(csv_path, index=False)
    log_dir = tmp_path / 'logs'
    log_dir.mkdir()
    trades = pd.DataFrame({
        'entry_time': df['timestamp'].iloc[10:12].astype(str),
        'exit_reason': ['tp2', 'sl']
    })
    trades_path = log_dir / 'trades_v12_tp1tp2.csv'
    trades.to_csv(trades_path, index=False)
    out_dir = tmp_path / 'data'
    out_dir.mkdir()
    out_csv = out_dir / 'ml_dataset_m1.csv'
    monkeypatch.chdir(tmp_path)
    generate_ml_dataset_m1(str(csv_path), str(out_csv))
    assert out_csv.exists()
    out_df = pd.read_csv(out_csv)
    assert 'tp2_hit' in out_df.columns


def test_generate_ml_dataset_auto_trade_log(tmp_path, monkeypatch):
    df = sample_m1_data()
    csv_path = tmp_path / 'XAUUSD_M1.csv'
    df.to_csv(csv_path, index=False)
    out_dir = tmp_path / 'data'
    out_dir.mkdir()
    out_csv = out_dir / 'ml_dataset_m1.csv'
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('nicegold_v5.entry.generate_signals', lambda df: df)
    monkeypatch.setattr(
        'nicegold_v5.exit.simulate_partial_tp_safe',
        lambda df: pd.DataFrame({'entry_time': df['entry_time'], 'exit_reason': ['tp2'] * len(df)})
    )
    generate_ml_dataset_m1(str(csv_path), str(out_csv))
    assert out_csv.exists()
    assert (tmp_path / 'logs' / 'trades_v12_tp1tp2.csv').exists()


def test_train_lstm(tmp_path, monkeypatch):
    df = sample_m1_data()
    csv_path = tmp_path / 'XAUUSD_M1.csv'
    df.to_csv(csv_path, index=False)
    log_dir = tmp_path / 'logs'
    log_dir.mkdir()
    trades = pd.DataFrame({
        'entry_time': df['timestamp'].iloc[10:12].astype(str),
        'exit_reason': ['tp2', 'sl']
    })
    trades_path = log_dir / 'trades_v12_tp1tp2.csv'
    trades.to_csv(trades_path, index=False)
    out_dir = tmp_path / 'data'
    out_dir.mkdir()
    out_csv = out_dir / 'ml_dataset_m1.csv'
    monkeypatch.chdir(tmp_path)
    generate_ml_dataset_m1(str(csv_path), str(out_csv))
    X, y = load_dataset(str(out_csv), seq_len=5)
    model = train_lstm(X, y, hidden_dim=8, epochs=1, batch_size=2, optimizer_name='adam')
    assert isinstance(model, torch.nn.Module)
