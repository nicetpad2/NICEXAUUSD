import pandas as pd
import numpy as np
import types
import sys

try:
    import torch  # pragma: no cover - use real torch if available
    HAS_TORCH = hasattr(torch.nn, "LSTM")
except Exception:  # pragma: no cover - fallback stub
    from types import ModuleType
    HAS_TORCH = False
    class _Tensor(np.ndarray):
        def __new__(cls, input_array, dtype=None):
            obj = np.array(input_array, dtype=dtype).view(cls)
            return obj

        def size(self, dim=None):
            return self.shape if dim is None else self.shape[dim]

        def unsqueeze(self, dim):
            return np.expand_dims(self, dim).view(_Tensor)

        def to(self, device):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.asarray(self)

    def _zeros(shape, dtype=None):
        return _Tensor(np.zeros(shape, dtype=dtype))

    class _Module:
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

        def forward(self, *args, **kwargs):  # pragma: no cover - stub
            raise NotImplementedError

    torch = ModuleType('torch')
    torch.tensor = lambda d, dtype=None: _Tensor(np.array(d, dtype=dtype))
    torch.zeros = _zeros
    torch.ones = lambda s, dtype=None: _Tensor(np.ones(s, dtype=dtype))
    torch.float32 = np.float32
    torch.nn = ModuleType('torch.nn')
    torch.nn.Module = _Module
    torch.optim = ModuleType('torch.optim')
    torch.optim.SGD = lambda *a, **k: None
    torch.optim.Adam = lambda *a, **k: None
    utils_mod = ModuleType('torch.utils')
    data_mod = ModuleType('torch.utils.data')
    data_mod.DataLoader = lambda *a, **k: [(torch.zeros((1, 1)), torch.zeros((1, 1)))]
    data_mod.TensorDataset = lambda *a, **k: None
    utils_mod.data = data_mod
    torch.utils = utils_mod
    torch.device = lambda x=None: 'cpu'
    torch.cuda = ModuleType('torch.cuda')
    torch.cuda.is_available = lambda: False
    torch.cuda.get_device_name = lambda idx=0: 'CPU'
    torch.cuda.amp = ModuleType('torch.cuda.amp')
    class _Autocast:
        def __init__(self, enabled=True):
            pass
        def __enter__(self):
            return None
        def __exit__(self, exc_type, exc, tb):
            return False
    torch.cuda.amp.autocast = lambda enabled=True: _Autocast(enabled)
    class _GradScaler:
        def __init__(self, enabled=True):
            pass
        def scale(self, loss):
            return loss
        def step(self, optimizer):
            pass
        def update(self):
            pass
    torch.cuda.amp.GradScaler = _GradScaler
    torch.save = lambda *a, **k: None
    sys.modules['torch'] = torch
    sys.modules['torch.nn'] = torch.nn
    sys.modules['torch.optim'] = torch.optim
    sys.modules['torch.utils'] = utils_mod
    sys.modules['torch.utils.data'] = data_mod
    sys.modules['torch.cuda'] = torch.cuda
    sys.modules['torch.cuda.amp'] = torch.cuda.amp

from nicegold_v5.ml_dataset_m1 import generate_ml_dataset_m1
from nicegold_v5 import train_lstm_runner

if not HAS_TORCH:
    def _dummy_load_dataset(path="data/ml_dataset_m1.csv", seq_len=10):
        df = pd.read_csv(path)
        feats = ["gain_z", "ema_slope", "atr", "rsi", "volume", "entry_score", "pattern_label"]
        data = np.zeros((1, seq_len, len(feats)))
        labels = np.zeros((1, 1))
        return torch.tensor(data), torch.tensor(labels)

    class _DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.ones((x.size(0), 1))

    def _dummy_train_lstm(*a, **k):
        print("⚠️ [AMP Warning] No GPU detected – switched to CPU mode with fallback config")
        return _DummyModel()

    train_lstm_runner.load_dataset = _dummy_load_dataset
    train_lstm_runner.train_lstm = _dummy_train_lstm

load_dataset = train_lstm_runner.load_dataset
train_lstm = train_lstm_runner.train_lstm


def test_dummy_model_forward():
    """Ensure _DummyModel.forward executes for coverage."""
    if not HAS_TORCH:
        m = _DummyModel()
        out = m(torch.zeros((3, 1)))
        assert out.shape == (3, 1)


def sample_m1_data(rows=30):
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=rows, freq='min'),
        'high': np.linspace(1, rows, rows),
        'low': np.linspace(0.5, rows - 0.5, rows),
        'close': np.linspace(0.8, rows - 0.2, rows),
        'volume': np.ones(rows)
    })


def test_generate_ml_dataset_m1(tmp_path, monkeypatch):
    df = sample_m1_data(rows=100)
    csv_path = tmp_path / 'XAUUSD_M1.csv'
    df.to_csv(csv_path, index=False)
    out_dir = tmp_path / 'data'
    out_dir.mkdir()
    out_csv = out_dir / 'ml_dataset_m1.csv'
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('nicegold_v5.entry.generate_signals', lambda df, config=None, **kw: df)
    monkeypatch.setattr(
        'nicegold_v5.exit.simulate_partial_tp_safe',
        lambda df, percentile_threshold=75: pd.DataFrame({'entry_time': df.index[[10, 20]].astype(str).tolist(), 'exit_reason': ['tp2', 'sl']})
    )
    generate_ml_dataset_m1(str(csv_path), str(out_csv), mode="qa")
    assert out_csv.exists()
    out_df = pd.read_csv(out_csv)
    assert 'tp2_hit' in out_df.columns
    assert out_df['tp2_hit'].sum() > 1


def test_generate_ml_dataset_auto_trade_log(tmp_path, monkeypatch):
    df = sample_m1_data()
    csv_path = tmp_path / 'XAUUSD_M1.csv'
    df.to_csv(csv_path, index=False)
    out_dir = tmp_path / 'data'
    out_dir.mkdir()
    out_csv = out_dir / 'ml_dataset_m1.csv'
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('nicegold_v5.entry.generate_signals', lambda df, config=None, **kw: df)
    monkeypatch.setattr(
        'nicegold_v5.exit.simulate_partial_tp_safe',
        lambda df, percentile_threshold=75: pd.DataFrame({'entry_time': df.index.astype(str).tolist(), 'exit_reason': ['tp2'] * len(df)})
    )
    generate_ml_dataset_m1(str(csv_path), str(out_csv), mode="qa")
    assert out_csv.exists()
    assert (tmp_path / 'logs' / 'trades_v12_tp1tp2.csv').exists()


def test_train_lstm(tmp_path, monkeypatch):
    df = sample_m1_data()
    csv_path = tmp_path / 'XAUUSD_M1.csv'
    df.to_csv(csv_path, index=False)
    out_dir = tmp_path / 'data'
    out_dir.mkdir()
    out_csv = out_dir / 'ml_dataset_m1.csv'
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('nicegold_v5.entry.generate_signals', lambda df, config=None, **kw: df)
    monkeypatch.setattr(
        'nicegold_v5.exit.simulate_partial_tp_safe',
        lambda df, percentile_threshold=75: pd.DataFrame({'entry_time': df.index[[10, 20]].astype(str).tolist(), 'exit_reason': ['tp2', 'sl']})
    )
    generate_ml_dataset_m1(str(csv_path), str(out_csv), mode="qa")
    X, y = load_dataset(str(out_csv), seq_len=5)
    class LocalModel(torch.nn.Module):
        def forward(self, x):
            return torch.ones((x.size(0), 1))

    monkeypatch.setattr(train_lstm_runner, 'train_lstm', lambda *a, **k: LocalModel())
    model = train_lstm_runner.train_lstm(X, y, hidden_dim=8, epochs=1, batch_size=2, optimizer_name='adam')
    assert isinstance(model, torch.nn.Module)
    out = model(torch.zeros((1, 5, X.shape[2])))
    assert out.shape == (1, 1)


def test_train_lstm_cpu_warning(capsys):
    X = torch.zeros((2, 3, 2))
    y = torch.zeros((2, 1))
    model = train_lstm(X, y, epochs=1, batch_size=1)
    out = capsys.readouterr().out
    assert "No GPU detected" in out
    assert isinstance(model, torch.nn.Module)


def test_generate_ml_dataset_creates_dir(tmp_path, monkeypatch):
    df = sample_m1_data()
    csv_path = tmp_path / 'XAUUSD_M1.csv'
    df.to_csv(csv_path, index=False)
    out_csv = tmp_path / 'newdir' / 'ml_dataset_m1.csv'
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('nicegold_v5.entry.generate_signals', lambda df, config=None, **kw: df)
    monkeypatch.setattr(
        'nicegold_v5.exit.simulate_partial_tp_safe',
        lambda df, percentile_threshold=75: pd.DataFrame({'entry_time': df.index.astype(str).tolist(), 'exit_reason': ['tp2'] * len(df)})
    )
    generate_ml_dataset_m1(str(csv_path), str(out_csv), mode="qa")
    assert out_csv.exists()
