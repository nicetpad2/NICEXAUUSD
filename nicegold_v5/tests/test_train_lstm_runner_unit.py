import importlib
import types
import numpy as np
import pandas as pd
from types import SimpleNamespace

import nicegold_v5.train_lstm_runner as tlr_module


def test_load_dataset_shapes(tmp_path):
    tlr = importlib.reload(tlr_module)
    df = pd.DataFrame({
        'gain_z': range(12),
        'ema_slope': range(12),
        'atr': range(12),
        'rsi': range(12),
        'volume': range(12),
        'entry_score': range(12),
        'pattern_label': range(12),
        'tp2_hit': [0, 1] * 6
    })
    csv = tmp_path / 'ml.csv'
    df.to_csv(csv, index=False)
    X, y = tlr.load_dataset(str(csv), seq_len=5)
    assert X.shape[1:] == (5, 7)
    assert y.shape[1] == 1


def test_train_lstm_stub(monkeypatch):
    tlr = importlib.reload(tlr_module)

    class DummyTensor:
        def __init__(self, arr):
            self.arr = np.array(arr, dtype=np.float32)
        @property
        def shape(self):
            return self.arr.shape
        def to(self, device):
            return self
        def size(self, dim=None):
            return self.arr.shape if dim is None else self.arr.shape[dim]

    class DummyModel:
        def __init__(self, input_dim, hidden_dim):
            self.input_dim = input_dim
        def __call__(self, x):
            return DummyTensor(np.zeros((len(x.arr), 1), dtype=np.float32))
        def parameters(self):
            return []
        def train(self):
            pass

    class DummyOpt:
        def __init__(self, params, lr=0.001):
            pass
        def zero_grad(self):
            pass
        def step(self):
            pass

    class DummyScaler:
        def __init__(self, enabled=True):
            pass
        def scale(self, loss):
            return SimpleNamespace(backward=lambda: None)
        def step(self, opt):
            pass
        def update(self):
            pass

    monkeypatch.setattr(tlr, 'LSTMClassifier', DummyModel)
    monkeypatch.setattr(tlr, 'nn', types.SimpleNamespace(BCELoss=lambda: lambda preds, target: SimpleNamespace(item=lambda: 0.0)))
    monkeypatch.setattr(tlr, 'GradScaler', lambda enabled=True: DummyScaler())
    class DummyCast:
        def __init__(self, enabled=True):
            pass
        def __enter__(self):
            return None
        def __exit__(self, exc_type, exc, tb):
            return False
    monkeypatch.setattr(tlr, 'autocast', lambda enabled=True: DummyCast(enabled))
    monkeypatch.setattr(tlr.optim, 'Adam', lambda params, lr=0.001: DummyOpt(params, lr))
    monkeypatch.setattr(tlr.optim, 'SGD', lambda params, lr=0.001: DummyOpt(params, lr))
    monkeypatch.setattr(tlr, 'DataLoader', lambda dataset, batch_size, shuffle, num_workers, prefetch_factor: [(DummyTensor(np.zeros((1,1,1))), DummyTensor(np.zeros((1,1))))])
    monkeypatch.setattr(tlr, 'TensorDataset', lambda X, y: None)
    monkeypatch.setattr(tlr.torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(tlr, 'torch', types.SimpleNamespace(tensor=lambda d, dtype=None: DummyTensor(d), float32=np.float32, device=lambda x=None: 'cpu', cuda=types.SimpleNamespace(is_available=lambda: False)))

    X = DummyTensor(np.zeros((2,5,1)))
    y = DummyTensor(np.zeros((2,1)))
    model = tlr.train_lstm(X, y, epochs=1, batch_size=1)
    assert isinstance(model, DummyModel)
