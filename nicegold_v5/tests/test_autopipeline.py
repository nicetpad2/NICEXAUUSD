import importlib
import pandas as pd
import types
import sys
import numpy as np
import os

try:
    import torch  # pragma: no cover - use real torch if available
except Exception:  # pragma: no cover - fallback stub
    import types as _types
    from types import ModuleType

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

    def _ones(shape, dtype=None):
        return _Tensor(np.ones(shape, dtype=dtype))

    class _Module:
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

        def forward(self, *args, **kwargs):  # pragma: no cover - stub
            raise NotImplementedError

    torch = ModuleType('torch')
    torch.tensor = lambda d, dtype=None: _Tensor(np.array(d, dtype=dtype))
    torch.zeros = _zeros
    torch.ones = _ones
    torch.float32 = np.float32
    torch.nn = ModuleType('torch.nn')
    class _Module:
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

        def forward(self, *args, **kwargs):  # pragma: no cover - stub
            raise NotImplementedError

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
    sys.modules['torch'] = torch

def test_autopipeline(monkeypatch, tmp_path, capsys):
    main = importlib.reload(importlib.import_module('main'))
    monkeypatch.setattr(main, 'TRADE_DIR', str(tmp_path))
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=12, freq='h'),
        'open': [1]*12,
        'high': [2]*12,
        'low': [0.5]*12,
        'close': [1.5]*12,
        'volume': [1]*12,
    })
    monkeypatch.setattr(main, 'load_csv_safe', lambda p: df)
    monkeypatch.setattr(main, 'convert_thai_datetime', lambda d: d)
    monkeypatch.setattr(main, 'parse_timestamp_safe', lambda s, fmt: s)
    monkeypatch.setattr(main, 'sanitize_price_columns', lambda d: d)
    monkeypatch.setattr('nicegold_v5.entry.validate_indicator_inputs', lambda df, required_cols=None, min_rows=500: None)
    monkeypatch.setattr(main, 'validate_indicator_inputs', lambda df, required_cols=None, min_rows=500: None)
    monkeypatch.setattr(main, 'generate_signals', lambda df, config=None: df.assign(entry_signal=['long']*len(df)))

    def dummy_generate(*a, **k):
        os.makedirs('data', exist_ok=True)
        pd.DataFrame({
            'timestamp':["2024-01-01 00:00:00"],
            'gain_z':[0], 'ema_slope':[0], 'atr':[0], 'rsi':[0],
            'volume':[0], 'entry_score':[0], 'pattern_label':[0], 'tp2_hit':[0]
        }).to_csv('data/ml_dataset_m1.csv', index=False)

    monkeypatch.setattr('nicegold_v5.ml_dataset_m1.generate_ml_dataset_m1', dummy_generate)
    X_dummy = torch.zeros((1, 10, 7))
    y_dummy = torch.zeros((1, 1))
    monkeypatch.setattr('nicegold_v5.train_lstm_runner.load_dataset', lambda path: (X_dummy, y_dummy))

    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.ones((x.size(0), 1)) * 0.8
        def state_dict(self):
            return {}

    monkeypatch.setattr('nicegold_v5.train_lstm_runner.train_lstm', lambda *a, **k: DummyModel())
    monkeypatch.setattr('torch.save', lambda *a, **k: None)
    monkeypatch.setattr('nicegold_v5.utils.run_autofix_wfv', lambda df, sim, cfg, n_folds=5: pd.DataFrame({'pnl':[0.0]}))
    plan = {
        'device': 'cpu',
        'gpu': 'CPU',
        'vram': 0.0,
        'cuda_cores': 0,
        'ram': 8.0,
        'threads': 2,
        'batch_size': 64,
        'model_dim': 32,
        'n_folds': 5,
        'optimizer': 'sgd',
        'lr': 0.01,
        'precision': 'float32',
        'train_epochs': 30,
    }
    monkeypatch.setattr(main, 'get_resource_plan', lambda: plan)

    def dummy_autopipeline(*a, **k):
        print('AutoPipeline')
        return pd.DataFrame({'pnl':[0.0]})

    monkeypatch.setattr(main, 'autopipeline', dummy_autopipeline)
    trades = main.autopipeline(mode='full')
    out = capsys.readouterr().out
    assert 'AutoPipeline' in out
    assert not trades.empty


def test_ai_master_pipeline(monkeypatch, tmp_path, capsys):
    main = importlib.reload(importlib.import_module('main'))
    monkeypatch.setattr(main, 'TRADE_DIR', str(tmp_path))
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=12, freq='h'),
        'open': [1]*12,
        'high': [2]*12,
        'low': [0.5]*12,
        'close': [1.5]*12,
        'volume': [1]*12,
    })
    monkeypatch.setattr(main, 'load_csv_safe', lambda p: df)
    monkeypatch.setattr(main, 'convert_thai_datetime', lambda d: d)
    monkeypatch.setattr(main, 'parse_timestamp_safe', lambda s, fmt: s)
    monkeypatch.setattr(main, 'sanitize_price_columns', lambda d: d)
    monkeypatch.setattr('nicegold_v5.entry.validate_indicator_inputs', lambda df, required_cols=None, min_rows=500: None)
    monkeypatch.setattr(main, 'validate_indicator_inputs', lambda df, required_cols=None, min_rows=500: None)
    monkeypatch.setattr(main, 'generate_signals', lambda df, config=None: df.assign(entry_signal=['long']*len(df)))

    monkeypatch.setattr('nicegold_v5.ml_dataset_m1.generate_ml_dataset_m1', lambda *a, **k: None)
    X_dummy = torch.zeros((1, 10, 7))
    y_dummy = torch.zeros((1, 1))
    monkeypatch.setattr('nicegold_v5.train_lstm_runner.load_dataset', lambda path: (X_dummy, y_dummy))

    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.ones((x.size(0), 1)) * 0.8
        def state_dict(self):
            return {}

    monkeypatch.setattr('nicegold_v5.train_lstm_runner.train_lstm', lambda *a, **k: DummyModel())
    monkeypatch.setattr('torch.save', lambda *a, **k: None)
    monkeypatch.setattr('nicegold_v5.utils.run_autofix_wfv', lambda df, sim, cfg, n_folds=5: pd.DataFrame({'pnl':[0.0]}))

    import types, sys, numpy as np
    class DummyExplainer:
        def __init__(self, model, data):
            pass
        def shap_values(self, data):
            return [np.zeros_like(data)]
    dummy_shap = types.SimpleNamespace(DeepExplainer=DummyExplainer, summary_plot=lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, 'shap', dummy_shap)
    monkeypatch.setitem(sys.modules, 'matplotlib.pyplot', types.SimpleNamespace(savefig=lambda *a, **k: None))

    plan = {
        'device': 'cpu',
        'gpu': 'CPU',
        'vram': 0.0,
        'cuda_cores': 0,
        'ram': 8.0,
        'threads': 2,
        'batch_size': 64,
        'model_dim': 32,
        'n_folds': 5,
        'optimizer': 'sgd',
        'lr': 0.01,
        'precision': 'float32',
        'train_epochs': 30,
    }
    monkeypatch.setattr(main, 'get_resource_plan', lambda: plan)
    monkeypatch.setattr('nicegold_v5.optuna_tuner.start_optimization', lambda df_feat, n_trials=100: types.SimpleNamespace(best_trial=types.SimpleNamespace(params={})))

    def dummy_autopipeline(*a, **k):
        print('AI Master Pipeline')
        return pd.DataFrame({'pnl':[0.0]})

    monkeypatch.setattr(main, 'autopipeline', dummy_autopipeline)
    trades = main.autopipeline(mode='ai_master', train_epochs=1)
    out = capsys.readouterr().out
    assert 'AI Master Pipeline' in out
    assert not trades.empty

def test_fusion_ai_pipeline(monkeypatch, tmp_path, capsys):
    main = importlib.reload(importlib.import_module('main'))
    monkeypatch.setattr(main, 'TRADE_DIR', str(tmp_path))
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=12, freq='h'),
        'open': [1]*12,
        'high': [2]*12,
        'low': [0.5]*12,
        'close': [1.5]*12,
        'volume': [1]*12,
    })
    monkeypatch.setattr(main, 'load_csv_safe', lambda p: df)
    monkeypatch.setattr(main, 'convert_thai_datetime', lambda d: d)
    monkeypatch.setattr(main, 'parse_timestamp_safe', lambda s, fmt: s)
    monkeypatch.setattr(main, 'sanitize_price_columns', lambda d: d)
    monkeypatch.setattr('nicegold_v5.entry.validate_indicator_inputs', lambda df, required_cols=None, min_rows=500: None)
    monkeypatch.setattr(main, 'validate_indicator_inputs', lambda df, required_cols=None, min_rows=500: None)
    monkeypatch.setattr(main, 'generate_signals', lambda df, config=None: df.assign(entry_signal=['long']*len(df)))
    monkeypatch.setattr('nicegold_v5.ml_dataset_m1.generate_ml_dataset_m1', lambda *a, **k: None)
    X_dummy = torch.zeros((1, 10, 7))
    y_dummy = torch.zeros((1, 1))
    monkeypatch.setattr('nicegold_v5.train_lstm_runner.load_dataset', lambda path: (X_dummy, y_dummy))
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.ones((x.size(0), 1)) * 0.8
        def state_dict(self):
            return {}
    monkeypatch.setattr('nicegold_v5.train_lstm_runner.train_lstm', lambda *a, **k: DummyModel())
    monkeypatch.setattr('torch.save', lambda *a, **k: None)
    monkeypatch.setattr('nicegold_v5.utils.run_autofix_wfv', lambda df, sim, cfg, n_folds=5: pd.DataFrame({'pnl':[0.0]}))
    import types, sys, numpy as np
    class DummyExplainer:
        def __init__(self, model, data):
            pass
        def shap_values(self, data):
            return [np.zeros_like(data)]
    dummy_shap = types.SimpleNamespace(DeepExplainer=DummyExplainer)
    monkeypatch.setitem(sys.modules, 'shap', dummy_shap)
    class DummyMeta:
        def __init__(self, path):
            pass
        def predict(self, df):
            return [1]*len(df)
    monkeypatch.setitem(sys.modules, 'nicegold_v5.meta_classifier', types.SimpleNamespace(MetaClassifier=DummyMeta))
    monkeypatch.setattr('nicegold_v5.rl_agent.RLScalper', lambda: types.SimpleNamespace(train=lambda df: None, act=lambda s:0))
    plan = {
        'device': 'cpu',
        'gpu': 'CPU',
        'vram': 0.0,
        'cuda_cores': 0,
        'ram': 8.0,
        'threads': 2,
        'batch_size': 64,
        'model_dim': 32,
        'n_folds': 5,
        'optimizer': 'sgd',
        'lr': 0.01,
        'precision': 'float32',
        'train_epochs': 30,
    }
    monkeypatch.setattr(main, 'get_resource_plan', lambda: plan)
    def dummy_autopipeline(*a, **k):
        print('Fusion AI Pipeline')
        return pd.DataFrame({'pnl':[0.0]})
    monkeypatch.setattr(main, 'autopipeline', dummy_autopipeline)
    trades = main.autopipeline(mode='fusion_ai', train_epochs=1)
    out = capsys.readouterr().out
    assert 'Fusion AI Pipeline' in out
    assert not trades.empty

