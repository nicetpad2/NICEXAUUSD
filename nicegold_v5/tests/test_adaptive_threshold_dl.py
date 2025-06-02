import importlib
import sys
import types
import numpy as np
import pandas as pd

def make_stub(monkeypatch):
    torch = types.ModuleType('torch')
    class _Tensor(np.ndarray):
        def __new__(cls, arr, dtype=None):
            return np.array(arr, dtype=dtype).view(cls)
        def size(self, dim=None):
            return self.shape if dim is None else self.shape[dim]
        def unsqueeze(self, dim):
            return np.expand_dims(self, dim).view(_Tensor)
        def to(self, device):
            return self
        def cpu(self):
            return self
    def _tensor(d, dtype=None):
        return _Tensor(np.array(d, dtype=dtype))
    torch.tensor = _tensor
    torch.from_numpy = lambda arr: _tensor(arr, dtype=np.float32)
    torch.zeros = lambda s, dtype=None: _tensor(np.zeros(s, dtype=dtype))
    torch.ones = lambda s, dtype=None: _tensor(np.ones(s, dtype=dtype))
    torch.cat = lambda ts, dim=0: _tensor(np.concatenate([np.asarray(t) for t in ts], axis=dim))
    import contextlib
    @contextlib.contextmanager
    def _no_grad():
        yield
    torch.no_grad = _no_grad
    nn = types.ModuleType('torch.nn')
    class _Module:
        def __call__(self, *a, **k):
            return self.forward(*a, **k)
        def forward(self, *a, **k):
            raise NotImplementedError
        def to(self, device):
            return self
        def eval(self):
            pass
        def train(self):
            pass
        def parameters(self):
            return []
    class _LSTM(_Module):
        def __init__(self, *a, **k):
            self.hidden_size = k.get('hidden_size', 1)
        def forward(self, x):
            b, s, _ = x.shape
            return _tensor(np.zeros((b, s, self.hidden_size))), None
    class _Linear(_Module):
        def __init__(self, in_f, out_f):
            self.out_f = out_f
        def forward(self, x):
            b = x.shape[0]
            return _tensor(np.zeros((b, self.out_f)))
    class _Seq(_Module):
        def __init__(self, *layers):
            self.layers = layers
        def forward(self, x):
            for l in self.layers:
                x = l(x)
            return x
    class _ReLU(_Module):
        def forward(self, x):
            return x
    class _Sigmoid(_Module):
        def forward(self, x):
            return x
    class _BCEWithLogitsLoss(_Module):
        def __call__(self, pred, target):
            return _tensor([0.0])
    nn.Module = _Module
    nn.LSTM = _LSTM
    nn.Linear = _Linear
    nn.Sequential = _Seq
    nn.ReLU = _ReLU
    nn.Sigmoid = _Sigmoid
    nn.BCEWithLogitsLoss = _BCEWithLogitsLoss
    torch.nn = nn
    torch.optim = types.ModuleType('torch.optim')
    torch.optim.SGD = lambda *a, **k: types.SimpleNamespace(step=lambda: None, zero_grad=lambda: None)
    torch.optim.Adam = torch.optim.SGD
    torch.device = lambda x=None: 'cpu'
    torch.utils = types.ModuleType('torch.utils')
    torch.utils.data = types.ModuleType('torch.utils.data')
    class _TensorDataset:
        def __init__(self, *tensors):
            self.tensors = tensors
        def __len__(self):
            return len(self.tensors[0])
        def __getitem__(self, idx):
            return tuple(t[idx] for t in self.tensors)
    class _DataLoader:
        def __init__(self, dataset, batch_size=1, shuffle=False, **kw):
            self.dataset = dataset
            self.bs = batch_size
        def __iter__(self):
            for i in range(0, len(self.dataset), self.bs):
                batch = [self.dataset[j] for j in range(i, min(i+self.bs, len(self.dataset)))]
                xs, ys = zip(*batch)
                yield _tensor(np.stack(xs)), _tensor(np.stack(ys))
        def __len__(self):
            return (len(self.dataset)+self.bs-1)//self.bs
    torch.utils.data.TensorDataset = _TensorDataset
    torch.utils.data.DataLoader = _DataLoader
    torch.cuda = types.ModuleType('torch.cuda')
    torch.cuda.is_available = lambda: False
    torch.cuda.amp = types.ModuleType('torch.cuda.amp')
    @contextlib.contextmanager
    def _autocast(enabled=True):
        yield
    torch.cuda.amp.autocast = _autocast
    torch.cuda.amp.GradScaler = lambda enabled=True: types.SimpleNamespace(scale=lambda x:x, step=lambda opt: None, update=lambda: None)
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'torch.nn', nn)
    monkeypatch.setitem(sys.modules, 'torch.optim', torch.optim)
    monkeypatch.setitem(sys.modules, 'torch.utils', torch.utils)
    monkeypatch.setitem(sys.modules, 'torch.utils.data', torch.utils.data)
    monkeypatch.setitem(sys.modules, 'torch.cuda', torch.cuda)
    monkeypatch.setitem(sys.modules, 'torch.cuda.amp', torch.cuda.amp)


def test_threshold_dataset(tmp_path, monkeypatch):
    make_stub(monkeypatch)
    adaptive = importlib.reload(importlib.import_module('nicegold_v5.adaptive_threshold_dl'))
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=65, freq='h'),
        'gain_z': np.linspace(0,1,65),
        'ema_slope': np.linspace(0,1,65),
        'atr': np.linspace(0,1,65),
        'pnl': np.linspace(0,1,65),
        'max_dd': np.linspace(0,1,65),
        'winrate': np.linspace(0,1,65),
        'optimal_gain_z_thresh': [0.1]*65,
        'optimal_ema_slope_thresh': [0.2]*65,
        'optimal_atr_thresh': [0.3]*65,
    })
    csv = tmp_path / 'wfv_results_fold1.csv'
    df.to_csv(csv, index=False)
    dataset = adaptive.load_wfv_training_data(str(tmp_path), seq_len=60)
    item = dataset[0]
    assert item['seq'].shape == (60,3)
    assert item['feedback'].shape == (3,)
    assert item['target'].shape == (3,)


def test_threshold_dataset_empty(tmp_path, monkeypatch):
    make_stub(monkeypatch)
    adaptive = importlib.reload(importlib.import_module('nicegold_v5.adaptive_threshold_dl'))
    dataset = adaptive.load_wfv_training_data(str(tmp_path), seq_len=60)
    assert len(dataset) == 0


def test_threshold_predictor_forward(monkeypatch):
    make_stub(monkeypatch)
    adaptive = importlib.reload(importlib.import_module('nicegold_v5.adaptive_threshold_dl'))
    model = adaptive.ThresholdPredictor()
    seq = np.zeros((2,60,3), dtype=np.float32)
    fb = np.zeros((2,3), dtype=np.float32)
    out = model(adaptive.torch.from_numpy(seq), adaptive.torch.from_numpy(fb))
    assert out.shape == (2,3)



def test_utils_loaders(tmp_path, monkeypatch):
    make_stub(monkeypatch)
    utils = importlib.reload(importlib.import_module('nicegold_v5.utils'))
    df = pd.DataFrame({'gain_z':[0.1],'ema_slope':[0.2],'atr':[0.3]})
    seq = utils.load_recent_indicators(df, seq_len=5)
    assert seq.shape == (1,5,3)
    vals = utils.load_previous_performance(str(tmp_path))
    assert vals == [0.0,0.0,0.0]
