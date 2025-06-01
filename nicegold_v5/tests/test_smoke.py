import pkgutil
import importlib
import types
import sys

# Ensure torch stubs so imports succeed
try:  # pragma: no cover - use real torch if available
    import torch  # noqa: F401
except Exception:  # pragma: no cover - stub torch package
    torch = types.ModuleType('torch')
    torch.__path__ = []
    torch.float32 = 0.0
    torch.nn = types.ModuleType('torch.nn')
    torch.nn.Module = object
    torch.nn.LSTM = lambda *a, **k: None
    torch.nn.Linear = lambda *a, **k: None
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        empty_cache=lambda: None,
        memory_allocated=lambda: 0,
        get_device_name=lambda idx=0: 'CPU',
        get_device_properties=lambda idx=0: types.SimpleNamespace(total_memory=0, multi_processor_count=0),
    )
    torch.cuda.amp = types.ModuleType('torch.cuda.amp')
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
        def step(self, opt):
            pass
        def update(self):
            pass
    torch.cuda.amp.GradScaler = _GradScaler
    torch.optim = types.ModuleType('torch.optim')
    torch.utils = types.ModuleType('torch.utils')
    torch.utils.data = types.ModuleType('torch.utils.data')
    torch.utils.data.DataLoader = lambda *a, **k: []
    torch.utils.data.TensorDataset = lambda *a, **k: None
    sys.modules['torch'] = torch
    sys.modules['torch.nn'] = torch.nn
    sys.modules['torch.cuda'] = torch.cuda
    sys.modules['torch.cuda.amp'] = torch.cuda.amp
    sys.modules['torch.optim'] = torch.optim
    sys.modules['torch.utils'] = torch.utils
    sys.modules['torch.utils.data'] = torch.utils.data


def test_module_imports():
    pkg = importlib.import_module('nicegold_v5')
    pkg_path = pkg.__path__[0]
    for mod in pkgutil.iter_modules([pkg_path]):
        importlib.import_module(f'{pkg.__name__}.{mod.name}')
