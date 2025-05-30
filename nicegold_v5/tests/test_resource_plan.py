import importlib
import types
import sys


def test_get_resource_plan(monkeypatch):
    dummy_psutil = types.SimpleNamespace(virtual_memory=lambda: types.SimpleNamespace(total=16 * 1024**3))
    dummy_cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda idx: "DummyGPU",
    )
    dummy_torch = types.SimpleNamespace(cuda=dummy_cuda)
    monkeypatch.setitem(sys.modules, 'psutil', dummy_psutil)
    monkeypatch.setitem(sys.modules, 'torch', dummy_torch)
    import multiprocessing
    monkeypatch.setattr(multiprocessing, 'cpu_count', lambda: 4)

    utils = importlib.reload(importlib.import_module('nicegold_v5.utils'))
    plan = utils.get_resource_plan()
    assert plan['device'] == 'cuda'
    assert plan['gpu'] == 'DummyGPU'
    assert plan['batch_size'] == 128
    assert plan['optimizer'] == 'adam'
    assert plan['n_folds'] == 6
