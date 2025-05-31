import importlib
import types
import sys


def test_get_resource_plan(monkeypatch):
    dummy_psutil = types.SimpleNamespace(
        virtual_memory=lambda: types.SimpleNamespace(total=16 * 1024**3),
        cpu_count=lambda logical=False: 4,
    )
    dummy_cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda idx: "DummyGPU",
        get_device_properties=lambda idx: types.SimpleNamespace(total_memory=16 * 1024**3, multi_processor_count=5),
    )
    dummy_torch = types.SimpleNamespace(
        cuda=dummy_cuda,
        set_float32_matmul_precision=lambda x: None,
    )
    monkeypatch.setitem(sys.modules, 'psutil', dummy_psutil)
    monkeypatch.setitem(sys.modules, 'torch', dummy_torch)

    utils = importlib.reload(importlib.import_module('nicegold_v5.utils'))
    plan = utils.get_resource_plan()
    assert plan['device'] == 'cuda'
    assert plan['gpu'] == 'DummyGPU'
    assert plan['batch_size'] == 384
    assert plan['optimizer'] == 'adamw'
    assert plan['n_folds'] == 7
    assert plan['precision'] == 'float16'
