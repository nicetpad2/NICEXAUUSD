import importlib

def test_import_order_no_circular():
    cfg = importlib.import_module('nicegold_v5.config')
    entry = importlib.import_module('nicegold_v5.entry')
    assert hasattr(entry, 'generate_signals')
