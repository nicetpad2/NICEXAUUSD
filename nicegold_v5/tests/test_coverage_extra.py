import types
import pandas as pd
import importlib
import sys
import pytest
from pathlib import Path

import nicegold_v5.utils as utils
from nicegold_v5.ml_dataset_m1 import generate_ml_dataset_m1
from nicegold_v5.exit import should_exit


def make_sample_csv(path: str):
    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=20, freq='min'),
        'open': [1]*20,
        'high': [1]*20,
        'low': [1]*20,
        'close': [1]*20,
        'volume': [1]*20,
    })
    df.to_csv(path, index=False)


def test_generate_ml_dataset_alt_path(monkeypatch, tmp_path):
    mod_file = importlib.import_module('nicegold_v5.ml_dataset_m1').__file__
    alt_path = Path(mod_file).parent / 'does_not_exist.csv'
    make_sample_csv(alt_path)
    monkeypatch.setattr('nicegold_v5.entry.generate_signals', lambda df, config=None, **kw: df)
    monkeypatch.setattr('nicegold_v5.exit.simulate_partial_tp_safe', lambda df: pd.DataFrame({'entry_time': df['timestamp'].iloc[:2], 'exit_reason': ['tp2', 'sl']}))
    out_csv = tmp_path / 'out.csv'
    generate_ml_dataset_m1('does_not_exist.csv', str(out_csv))
    assert out_csv.exists()
    alt_path.unlink()


def test_generate_ml_dataset_missing_timestamp(tmp_path, monkeypatch):
    df = pd.DataFrame({'open': [1], 'high': [1], 'low': [1], 'close': [1], 'volume': [1]})
    csv_path = tmp_path / 'data.csv'
    df.to_csv(csv_path, index=False)
    monkeypatch.setattr('main.M1_PATH', str(csv_path))
    monkeypatch.setattr('nicegold_v5.entry.generate_signals', lambda df, config=None, **kw: df)
    monkeypatch.setattr('nicegold_v5.exit.simulate_partial_tp_safe', lambda df: pd.DataFrame())
    with pytest.raises(KeyError):
        generate_ml_dataset_m1(None, str(tmp_path / 'out.csv'))


def test_print_qa_summary_no_commission():
    trades = pd.DataFrame({'pnl': [1.0, -0.5], 'lot': [0.1, 0.1]})
    equity = pd.DataFrame({'equity': [100, 101]})
    metrics = utils.print_qa_summary(trades, equity)
    assert metrics['commission_paid'] == 0


def test_simulate_tp_exit_tp_hits():
    trades = pd.DataFrame([
        {'timestamp': pd.Timestamp('2025-01-01 00:00:00'), 'entry_price': 100.0, 'tp1_price': 101.0, 'tp2_price': 102.0, 'sl_price': 99.0, 'direction': 'buy'},
        {'timestamp': pd.Timestamp('2025-01-01 01:00:00'), 'entry_price': 100.0, 'tp1_price': 99.0, 'tp2_price': 98.0, 'sl_price': 101.0, 'direction': 'sell'},
    ])
    m1 = pd.DataFrame([
        {'timestamp': '2025-01-01 00:00:00', 'high': 101.5, 'low': 99.5},
        {'timestamp': '2025-01-01 01:00:00', 'high': 100.0, 'low': 97.5},
    ])
    result = utils.simulate_tp_exit(trades, m1, window_minutes=1)
    assert list(result['exit_reason']) == ['TP1', 'TP2']


def test_get_resource_plan_branches(monkeypatch):
    dummy_psutil = types.SimpleNamespace(
        virtual_memory=lambda: types.SimpleNamespace(total=16 * 1024**3),
        cpu_count=lambda logical=False: 4,
    )
    dummy_cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda idx: 'GPU',
        get_device_properties=lambda idx: types.SimpleNamespace(total_memory=16 * 1024**3, multi_processor_count=3),
    )
    dummy_torch = types.SimpleNamespace(cuda=dummy_cuda, set_float32_matmul_precision=lambda x: None)
    monkeypatch.setitem(sys.modules, 'psutil', dummy_psutil)
    monkeypatch.setitem(sys.modules, 'torch', dummy_torch)
    mod = importlib.reload(importlib.import_module('nicegold_v5.utils'))
    plan_gpu = mod.get_resource_plan()
    assert plan_gpu['batch_size'] == 256

    dummy_psutil2 = types.SimpleNamespace(
        virtual_memory=lambda: types.SimpleNamespace(total=13 * 1024**3),
        cpu_count=lambda logical=False: 2,
    )
    dummy_torch2 = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, 'psutil', dummy_psutil2)
    monkeypatch.setitem(sys.modules, 'torch', dummy_torch2)
    mod = importlib.reload(importlib.import_module('nicegold_v5.utils'))
    plan_cpu = mod.get_resource_plan()
    assert plan_cpu['batch_size'] == 128

    dummy_psutil3 = types.SimpleNamespace(
        virtual_memory=lambda: types.SimpleNamespace(total=10 * 1024**3),
        cpu_count=lambda logical=False: 2,
    )
    monkeypatch.setitem(sys.modules, 'psutil', dummy_psutil3)
    monkeypatch.setitem(sys.modules, 'torch', dummy_torch2)
    mod = importlib.reload(importlib.import_module('nicegold_v5.utils'))
    plan_cpu2 = mod.get_resource_plan()
    assert plan_cpu2['batch_size'] == 64


def test_should_exit_edge_cases():
    now = pd.Timestamp('2025-01-01 06:00:00')
    trade = {'entry': 100, 'type': 'buy', 'lot': 0.1, 'entry_time': now - pd.Timedelta(minutes=400), 'breakeven': True}
    row = {'close': 99.0, 'atr': 1.0, 'atr_ma': 0.5, 'gain_z': -0.1, 'timestamp': now}
    exit_now, reason = should_exit(trade, row)
    assert exit_now and reason == 'timeout_exit'

    trade2 = {'entry': 100, 'type': 'buy', 'lot': 0.1, 'entry_time': now, 'mfe': 4.0}
    row2 = {'close': 99.0, 'atr': 1.0, 'atr_ma': 0.6, 'gain_z': -0.1, 'timestamp': now}
    exit_now2, reason2 = should_exit(trade2, row2)
    assert not exit_now2 and reason2 is None

    trade3 = {'entry': 100, 'type': 'buy', 'lot': 0.1, 'entry_time': now}
    row3 = {'close': 101.2, 'atr': 1.0, 'atr_ma': 0.6, 'gain_z': -0.1, 'timestamp': now}
    exit_now3, reason3 = should_exit(trade3, row3)
    assert exit_now3 and reason3 == 'atr_contract_exit'
