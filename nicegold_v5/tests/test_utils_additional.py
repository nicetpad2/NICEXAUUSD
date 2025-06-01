import importlib
import sys
import types
import pandas as pd
import matplotlib.pyplot as plt

import nicegold_v5.utils as utils
import nicegold_v5.wfv as wfv
from nicegold_v5.exit import _rget
from nicegold_v5.deep_model_m1 import LSTMClassifier


def test_split_folds():
    df = pd.DataFrame({'a': range(10)})
    folds = utils.split_folds(df, n_folds=2)
    assert len(folds) == 2
    assert all(len(f) == 5 for f in folds)
    assert folds[0].index[0] == 0
    assert folds[1].index[0] == 0


def test_load_and_save_results(tmp_path):
    df = pd.DataFrame({
        'timestamp': ['2024-01-01 00:00:00', '2024-01-01 01:00:00'],
        'price': [1, 2]
    })
    csv_path = tmp_path / 'data.csv'
    df.to_csv(csv_path, index=False)
    out = utils.load_data(csv_path)
    assert list(out['timestamp']) == list(pd.to_datetime(df['timestamp']))

    trades = pd.DataFrame({'pnl': [1, -0.5]})
    equity = pd.DataFrame({'equity': [100, 101]})
    metrics = utils.summarize_results(trades, equity)
    utils.save_results(trades, equity, metrics, tmp_path)
    files = list(tmp_path.iterdir())
    assert any('trades_' in f.name for f in files)
    assert any('equity_' in f.name for f in files)
    assert any('summary_' in f.name for f in files)


def test_pass_filters():
    ts = pd.Timestamp('2024-01-01 09:00')
    row = pd.Series({'EMA_50_slope': 1.0, 'ATR_14': 1, 'ATR_14_MA50': 1.0}, name=ts)
    assert wfv.pass_filters(row)

    row_bad = pd.Series({'EMA_50_slope': -1.0, 'ATR_14': 10, 'ATR_14_MA50': 1.0}, name=ts)
    assert not wfv.pass_filters(row_bad)


def test_plot_equity(monkeypatch):
    df = pd.DataFrame({'equity_total': [10000, 10050]}, index=pd.date_range('2024-01-01', periods=2, freq='h'))
    called = {'show': False}
    monkeypatch.setattr(plt, 'show', lambda: called.__setitem__('show', True))
    wfv.plot_equity(df)
    assert called['show']


def test_print_qa_summary_empty(capsys):
    trades = pd.DataFrame()
    equity = pd.DataFrame()
    metrics = utils.print_qa_summary(trades, equity)
    out = capsys.readouterr().out
    assert "ไม่มีไม้" in out
    assert metrics == {}


def test_rget_variants():
    row_dict = {'a': 1}
    assert _rget(row_dict, 'a') == 1

    obj = types.SimpleNamespace(b=2)
    assert _rget(obj, 'b') == 2

    series = pd.Series({'c': 3})
    assert _rget(series, 'c') == 3
    assert _rget(series, 'd', default=4) == 4


def test_lstm_forward(monkeypatch):
    import numpy as np

    class DummyModule:
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    class DummyLSTM(DummyModule):
        def __init__(self, input_dim, hidden_dim, num_layers, batch_first=True):
            self.hidden_dim = hidden_dim

        def forward(self, x):
            batch, seq, _ = x.shape
            return np.zeros((batch, seq, self.hidden_dim)), None

    class DummyLinear(DummyModule):
        def __init__(self, in_dim, out_dim):
            self.out_dim = out_dim

        def forward(self, x):
            batch = x.shape[0]
            return np.zeros((batch, self.out_dim))

    dummy_nn = types.SimpleNamespace(Module=DummyModule, LSTM=DummyLSTM, Linear=DummyLinear)
    dummy_torch = types.SimpleNamespace(randn=lambda *s: np.zeros(s), nn=dummy_nn)
    monkeypatch.setitem(sys.modules, 'torch', dummy_torch)
    dm = importlib.reload(importlib.import_module('nicegold_v5.deep_model_m1'))
    model = dm.LSTMClassifier(4, 8, output_dim=1)
    x = dummy_torch.randn(2, 5, 4)
    out = model(x)
    assert out.shape == (2, 1)


def test_get_resource_plan_cpu(monkeypatch):
    dummy_psutil = types.SimpleNamespace(
        virtual_memory=lambda: types.SimpleNamespace(total=8 * 1024**3),
        cpu_count=lambda logical=False: 2,
    )
    dummy_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, 'psutil', dummy_psutil)
    monkeypatch.setitem(sys.modules, 'torch', dummy_torch)

    utils = importlib.reload(importlib.import_module('nicegold_v5.utils'))
    plan = utils.get_resource_plan()
    assert plan['device'] == 'cpu'
    assert plan['batch_size'] == 32


def test_get_resource_plan_no_torch(monkeypatch):
    dummy_psutil = types.SimpleNamespace(
        virtual_memory=lambda: types.SimpleNamespace(total=4 * 1024**3),
        cpu_count=lambda logical=False: 2,
    )
    monkeypatch.setitem(sys.modules, 'psutil', dummy_psutil)
    monkeypatch.setitem(sys.modules, 'torch', None)

    utils = importlib.reload(importlib.import_module('nicegold_v5.utils'))
    plan = utils.get_resource_plan()
    assert plan['device'] == 'cpu'
    assert plan['batch_size'] == 32


def test_safe_calculate_net_change_side():
    df = pd.DataFrame([
        {'entry_price': 100.0, 'exit_price': 105.0, 'side': 'buy'},
        {'entry_price': 120.0, 'exit_price': 115.0, 'side': 'sell'},
    ])
    assert utils.safe_calculate_net_change(df) == 10.0


def test_simulate_tp_exit_sl_and_timeout():
    trades = pd.DataFrame([
        {
            'timestamp': pd.Timestamp('2025-01-01 00:00:00'),
            'entry_price': 100.0,
            'tp1_price': 103.0,
            'tp2_price': 105.0,
            'sl_price': 99.0,
            'direction': 'buy',
        },
        {
            'timestamp': pd.Timestamp('2025-01-01 01:00:00'),
            'entry_price': 100.0,
            'tp1_price': 97.0,
            'tp2_price': 95.0,
            'sl_price': 101.0,
            'direction': 'sell',
        },
    ])
    m1 = pd.DataFrame([
        {'timestamp': '2025-01-01 00:00:00', 'high': 99.5, 'low': 98.5},
        {'timestamp': '2025-01-01 01:00:00', 'high': 100.5, 'low': 99.5},
    ])
    result = utils.simulate_tp_exit(trades, m1, window_minutes=1)
    assert result.loc[0, 'exit_reason'] == 'SL'
    assert result.loc[1, 'exit_reason'] == 'TIMEOUT'


def test_safe_calculate_net_change_missing_cols():
    df = pd.DataFrame()
    assert utils.safe_calculate_net_change(df) == 0.0


def test_convert_thai_datetime_no_columns():
    df = pd.DataFrame({'close': [1.0]})
    result = utils.convert_thai_datetime(df)
    assert result.equals(df)


def test_simulate_tp_exit_extra_branches():
    trades = pd.DataFrame([
        {
            'timestamp': pd.Timestamp('2025-01-02 00:00:00'),
            'entry_price': 100.0,
            'tp1_price': 103.0,
            'tp2_price': 106.0,
            'sl_price': 95.0,
            'direction': 'buy',
        },
        {
            'timestamp': pd.Timestamp('2025-01-02 01:00:00'),
            'entry_price': 100.0,
            'tp1_price': 99.0,
            'tp2_price': 97.0,
            'sl_price': 103.0,
            'direction': 'sell',
        },
        {
            'timestamp': pd.Timestamp('2025-01-02 02:00:00'),
            'entry_price': 100.0,
            'tp1_price': 98.0,
            'tp2_price': 95.0,
            'sl_price': 110.0,
            'direction': 'sell',
        },
    ])

    m1 = pd.DataFrame([
        {'timestamp': '2025-01-02 00:00:30', 'high': 107.0, 'low': 99.0},
        {'timestamp': '2025-01-02 01:00:30', 'high': 104.0, 'low': 99.0},
        {'timestamp': '2025-01-02 02:00:30', 'high': 105.0, 'low': 97.5},
    ])

    result = utils.simulate_tp_exit(trades, m1, window_minutes=1)
    assert result.loc[0, 'exit_reason'] == 'TP2'
    assert result.loc[1, 'exit_reason'] == 'SL'
    assert result.loc[2, 'exit_reason'] == 'TP1'


def test_autotune_resource_cpu(monkeypatch):
    dummy_psutil = types.SimpleNamespace(
        virtual_memory=lambda: types.SimpleNamespace(total=8 * 1024**3)
    )
    dummy_cuda = types.SimpleNamespace(is_available=lambda: False)
    dummy_torch = types.SimpleNamespace(cuda=dummy_cuda)
    monkeypatch.setitem(sys.modules, 'torch', dummy_torch)
    monkeypatch.setitem(sys.modules, 'psutil', dummy_psutil)
    utils_mod = importlib.reload(importlib.import_module('nicegold_v5.utils'))
    device, batch = utils_mod.autotune_resource()
    assert device == 'cpu'
    assert batch >= 64


def test_dynamic_batch_scaler(monkeypatch):
    import importlib as _imp
    utils_mod = _imp.reload(importlib.import_module('nicegold_v5.utils'))

    called = {'count': 0}

    def train_fn(batch_size, **kwargs):
        called['count'] += 1
        if called['count'] == 1:
            raise RuntimeError('OOM')
        return True

    monkeypatch.setattr(utils_mod, 'time', types.SimpleNamespace(sleep=lambda x: None))
    bs = utils_mod.dynamic_batch_scaler(train_fn, batch_start=128, min_batch=64, max_retry=2)
    assert bs == 64


def test_export_audit_report(tmp_path):
    import importlib as _imp
    utils_mod = _imp.reload(importlib.import_module('nicegold_v5.utils'))
    cfg = {"param": 1}
    metrics = {"score": 0.5}
    utils_mod.export_audit_report(cfg, metrics, run_type="QA", version="v28.2.0", fold=1, outdir=str(tmp_path))
    csv_files = list(tmp_path.glob('QA_audit_*.csv'))
    json_files = list(tmp_path.glob('QA_audit_*.json'))
    assert len(csv_files) == 1
    assert len(json_files) == 1
