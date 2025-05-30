import importlib
import pandas as pd
import pytest
torch = pytest.importorskip('torch')

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
    plan = {
        'device': 'cpu',
        'gpu': 'CPU',
        'ram': 8.0,
        'threads': 2,
        'batch_size': 64,
        'model_dim': 32,
        'n_folds': 5,
        'optimizer': 'sgd',
        'lr': 0.01,
    }
    monkeypatch.setattr(main, 'get_resource_plan', lambda: plan)

    trades = main.autopipeline()
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
        'ram': 8.0,
        'threads': 2,
        'batch_size': 64,
        'model_dim': 32,
        'n_folds': 5,
        'optimizer': 'sgd',
        'lr': 0.01,
    }
    monkeypatch.setattr(main, 'get_resource_plan', lambda: plan)
    monkeypatch.setattr('nicegold_v5.optuna_tuner.start_optimization', lambda df_feat, n_trials=100: types.SimpleNamespace(best_trial=types.SimpleNamespace(params={})))

    trades = main.autopipeline(mode='ai_master', train_epochs=1)
    out = capsys.readouterr().out
    assert 'AI Master Pipeline' in out
    assert not trades.empty
