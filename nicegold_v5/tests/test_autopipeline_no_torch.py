import importlib
import pandas as pd


def test_autopipeline_no_torch(monkeypatch, tmp_path, capsys):
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
    monkeypatch.setattr(main, 'validate_indicator_inputs', lambda *a, **k: None)
    monkeypatch.setattr(main, 'generate_signals', lambda df, config=None, **kw: df.assign(entry_signal=['long']*len(df)))
    monkeypatch.setattr('nicegold_v5.utils.run_autofix_wfv', lambda df, sim, cfg, n_folds=5: pd.DataFrame({'pnl':[0.0], 'exit_reason':['tp1']}))
    monkeypatch.setattr(main, 'check_exit_reason_variety', lambda df: True)
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

    trades = main.autopipeline()
    out = capsys.readouterr().out
    assert 'AutoPipeline' in out
    assert not trades.empty
