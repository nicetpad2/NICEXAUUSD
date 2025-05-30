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

    monkeypatch.setattr('nicegold_v5.ml_dataset_m1.generate_ml_dataset_m1', lambda: None)
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

    trades = main.autopipeline()
    out = capsys.readouterr().out
    assert 'AutoPipeline' in out
    assert not trades.empty
