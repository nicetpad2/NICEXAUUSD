import pandas as pd
import types
import importlib

import nicegold_v5.entry as entry
from nicegold_v5.optuna_tuner import objective
import optuna


def test_filter_entry_signals_session():
    df = pd.DataFrame({
        'gain_z': [0.3, 0.5],
        'ema_slope': [0.2, 0.3],
        'atr': [1.1, 1.2],
        'sniper_risk_score': [6.0, 7.0],
    })
    cfg = {
        'gain_z_thresh': 0.1,
        'ema_slope_min': 0.1,
        'atr_thresh': 1.0,
        'sniper_risk_score_min': 5.0,
        'tp1_rr_ratio': 1.2,
        'tp_rr_ratio': 2.0,
        'session_adaptive': True,
    }
    entry.SESSION_CONFIG['London'] = {
        'gain_z_thresh': 0.4,
        'tp1_rr_ratio': 1.5,
        'tp2_rr_ratio': 3.0,
    }
    out = entry.filter_entry_signals(df, cfg, session='London')
    assert len(out) == 1
    row = out.iloc[0]
    assert row['tp1_rr_ratio'] == 1.5 and row['tp2_rr_ratio'] == 3.0


def test_filter_entry_signals_all_filters():
    df = pd.DataFrame({
        'gain_z': [-1, 0.2, 0.2, 0.2],
        'ema_slope': [0.2, 0.0, 0.2, 0.2],
        'atr': [1.2, 1.2, 0.5, 1.2],
        'sniper_risk_score': [6.0, 6.0, 6.0, 4.0],
    })
    cfg = {
        'gain_z_thresh': 0.1,
        'ema_slope_min': 0.1,
        'atr_thresh': 1.0,
        'sniper_risk_score_min': 5.0,
    }
    out = entry.filter_entry_signals(df, cfg)
    assert out.empty


def test_calc_lot_size_branches():
    account = {'init_lot': 0.5}
    cfg = {
        'dynamic_lot': True,
        'lot_win_multiplier': 2.0,
        'lot_loss_multiplier': 0.5,
        'max_lot': 1.0,
    }
    lot_win = entry.calc_lot_size(account, 2, 0, cfg)
    lot_loss = entry.calc_lot_size(account, 0, 2, cfg)
    lot_static = entry.calc_lot_size(account, 0, 0, {'dynamic_lot': False})
    assert lot_win == 1.0
    assert lot_loss == 0.25
    assert lot_static == 0.5


def test_generate_signals_force_entry(monkeypatch):
    df = pd.DataFrame({
        'entry_signal': [None] * 10,
        'close': [1] * 10,
        'high': [1] * 10,
        'low': [1] * 10,
        'volume': [1] * 10,
        'session': ['Asia'] * 10,
    })
    monkeypatch.setattr(entry, 'generate_signals_v8_0', lambda d, config=None: d)
    cfg = {
        'force_entry': True,
        'force_entry_ratio': 0.2,
        'force_entry_seed': 0,
        'force_entry_min_orders': 2,
        'force_entry_side': 'buy',
        'force_entry_session': 'Asia',
    }
    out = entry.generate_signals(df, config=cfg, test_mode=True)
    assert out['entry_signal'].eq('buy').sum() >= 2


def test_generate_signals_force_entry_random(monkeypatch):
    df = pd.DataFrame({
        'entry_signal': [None] * 5,
        'close': [1] * 5,
        'high': [1] * 5,
        'low': [1] * 5,
        'volume': [1] * 5,
    })
    monkeypatch.setattr(entry, 'generate_signals_v8_0', lambda d, config=None: d)
    cfg = {
        'force_entry': True,
        'force_entry_ratio': 0.4,
        'force_entry_min_orders': 2,
        'force_entry_session': 'all',
        }
    out = entry.generate_signals(df, config=cfg, test_mode=True)
    assert out['entry_signal'].notnull().sum() >= 2


def test_generate_signals_force_entry_no_eligible(monkeypatch, capsys):
    df = pd.DataFrame({'entry_signal': ['buy'] * 5, 'close': [1]*5, 'high': [1]*5, 'low':[1]*5, 'volume':[1]*5})
    monkeypatch.setattr(entry, 'generate_signals_v8_0', lambda d, config=None: d)
    cfg = {'force_entry': True}
    out = entry.generate_signals(df, config=cfg, test_mode=True)
    assert out.equals(df)
    assert 'No eligible bars for ForceEntry' in capsys.readouterr().out


def test_generate_signals_v12_test_mode(monkeypatch):
    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=1, freq='min'),
        'close': [1],
        'high': [1],
        'low': [1],
        'volume': [1],
    })
    monkeypatch.setattr(entry, 'sanitize_price_columns', lambda d: d)
    monkeypatch.setattr(entry, 'validate_indicator_inputs', lambda d: None)
    out = entry.generate_signals_v12_0(df, test_mode=True)
    assert 'entry_signal' in out.columns


def test_optuna_objective_trades_empty(monkeypatch):
    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=3, freq='h'),
        'close': [1, 2, 3],
        'high': [1, 2, 3],
        'low': [0, 1, 2],
        'volume': [1, 1, 1],
    })
    import nicegold_v5.optuna_tuner as tuner
    tuner.session_folds = {'London': df}
    monkeypatch.setattr(tuner, 'generate_signals', lambda d, config=None: d)
    monkeypatch.setattr(tuner, 'run_backtest', lambda d: (pd.DataFrame(), pd.DataFrame()))
    trial = optuna.trial.FixedTrial({
        'gain_z_thresh': 0.1,
        'ema_slope_min': 0.1,
        'atr_thresh': 0.5,
        'sniper_risk_score_min': 5.5,
        'tp_rr_ratio': 4.5,
        'volume_ratio': 1.0,
    })
    assert objective(trial) == -999
