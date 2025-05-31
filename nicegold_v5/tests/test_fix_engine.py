import pandas as pd
import pytest
from nicegold_v5.fix_engine import run_self_diagnostic, auto_fix_logic, simulate_and_autofix


def test_run_self_diagnostic_basic():
    trades = pd.DataFrame({
        'exit_reason': ['tp1', 'sl'],
        'mfe': [1.0, 2.0],
        'duration_min': [1.0, 2.0],
        'pnl': [1.0, -0.5],
    })
    df = pd.DataFrame({'close': [1, 2]})
    summary = run_self_diagnostic(trades, df)
    assert summary['tp1_count'] == 1
    assert summary['sl_count'] == 1
    assert summary['total_trades'] == 2
    assert summary['tp_rate'] == pytest.approx(0.5)
    assert summary['sl_rate'] == pytest.approx(0.5)


def test_auto_fix_logic_multiple_rules():
    summary = {
        'tp1_count': 0,
        'tp2_count': 0,
        'sl_rate': 0.4,
        'avg_mfe': 3.0,
        'avg_duration': 1.0,
        'net_pnl': -1.0,
    }
    config = {'tp1_rr_ratio': 1.5, 'atr_multiplier': 1.0}
    new_cfg = auto_fix_logic(summary, config, session='London')
    assert new_cfg['tp1_rr_ratio'] == 1.0
    assert new_cfg['tp2_delay_min'] == 5
    assert new_cfg['atr_multiplier'] == 1.8
    assert new_cfg['min_hold_minutes'] == 10
    assert new_cfg['use_dynamic_tsl']


def test_simulate_and_autofix_basic():
    df = pd.DataFrame({'close': [1, 2]})

    def fake_simulate(data):
        return pd.DataFrame({
            'exit_reason': ['tp1', 'sl'],
            'mfe': [1.0, 0.5],
            'duration_min': [2.0, 1.0],
            'pnl': [1.0, -0.5],
        })

    trades, equity, cfg = simulate_and_autofix(df, fake_simulate, {'atr_multiplier': 1.0}, session='London')
    assert not trades.empty
    assert cfg['atr_multiplier'] == 1.7
