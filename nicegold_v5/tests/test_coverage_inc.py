import pandas as pd
import numpy as np
from nicegold_v5.entry import sanitize_price_columns, generate_signals_v9_0
from nicegold_v5.backtester import calc_lot_risk, calc_lot_recovery
from nicegold_v5.fix_engine import auto_fix_logic
from nicegold_v5.qa import (
    detect_noise_exit,
    compute_fold_bias,
    analyze_drawdown,
    detect_fold_drift,
    auto_qa_after_backtest,
    run_qa_guard,
)
import nicegold_v5.qa as qa_mod


def test_sanitize_price_columns_volume_fix(capsys):
    df = pd.DataFrame({
        'close': [1]*10,
        'high': [1]*10,
        'low': [1]*10,
        'open': [1]*10,
        'volume': [np.nan]*10,
    })
    out = sanitize_price_columns(df)
    out_str = capsys.readouterr().out
    assert 'volume เป็น NaN/0' in out_str
    assert out['volume'].eq(1.0).all()


def test_generate_signals_v9_0_calls_v8(monkeypatch):
    called = {}
    def fake_v8(df, config=None):
        called['ok'] = True
        return df.assign(entry_signal='buy')
    monkeypatch.setattr('nicegold_v5.entry.generate_signals_v8_0', fake_v8)
    df = pd.DataFrame({'close':[1], 'high':[1], 'low':[1], 'volume':[1]})
    out = generate_signals_v9_0(df)
    assert called.get('ok')
    assert 'entry_signal' in out.columns


def test_calc_lot_recovery_scale():
    base = calc_lot_risk(1000, 1.0, 1.5)
    lot = calc_lot_recovery(1000, 1.0, 1.5)
    assert lot == min(1.0, round(base * 1.5, 2))


def test_auto_fix_logic_sl_mfe_branch():
    summary = {
        'tp1_count': 0,
        'tp2_count': 0,
        'sl_rate': 0.6,
        'avg_mfe': 0.5,
        'avg_duration': 5.0,
        'net_pnl': 1.0,
    }
    cfg = auto_fix_logic(summary, {'atr_multiplier': 1.0})
    assert cfg['min_hold_minutes'] >= 15
    assert cfg['atr_multiplier'] >= 2.0


def test_detect_noise_exit_missing_cols():
    df = pd.DataFrame({'pnl':[1.0]})
    assert detect_noise_exit(df).empty


def test_compute_fold_bias_empty():
    assert compute_fold_bias(pd.DataFrame()) == -999.0


def test_analyze_drawdown_no_equity():
    result = analyze_drawdown(pd.DataFrame({'a':[1]}))
    assert result == {'max_drawdown_pct': 'N/A'}


def test_detect_fold_drift_insufficient():
    df = detect_fold_drift([{'winrate':50, 'avg_pnl':1, 'sl_count':1}])
    assert df.empty


def test_auto_qa_after_backtest_empty(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(qa_mod, 'QA_BASE_PATH', str(tmp_path))
    auto_qa_after_backtest(pd.DataFrame(), pd.DataFrame({'equity':[1]}))
    assert 'ไม่มีข้อมูล trades' in capsys.readouterr().out


def test_run_qa_guard_with_leaks(capsys):
    trades = pd.DataFrame({
        'pnl':[1.0],
        'mfe':[3.0],
        'duration_min':[3],
        '_id':[1],
        'entry_time':[pd.Timestamp('2025-01-01')],
        'exit_reason':['sl']
    })
    df_feat = pd.DataFrame({'future_price':[1]})
    run_qa_guard(trades, df_feat)
    out = capsys.readouterr().out
    assert 'พบคอลัมน์ต้องสงสัย' in out

from nicegold_v5.entry import simulate_partial_tp_safe

def test_simulate_partial_tp_safe_sell_branches():
    df = pd.DataFrame([
        {'timestamp': pd.Timestamp('2025-01-01 00:00'), 'close': 100.0, 'high': 100.2, 'low': 97.0, 'entry_signal': 'sell', 'atr': 1.0, 'ema_slope': -0.1},
        {'timestamp': pd.Timestamp('2025-01-01 01:00'), 'close': 100.0, 'high': 100.0, 'low': 98.4, 'entry_signal': 'sell', 'atr': 1.0, 'ema_slope': -0.1},
        {'timestamp': pd.Timestamp('2025-01-01 02:00'), 'close': 100.0, 'high': 101.5, 'low': 99.5, 'entry_signal': 'sell', 'atr': 1.0, 'ema_slope': -0.1},
    ])
    trades = simulate_partial_tp_safe(df)
    assert list(trades['exit_reason']) == ['tp2', 'tp1', 'sl']
