import pandas as pd
import numpy as np
from nicegold_v5.entry import generate_signals, rsi, generate_signals_qa_clean
from nicegold_v5.risk import (
    calc_lot,
    kill_switch,
    apply_recovery_lot,
    calc_lot_risk,
    get_sl_tp,
    get_sl_tp_recovery,
)
from nicegold_v5.exit import should_exit
from nicegold_v5.backtester import run_backtest
from nicegold_v5.utils import summarize_results, run_auto_wfv
from nicegold_v5.utils import auto_entry_config
from nicegold_v5 import wfv


def sample_df():
    rows = 60
    data = {
        'timestamp': pd.date_range('2024-01-01', periods=rows, freq='D'),
        'open': pd.Series(range(rows)) + 100,
        'high': pd.Series(range(rows)) + 101,
        'low': pd.Series(range(rows)) + 99,
        'close': pd.Series(range(rows)) + 100,
        'gain_z': [0.0] * rows,
        'atr_ma': [1.0] * rows
    }
    return pd.DataFrame(data)


def sample_wfv_df():
    ts = pd.date_range('2024-01-01', periods=40, freq='h')
    df = pd.DataFrame(index=ts)
    df['Open'] = range(100, 140)
    df['feat1'] = [i * 0.1 for i in range(40)]
    df['feat2'] = [i * -0.1 for i in range(40)]
    df['label'] = [0, 1] * 20
    df['EMA_50_slope'] = 1.0
    df['ATR_14'] = 1.0
    df['ATR_14_MA50'] = 0.5
    return df

# New helper with lowercase 'open'
def sample_wfv_df_lower():
    df = sample_wfv_df().rename(columns={'Open': 'open'})
    return df


def sample_qa_df():
    dates = pd.date_range('2024-01-01 09:00', periods=100, freq='h')
    thai_dates = [f"{dt.year + 543}{dt.strftime('%m%d')}" for dt in dates]
    close = np.linspace(100, 110, 100)
    df = pd.DataFrame({
        'Date': thai_dates,
        'Timestamp': dates.strftime('%H:%M:%S'),
        'open': close,
        'high': close + 1,
        'low': close - 1,
        'close': close,
    })
    return df


def test_rsi_vectorized():
    series = pd.Series(range(1, 16))
    result = rsi(series, period=14)
    assert round(result.iloc[-1], 2) == 100.0


def test_generate_signals():
    df = sample_df()
    out = generate_signals(df)
    assert 'entry_signal' in out.columns
    assert 'entry_blocked_reason' in out.columns


def test_generate_signals_with_config():
    df = sample_df()
    df['timestamp'] = df['timestamp'] + pd.Timedelta(hours=14)
    df['gain_z'] = -0.15
    out_default = generate_signals(df)
    out_cfg = generate_signals(df, config={'gain_z_thresh': -0.2})
    assert out_cfg['entry_signal'].notnull().sum() >= out_default['entry_signal'].notnull().sum()


def test_generate_signals_volatility_filter():
    df = sample_df()
    out = generate_signals(df, config={'volatility_thresh': 3.0})
    assert out['entry_signal'].notnull().sum() == 0


def test_generate_signals_session_filter():
    ts_in = pd.date_range('2024-01-01 13:00', periods=30, freq='D')
    df_in = pd.DataFrame({
        'timestamp': ts_in,
        'open': pd.Series(range(30)) + 100,
        'high': pd.Series(range(30)) + 101,
        'low': pd.Series(range(30)) + 99,
        'close': pd.Series(range(30)) + 100,
        'gain_z': [0.1]*30,
        'atr_ma': [1.0]*30,
    })
    out_in = generate_signals(df_in)

    ts_out = pd.date_range('2024-01-01 23:00', periods=30, freq='D')
    df_out = df_in.copy()
    df_out['timestamp'] = ts_out
    out_out = generate_signals(df_out)
    assert out_out['entry_blocked_reason'].str.contains('session').any()


def test_generate_signals_qa_clean():
    df = sample_qa_df()
    out = generate_signals_qa_clean(df)
    assert 'entry_signal' in out.columns
    assert out['entry_signal'].notnull().any()


def test_auto_entry_config():
    df = sample_df()
    df = df.assign(ema_fast=1.0, gain_z=0.0, atr=1.0)
    config = auto_entry_config(df)
    assert 'gain_z_thresh' in config and 'ema_slope_min' in config


def test_calc_lot():
    lot = calc_lot(100)
    assert lot >= 0.01


def test_kill_switch_trigger():
    curve = [100] * 100 + [95, 70]
    assert kill_switch(curve)


def test_kill_switch_waits_min_trades():
    curve = [100, 95, 70]
    assert not kill_switch(curve)


def test_recovery_lot():
    lot = apply_recovery_lot(1000, sl_streak=3, base_lot=0.02)
    assert lot > 0.02


def test_calc_lot_risk_and_sl_tp():
    sl, tp = get_sl_tp(100, 1.0, "Asia", "buy")
    lot = calc_lot_risk(1000, 1.0, 1.5)
    assert sl < 100 and tp > 100
    assert lot >= 0.01


def test_get_sl_tp_recovery():
    sl, tp = get_sl_tp_recovery(100, 1.0, "buy")
    assert round(sl, 2) == 98.2
    assert round(tp, 2) == 101.8


def test_tsl_activation():
    from nicegold_v5 import exit as exit_mod
    exit_mod.MIN_HOLD_MINUTES = 0
    trade = {
        "entry": 100,
        "type": "buy",
        "lot": 0.1,
        "entry_time": pd.Timestamp("2025-01-01 00:00:00"),
    }
    row = {
        "close": 102.0,
        "gain_z": 0.0,
        "atr": 1.0,
        "atr_ma": 1.0,
        "timestamp": pd.Timestamp("2025-01-01 00:05:00"),
    }
    exit_now, reason = should_exit(trade, row)
    assert not exit_now
    assert trade.get("tsl_activated")


def test_should_exit():
    trade = {
        'entry': 100,
        'type': 'buy',
        'lot': 0.1,
        'entry_time': pd.Timestamp('2025-01-01 00:00:00')
    }
    row = {
        'close': 101.5,
        'gain_z': -1,
        'atr': 1.0,
        'atr_ma': 1.5,
        'timestamp': pd.Timestamp('2025-01-01 00:15:00')
    }
    exit_now, reason = should_exit(trade, row)
    assert exit_now


def test_backtester_run():
    df = sample_df()
    df = generate_signals(df)
    trades, equity = run_backtest(df)
    assert isinstance(trades, pd.DataFrame)
    assert isinstance(equity, pd.DataFrame)
    summarize_results(trades, equity)


def test_run_auto_wfv(tmp_path):
    df = sample_df()
    summary = run_auto_wfv(df, outdir=str(tmp_path), n_folds=2)
    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == 2


def test_run_walkforward_backtest():
    df = sample_wfv_df()
    trades = wfv.run_walkforward_backtest(
        df,
        features=['feat1', 'feat2'],
        label_col='label',
        n_folds=2
    )
    assert isinstance(trades, pd.DataFrame)
    assert not trades.empty
    assert 'r_multiple' in trades.columns
    assert 'session' in trades.columns


def test_session_performance():
    df = sample_wfv_df()
    trades = wfv.run_walkforward_backtest(df, ['feat1', 'feat2'], 'label', n_folds=2)
    perf = wfv.session_performance(trades)
    assert 'sum' in perf.columns


def test_run_parallel_wfv(tmp_path, monkeypatch):
    import importlib
    main = importlib.import_module('main')
    df = sample_wfv_df()
    monkeypatch.setattr(main, 'TRADE_DIR', str(tmp_path))
    monkeypatch.setattr(main, 'maximize_ram', lambda: None)
    trades = main.run_parallel_wfv(df, ['Open', 'feat1', 'feat2'], 'label', n_folds=2)
    assert isinstance(trades, pd.DataFrame)

def test_run_parallel_wfv_lowercase(tmp_path, monkeypatch):
    import importlib
    main = importlib.import_module('main')
    df = sample_wfv_df_lower()
    monkeypatch.setattr(main, 'TRADE_DIR', str(tmp_path))
    monkeypatch.setattr(main, 'maximize_ram', lambda: None)
    trades = main.run_parallel_wfv(df, ['Open', 'feat1', 'feat2'], 'label', n_folds=2)
    assert isinstance(trades, pd.DataFrame)


def test_print_qa_summary_and_export(tmp_path):
    from nicegold_v5.utils import print_qa_summary, export_chatgpt_ready_logs, create_summary_dict
    trades = pd.DataFrame({
        'pnl': [1.0, -0.5],
        'lot': [0.1, 0.1],
        'commission': [0.02, 0.02],
        'spread_cost': [0.01, 0.01],
        'slippage_cost': [0.0, 0.0]
    })
    equity = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=2, freq='D'),
        'equity': [100, 100.5]
    })

    metrics = print_qa_summary(trades, equity)
    assert metrics['total_trades'] == 2
    summary = create_summary_dict(trades, equity, file_name="test.csv")
    export_chatgpt_ready_logs(trades, equity, summary, outdir=str(tmp_path))
    files = list(tmp_path.iterdir())
    assert len(files) == 3
