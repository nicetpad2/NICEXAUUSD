import pandas as pd
import numpy as np
import logging
from nicegold_v5.entry import generate_signals, rsi, generate_signals_qa_clean
from nicegold_v5.backtester import (
    calc_lot,
    kill_switch,
    apply_recovery_lot,
    calc_lot_risk,
    get_sl_tp,
    get_sl_tp_recovery,
    MAX_LOT_CAP,
)
from nicegold_v5.exit import should_exit
from nicegold_v5.backtester import run_backtest
from nicegold_v5.utils import summarize_results, run_auto_wfv
from nicegold_v5.utils import auto_entry_config
from nicegold_v5 import wfv
from main import update_compound_lot, kill_switch_hedge


def sample_df():
    rows = 60
    data = {
        'timestamp': pd.date_range('2024-01-01', periods=rows, freq='D'),
        'open': pd.Series(range(rows)) + 100,
        'high': pd.Series(range(rows)) + 101,
        'low': pd.Series(range(rows)) + 99,
        'close': pd.Series(range(rows)) + 100,
        'volume': [100] * rows,
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

# New helper with only 'close' column
def sample_wfv_df_close_only():
    df = sample_wfv_df().drop(columns=['Open'])
    df['close'] = range(100, 140)
    return df

# Dataset with single class label
def sample_wfv_df_single_class():
    df = sample_wfv_df()
    df['label'] = 0
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
    assert 'lot_suggested' in out.columns
    assert 'entry_score' in out.columns
    assert 'sniper_risk_score' in out.columns
    assert 'session_label' in out.columns
    assert out['lot_suggested'].iloc[0] == 0.05
    assert 'use_be' in out.columns and out['use_be'].all()
    assert 'use_tsl' in out.columns and out['use_tsl'].all()



def test_generate_signals_confirm_zone():
    df = sample_df()
    out = generate_signals(df)
    assert "confirm_zone" in out.columns
    assert "rsi_14" in out.columns
    assert "tp1_rr_ratio" in out.columns
    assert "use_dynamic_tsl" in out.columns

def test_generate_signals_with_config():
    df = sample_df()
    df['timestamp'] = df['timestamp'] + pd.Timedelta(hours=14)
    df['gain_z'] = -0.15
    out_default = generate_signals(df)
    out_cfg = generate_signals(df, config={'gain_z_thresh': -0.2})
    assert 'entry_signal' in out_cfg.columns


def test_generate_signals_volatility_filter():
    df = sample_df()
    out = generate_signals(df, config={'volatility_thresh': 3.0})
    assert 'entry_signal' in out.columns


def test_generate_signals_session_filter():
    ts_in = pd.date_range('2024-01-01 13:00', periods=30, freq='D')
    df_in = pd.DataFrame({
        'timestamp': ts_in,
        'open': pd.Series(range(30)) + 100,
        'high': pd.Series(range(30)) + 101,
        'low': pd.Series(range(30)) + 99,
        'close': pd.Series(range(30)) + 100,
        'volume': [100]*30,
        'gain_z': [0.1]*30,
        'atr_ma': [1.0]*30,
    })
    out_in = generate_signals(df_in)

    ts_out = pd.date_range('2024-01-01 23:00', periods=30, freq='D')
    df_out = df_in.copy()
    df_out['timestamp'] = ts_out
    out_out = generate_signals(df_out)
    assert out_out['entry_blocked_reason'].str.contains('off_session').all()


def test_generate_signals_volume_filter():
    df = sample_df()
    df['volume'] = 10  # lower than rolling mean
    out = generate_signals(df)
    assert (out['entry_signal'].isnull()).all()


def test_generate_signals_qa_clean():
    df = sample_qa_df()
    out = generate_signals_qa_clean(df)
    assert 'entry_signal' in out.columns
    assert out['entry_signal'].notnull().any()


def test_generate_signals_v6_5():
    from nicegold_v5.entry import generate_signals_v6_5
    df = sample_df()
    out = generate_signals_v6_5(df, fold_id=1)
    assert 'entry_signal' in out.columns
    assert 'entry_blocked_reason' in out.columns

def test_generate_signals_v6_5_no_session_filter():
    from nicegold_v5.entry import generate_signals_v6_5
    df = sample_df()
    out = generate_signals_v6_5(df, fold_id=4)
    assert not out['entry_blocked_reason'].str.contains('off_session').any()


def test_generate_signals_v7_1_tp_ratio():
    from nicegold_v5.entry import generate_signals_v7_1
    df = sample_df()
    out = generate_signals_v7_1(df)
    assert 'tp_rr_ratio' in out.columns
    assert out['tp_rr_ratio'].iloc[0] == 4.8


def test_generate_signals_v8_0():
    from nicegold_v5.entry import generate_signals_v8_0
    df = sample_df()
    out = generate_signals_v8_0(df)
    assert 'tp1_rr_ratio' in out.columns
    assert 'use_dynamic_tsl' in out.columns


def test_generate_signals_profit_v10():
    from nicegold_v5.entry import generate_signals_profit_v10
    df = sample_df()
    out = generate_signals_profit_v10(df)
    assert 'entry_signal' in out.columns
    assert 'sniper_score' in out.columns


def test_generate_signals_v11_scalper_m1():
    from nicegold_v5.entry import generate_signals_v11_scalper_m1
    df = sample_df()
    out = generate_signals_v11_scalper_m1(df)
    assert 'entry_signal' in out.columns
    assert 'tp_rr_ratio' in out.columns


def test_generate_signals_v12_0(monkeypatch):
    from nicegold_v5.entry import generate_signals_v12_0
    monkeypatch.setattr('nicegold_v5.entry.validate_indicator_inputs', lambda df, required_cols=None, min_rows=500: None)
    df = sample_df().assign(pattern='inside_bar')
    out = generate_signals_v12_0(df)
    assert 'entry_signal' in out.columns
    assert 'tp1_price' in out.columns


def test_auto_entry_config():
    df = sample_df()
    df = df.assign(ema_fast=1.0, gain_z=0.0, atr=1.0)
    config = auto_entry_config(df)
    assert 'gain_z_thresh' in config and 'ema_slope_min' in config


def test_calc_lot():
    lot = calc_lot(100)
    assert lot >= 0.01
    assert lot <= MAX_LOT_CAP


def test_kill_switch_trigger():
    curve = [100] * 100 + [64]
    assert kill_switch(curve)


def test_kill_switch_waits_min_trades():
    curve = [100, 95, 70]
    assert not kill_switch(curve)


def test_kill_switch_empty_curve():
    assert not kill_switch([])


def test_update_compound_lot():
    lot, milestone = update_compound_lot(300, 100)
    assert lot == 0.02
    assert milestone == 200


def test_kill_switch_hedge_trigger():
    curve = [100, 90, 60]
    assert kill_switch_hedge(curve)


def test_recovery_lot():
    lot = apply_recovery_lot(1000, sl_streak=3, base_lot=0.02)
    assert lot > 0.02


def test_calc_lot_risk_and_sl_tp():
    sl, tp = get_sl_tp(100, 1.0, "Asia", "buy")
    lot = calc_lot_risk(1000, 1.0, 1.5)
    assert sl < 100 and tp > 100
    assert lot >= 0.01
    assert lot <= MAX_LOT_CAP


def test_get_sl_tp_recovery():
    sl, tp = get_sl_tp_recovery(100, 1.0, "buy")
    assert round(sl, 2) == 98.2
    assert round(tp, 2) == 101.8


def test_tsl_activation():
    from nicegold_v5 import exit as exit_mod
    exit_mod.BE_HOLD_MIN = 0
    trade = {
        "entry": 100,
        "type": "buy",
        "lot": 0.1,
        "entry_time": pd.Timestamp("2025-01-01 00:00:00"),
    }
    row = {
        "close": 103.0,
        "high": 103.5,
        "low": 102.0,
        "gain_z": 0.2,
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
    assert not exit_now

def test_no_early_profit_lock():
    trade = {
        'entry': 100,
        'type': 'buy',
        'lot': 0.1,
        'entry_time': pd.Timestamp('2025-01-01 00:00:00')
    }
    row = {
        'close': 100.2,
        'gain_z': -0.1,
        'atr': 1.0,
        'atr_ma': 1.0,
        'timestamp': pd.Timestamp('2025-01-01 00:20:00')
    }
    exit_now, reason = should_exit(trade, row)
    assert not exit_now
    assert reason is None


def test_atr_contract_exit():
    trade = {
        'entry': 100,
        'type': 'buy',
        'lot': 0.1,
        'entry_time': pd.Timestamp('2025-01-01 00:00:00')
    }
    row = {
        'close': 100.6,
        'gain_z': 0.2,
        'atr': 1.0,
        'atr_ma': 0.6,
        'timestamp': pd.Timestamp('2025-01-01 00:20:00')
    }
    exit_now, reason = should_exit(trade, row)
    # [Patch v31.0.0] Momentum guard ผ่อนปรน → คาดว่าออกด้วย atr_contract_exit
    assert exit_now
    assert reason == 'atr_contract_exit'


def test_micro_gain_lock():
    trade = {
        'entry': 100,
        'type': 'buy',
        'lot': 0.1,
        'entry_time': pd.Timestamp('2025-01-01 00:00:00')
    }
    row = {
        'close': 100.35,
        'gain_z': -0.05,
        'atr': 1.0,
        'atr_ma': 1.0,
        'timestamp': pd.Timestamp('2025-01-01 00:20:00')
    }
    exit_now, reason = should_exit(trade, row)
    assert not exit_now
    assert reason is None


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


def test_autorisk_adjust():
    from nicegold_v5.fix_engine import autorisk_adjust

    cfg = {"tp1_rr_ratio": 1.5, "atr_multiplier": 1.0}
    summary = {"tp_rate": 0.1, "sl_rate": 0.5, "net_pnl": -1.0}
    new_cfg = autorisk_adjust(cfg, summary)
    assert new_cfg["tp1_rr_ratio"] == 1.2
    assert new_cfg["atr_multiplier"] == 1.6
    assert new_cfg["enable_be"] and new_cfg["enable_trailing"] and new_cfg["use_dynamic_tsl"]


def test_run_autofix_wfv(tmp_path):
    from nicegold_v5.wfv import run_autofix_wfv

    def fake_sim(df):
        trades = pd.DataFrame({
            "exit_reason": ["sl"] * len(df),
            "mfe": [1.0] * len(df),
            "duration_min": [1.0] * len(df),
        })
        return trades, pd.DataFrame()

    df = sample_wfv_df()
    result = run_autofix_wfv(df, fake_sim, {"tp1_rr_ratio": 1.5}, n_folds=2)
    assert isinstance(result, pd.DataFrame)
    assert set(result["fold"]) == {1, 2}


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

def test_run_walkforward_backtest_single_class():
    df = sample_wfv_df_single_class()
    trades = wfv.run_walkforward_backtest(
        df,
        features=['feat1', 'feat2'],
        label_col='label',
        n_folds=2
    )
    assert isinstance(trades, pd.DataFrame)
    assert not trades.empty


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

def test_run_parallel_wfv_close_fallback(tmp_path, monkeypatch):
    import importlib
    main = importlib.import_module('main')
    df = sample_wfv_df_close_only()
    monkeypatch.setattr(main, 'TRADE_DIR', str(tmp_path))
    monkeypatch.setattr(main, 'maximize_ram', lambda: None)
    trades = main.run_parallel_wfv(df, ['Open', 'feat1', 'feat2'], 'label', n_folds=2)
    assert isinstance(trades, pd.DataFrame)

def test_run_parallel_wfv_exit_variety(tmp_path, monkeypatch):
    import importlib
    main = importlib.import_module('main')
    df = sample_wfv_df_single_class()
    monkeypatch.setattr(main, 'TRADE_DIR', str(tmp_path))
    monkeypatch.setattr(main, 'maximize_ram', lambda: None)
    trades = main.run_parallel_wfv(df, ['Open', 'feat1', 'feat2'], 'label', n_folds=2)
    assert main.check_exit_reason_variety(trades)


def test_run_wfv_with_progress_session_split(monkeypatch):
    import importlib
    main = importlib.import_module('main')
    df = sample_wfv_df()
    monkeypatch.setattr(main, 'maximize_ram', lambda: None)
    trades = main.run_wfv_with_progress(df, ['Open', 'feat1', 'feat2'], 'label')
    assert isinstance(trades, pd.DataFrame)
    assert set(trades['fold'].unique()) <= {'Asia', 'London', 'NY'}


def test_run_wfv_with_progress_lowercase(monkeypatch):
    import importlib
    main = importlib.import_module('main')
    df = sample_wfv_df_lower()
    monkeypatch.setattr(main, 'maximize_ram', lambda: None)
    trades = main.run_wfv_with_progress(df, ['Open', 'feat1', 'feat2'], 'label')
    assert isinstance(trades, pd.DataFrame)


def test_run_wfv_with_progress_close_fallback(monkeypatch):
    import importlib
    main = importlib.import_module('main')
    df = sample_wfv_df_close_only()
    monkeypatch.setattr(main, 'maximize_ram', lambda: None)
    trades = main.run_wfv_with_progress(df, ['Open', 'feat1', 'feat2'], 'label')
    assert isinstance(trades, pd.DataFrame)


def test_split_by_session():
    from nicegold_v5.utils import split_by_session
    df = sample_df()
    sessions = split_by_session(df)
    assert set(sessions.keys()) == {'Asia', 'London', 'NY'}


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


def test_backtest_skip_all_blocked():
    df = sample_df()
    df["entry_signal"] = None
    trades, equity = run_backtest(df)
    assert trades.empty and equity.empty


def test_optuna_objective():
    import nicegold_v5.optuna_tuner as tuner
    df = sample_df()
    study = tuner.start_optimization(df, n_trials=1)
    assert len(study.trials) == 1


def test_run_clean_backtest(monkeypatch, tmp_path):
    import importlib
    main = importlib.import_module('main')

    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=5, freq='min'),
        'open': [1]*5,
        'high': [1]*5,
        'low': [1]*5,
        'close': [1]*5,
        'volume': [100]*5,
    })

    monkeypatch.setattr(main, 'TRADE_DIR', str(tmp_path))
    monkeypatch.setattr(main, 'generate_signals', lambda d, config=None, **kw: d.assign(entry_signal='buy'))
    monkeypatch.setattr(main, 'simulate_partial_tp_safe', lambda d: pd.DataFrame({'pnl': [1]}))
    monkeypatch.setattr('nicegold_v5.utils.print_qa_summary', lambda *a, **k: {})
    monkeypatch.setattr('nicegold_v5.utils.export_chatgpt_ready_logs', lambda *a, **k: None)

    trades = main.run_clean_backtest(df)
    assert isinstance(trades, pd.DataFrame)
    assert not trades.empty


def test_run_clean_backtest_thai_date(monkeypatch, tmp_path):
    import importlib
    main = importlib.import_module('main')

    df = pd.DataFrame({
        'Date': ['25660101', '25660101'],
        'Timestamp': ['00:00:00', '00:01:00'],
        'open': [1, 1],
        'high': [1, 1],
        'low': [1, 1],
        'close': [1, 1],
        'volume': [100, 100],
    })

    monkeypatch.setattr(main, 'TRADE_DIR', str(tmp_path))

    monkeypatch.setattr(main, 'simulate_partial_tp_safe', lambda d: pd.DataFrame({'pnl': [1]}))
    monkeypatch.setattr(main, 'generate_signals', lambda d, config=None, **kw: d.assign(entry_signal='buy'))
    monkeypatch.setattr('nicegold_v5.utils.print_qa_summary', lambda *a, **k: {})
    monkeypatch.setattr('nicegold_v5.utils.export_chatgpt_ready_logs', lambda *a, **k: None)

    trades = main.run_clean_backtest(df)
    assert isinstance(trades, pd.DataFrame)
    assert not trades.empty


def test_run_clean_backtest_lowercase_date(monkeypatch, tmp_path):
    import importlib
    main = importlib.import_module('main')

    df = pd.DataFrame({
        'date': ['25660101', '25660101'],
        'timestamp': ['00:00:00', '00:01:00'],
        'open': [1, 1],
        'high': [1, 1],
        'low': [1, 1],
        'close': [1, 1],
        'volume': [100, 100],
    })

    monkeypatch.setattr(main, 'TRADE_DIR', str(tmp_path))

    monkeypatch.setattr(main, 'simulate_partial_tp_safe', lambda d: pd.DataFrame({'pnl': [1]}))
    monkeypatch.setattr(main, 'generate_signals', lambda d, config=None, **kw: d.assign(entry_signal='buy'))
    monkeypatch.setattr('nicegold_v5.utils.print_qa_summary', lambda *a, **k: {})
    monkeypatch.setattr('nicegold_v5.utils.export_chatgpt_ready_logs', lambda *a, **k: None)

    trades = main.run_clean_backtest(df)
    assert isinstance(trades, pd.DataFrame)
    assert not trades.empty


def test_run_clean_backtest_fallback(monkeypatch, capsys, tmp_path):
    import importlib
    main = importlib.import_module('main')

    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=2, freq='min'),
        'open': [1, 1],
        'high': [1, 1],
        'low': [1, 1],
        'close': [1, 1],
        'volume': [100, 100],
    })

    def fake_generate(d, config=None, **kw):
        return d.assign(entry_signal='buy')

    monkeypatch.setattr(main, 'generate_signals', fake_generate)
    monkeypatch.setattr(main, 'simulate_partial_tp_safe', lambda d: pd.DataFrame({'pnl': [1]}))
    monkeypatch.setattr('nicegold_v5.utils.print_qa_summary', lambda *a, **k: {})
    monkeypatch.setattr('nicegold_v5.utils.export_chatgpt_ready_logs', lambda *a, **k: None)
    monkeypatch.setattr(main, 'TRADE_DIR', str(tmp_path))

    trades = main.run_clean_backtest(df)
    output = capsys.readouterr().out
    assert isinstance(trades, pd.DataFrame)
    assert not trades.empty


def test_run_clean_backtest_signal_guard(monkeypatch, tmp_path):
    import importlib
    main = importlib.import_module('main')

    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=2, freq='min'),
        'open': [1, 1],
        'high': [1, 1],
        'low': [1, 1],
        'close': [1, 1],
        'volume': [100, 100],
    })

    monkeypatch.setattr(main, 'TRADE_DIR', str(tmp_path))
    monkeypatch.setattr(main, 'generate_signals', lambda d, config=None, **kw: d.assign(entry_signal=[None]*len(d)))
    monkeypatch.setattr('nicegold_v5.utils.print_qa_summary', lambda *a, **k: None)
    monkeypatch.setattr('nicegold_v5.utils.export_chatgpt_ready_logs', lambda *a, **k: None)

    with pytest.raises(RuntimeError):
        main.run_clean_backtest(df)


def test_strip_leakage_columns():
    import importlib
    module = importlib.import_module('nicegold_v5.backtester')
    df = pd.DataFrame({
        'a': [1],
        'future_price': [2],
        'next_val': [3],
        'value_lead': [4],
        'target': [0]
    })
    cleaned = module.strip_leakage_columns(df)
    assert 'future_price' not in cleaned.columns
    assert 'next_val' not in cleaned.columns
    assert 'value_lead' not in cleaned.columns
    assert 'target' not in cleaned.columns


def test_run_clean_exit_backtest(monkeypatch, tmp_path):
    import importlib
    module = importlib.import_module('nicegold_v5.backtester')

    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=5, freq='min'),
        'open': [1]*5,
        'high': [1]*5,
        'low': [1]*5,
        'close': [1]*5,
    })
    csv_path = tmp_path / 'data.csv'
    df.to_csv(csv_path, index=False)

    monkeypatch.setattr(module, 'M1_PATH', str(csv_path))
    monkeypatch.setattr(module, 'TRADE_DIR', str(tmp_path))
    monkeypatch.setattr('nicegold_v5.entry.generate_signals_v11_scalper_m1', lambda d, config=None: d.assign(entry_signal='buy'))
    monkeypatch.setattr('nicegold_v5.config.SNIPER_CONFIG_Q3_TUNED', None)
    captured = {}
    def fake_run_backtest(d):
        captured['df'] = d.copy()
        return (
            pd.DataFrame({'pnl': [1]}),
            pd.DataFrame({'timestamp': [pd.Timestamp('2025-01-01')], 'equity': [100]})
        )
    monkeypatch.setattr(module, 'run_backtest', fake_run_backtest)
    monkeypatch.setattr('nicegold_v5.utils.print_qa_summary', lambda *a, **k: {})
    monkeypatch.setattr('nicegold_v5.utils.export_chatgpt_ready_logs', lambda *a, **k: None)

    trades = module.run_clean_exit_backtest()
    assert isinstance(trades, pd.DataFrame)
    assert not trades.empty
    df_passed = captured.get('df')
    assert df_passed is not None
    for col in ['use_be', 'use_tsl', 'tp1_rr_ratio', 'use_dynamic_tsl']:
        assert col in df_passed.columns
        if col in ['use_be', 'use_tsl', 'use_dynamic_tsl']:
            assert df_passed[col].all()
import pandas as pd
from nicegold_v5.entry import generate_signals_v8_0


def test_entry_tier_handles_small_df():
    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=1, freq='min'),
        'open': [1],
        'high': [1],
        'low': [1],
        'close': [1],
        'volume': [1],
    })
    out = generate_signals_v8_0(df)
    assert 'entry_tier' in out.columns
    assert out['entry_tier'].iloc[0] == 'C'


def test_ultra_override_force_signal():
    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=10, freq='min'),
        'open': [1]*10,
        'high': [1]*10,
        'low': [1]*10,
        'close': [1]*10,
        'volume': [0]*10,
    })
    out = generate_signals_v8_0(df, {'gain_z_thresh': -9.5})
    assert out['entry_signal'].iloc[0] == 'buy'
import pandas as pd
from nicegold_v5.backtester import run_backtest


def test_backtest_hit_sl_expected():
    """ทดสอบว่าไม้โดน SL และออกด้วยเหตุผล SL"""
    df = pd.DataFrame({
        "timestamp": pd.date_range(start="2025-01-01", periods=5, freq="min"),
        "open": [100, 100.5, 100.8, 101.0, 101.2],
        "high": [100, 100.6, 100.9, 101.1, 101.3],
        "low": [99.5, 99.8, 100.0, 100.2, 100.4],
        "close": [100.0, 99.0, 98.9, 99.1, 99.0],
        "entry_signal": ["buy", None, None, None, None],
        "atr": [0.5] * 5,
        "atr_ma": [0.6] * 5,
        "gain_z": [0.0] * 5,
    })

    from nicegold_v5 import exit as exit_mod
    exit_mod.BE_HOLD_MIN = 0
    trades, equity = run_backtest(df)
    assert not trades.empty
    assert any(trades["exit_reason"].str.lower().isin(["sl", "recovery_sl"]))
    assert (trades["pnl"] < 0).any()


def test_backtest_avg_profit_over_one():
    df = pd.DataFrame({
        "timestamp": pd.date_range(start="2025-01-01", periods=3, freq="min"),
        "open": [100, 101, 103],
        "high": [101, 103, 105],
        "low": [99, 100, 102],
        "close": [101, 103, 105],
        "entry_signal": ["buy", None, None],
        "atr": [0.5, 0.5, 0.5],
        "atr_ma": [0.5, 0.5, 0.5],
        "gain_z": [0.5, 0.5, 0.5],
    })
    from nicegold_v5 import exit as exit_mod
    exit_mod.BE_HOLD_MIN = 0
    trades, _ = run_backtest(df)
    assert not trades.empty
    assert trades["pnl"].mean() > 1.0


def test_run_backtest_entry_tier_category():
    df = pd.DataFrame({
        "timestamp": pd.date_range(start="2025-01-01", periods=3, freq="min"),
        "close": [100.0, 100.5, 101.0],
        "high": [100.5, 101.0, 101.5],
        "low": [99.5, 100.0, 100.5],
        "entry_signal": ["buy", None, None],
        "atr": [0.5, 0.5, 0.5],
        "atr_ma": [0.5, 0.5, 0.5],
        "gain_z": [0.1, 0.1, 0.1],
        "entry_tier": pd.Categorical(["A", "B", "C"]),
    })
    from nicegold_v5 import exit as exit_mod
    exit_mod.BE_HOLD_MIN = 0
    trades, _ = run_backtest(df)
    assert not trades.empty
import pandas as pd
from nicegold_v5.backtester import (
    calculate_duration,
    calculate_mfe,
    calculate_planned_risk,
)


def test_calculate_duration():
    start = pd.Timestamp('2025-01-01 00:00:00')
    end = pd.Timestamp('2025-01-01 00:05:30')
    assert calculate_duration(start, end) == 5.5


def test_calculate_planned_risk():
    risk = calculate_planned_risk(100.0, 99.0, 0.02)
    assert risk == 0.2


def test_calculate_mfe():
    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=5, freq='min'),
        'high': [1, 2, 3, 4, 5],
        'low': [0, 1, 2, 3, 4],
    })
    start = df['timestamp'].iloc[1]
    end = df['timestamp'].iloc[3]
    mfe_buy = calculate_mfe(start, end, df, 1.5, 'buy')
    mfe_sell = calculate_mfe(start, end, df, 3.5, 'sell')
    assert mfe_buy == 2.5
    assert mfe_sell == 2.5
import pandas as pd
from nicegold_v5.utils import convert_thai_datetime

def test_convert_thai_datetime():
    df = pd.DataFrame({'Date': ['25660416'], 'Timestamp': ['22:00:00']})
    result = convert_thai_datetime(df)
    assert 'timestamp' in result.columns
    assert result.loc[0, 'timestamp'] == pd.Timestamp('2023-04-16 22:00:00')


def test_convert_thai_datetime_lowercase():
    df = pd.DataFrame({'date': ['25660416'], 'timestamp': ['22:00:00']})
    result = convert_thai_datetime(df)
    assert 'timestamp' in result.columns
    assert result.loc[0, 'timestamp'] == pd.Timestamp('2023-04-16 22:00:00')
import pandas as pd
from nicegold_v5.utils import safe_calculate_net_change

def test_safe_calculate_net_change():
    df = pd.DataFrame([
        {'entry_price': 100.0, 'exit_price': 105.0, 'direction': 'buy'},
        {'entry_price': 110.0, 'exit_price': None, 'direction': 'sell'},
        {'entry_price': None, 'exit_price': 120.0, 'direction': 'buy'},
        {'entry_price': 120.0, 'exit_price': 118.0, 'direction': 'sell'},
    ])
    result = safe_calculate_net_change(df)
    assert result == 7.0
import pandas as pd
from nicegold_v5.utils import simulate_tp_exit


def test_simulate_tp_exit():
    trades = pd.DataFrame([
        {
            'timestamp': pd.Timestamp('2025-01-01 00:00:00'),
            'entry_price': 100.0,
            'tp1_price': 101.0,
            'tp2_price': 102.0,
            'sl_price': 99.0,
            'direction': 'buy',
        },
        {
            'timestamp': pd.Timestamp('2025-01-01 00:10:00'),
            'entry_price': 100.0,
            'tp1_price': 99.0,
            'tp2_price': 98.0,
            'sl_price': 101.0,
            'direction': 'sell',
        },
    ])

    m1 = pd.DataFrame([
        {'timestamp': '2025-01-01 00:00:00', 'high': 100.5, 'low': 99.5},
        {'timestamp': '2025-01-01 00:01:00', 'high': 101.5, 'low': 100.0},
        {'timestamp': '2025-01-01 00:10:00', 'high': 100.5, 'low': 99.5},
        {'timestamp': '2025-01-01 00:11:00', 'high': 100.8, 'low': 97.5},
    ])

    result = simulate_tp_exit(trades, m1, window_minutes=2)

    assert result.loc[0, 'exit_reason'] == 'TP1'
    assert result.loc[1, 'exit_reason'] == 'TP2'


def test_build_trade_log_vectorized():
    df = pd.DataFrame({'Open': [100, 101, 102, 103]}, index=pd.date_range('2025-01-01', periods=4, freq='min'))
    position = {
        'entry': 100,
        'sl': 99,
        'tp': 102,
        'lot': 0.1,
        'side': 'buy',
        'entry_time': df.index[0],
        'commission': 0.02,
    }
    timestamp = df.index[-1]
    price = df.loc[timestamp, 'Open']
    trade = wfv.build_trade_log(position, timestamp, price, False, False, 1000, 0, df)
    assert trade['break_even_min'] == 1.0
    assert trade['mfe'] == pytest.approx(2.98, rel=1e-2)
import pandas as pd
import warnings
import importlib

def test_no_datetime_warning():
    main = importlib.import_module('main')
    df = pd.DataFrame({'timestamp': ['2025-01-01 00:00:00', '2025-01-01 01:00:00']})
    with warnings.catch_warnings(record=True) as w:
        df['timestamp'] = pd.to_datetime(
            df['timestamp'], format=main.DATETIME_FORMAT, errors='coerce'
        )
    assert len(w) == 0
import pandas as pd
import pytest
from nicegold_v5.entry import generate_signals_v8_0


def test_entry_blocked_reason_length_check(monkeypatch):
    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=3, freq='min'),
        'open': [1, 1, 1],
        'high': [1, 1, 1],
        'low': [1, 1, 1],
        'close': [1, 1, 1],
        'volume': [1, 1, 1],
    })

    original_reset_index = pd.Series.reset_index

    def fake_reset_index(self, level=None, drop=False, name=None, inplace=False):
        result = original_reset_index(self, level=level, drop=drop, name=name, inplace=False)
        return result.iloc[:-1]

    monkeypatch.setattr(pd.Series, "reset_index", fake_reset_index)
    with pytest.raises(ValueError):
        generate_signals_v8_0(df)

import pandas as pd
from nicegold_v5.entry import generate_signals_v8_0


def test_entry_blocked_reason_empty_df():
    df = pd.DataFrame({
        'timestamp': pd.to_datetime([]),
        'open': [],
        'high': [],
        'low': [],
        'close': [],
        'volume': [],
    })
    out = generate_signals_v8_0(df)
    assert 'entry_blocked_reason' in out.columns
    assert len(out) == 0
import pandas as pd
import pytest
from nicegold_v5.entry import validate_indicator_inputs


def test_validate_inputs_missing_column():
    df = pd.DataFrame({'close': [1, 2], 'high': [1, 2], 'low': [1, 2]})
    with pytest.raises(RuntimeError):
        validate_indicator_inputs(df)


def test_validate_inputs_min_rows(capsys):
    df = pd.DataFrame({'close': [1, 2], 'high': [1, 2], 'low': [1, 2], 'volume': [1, 2]})
    with pytest.raises(RuntimeError):
        validate_indicator_inputs(df, min_rows=5)
    out = capsys.readouterr().out
    assert 'Preview' in out


def test_validate_inputs_pass(capsys):
    df = pd.DataFrame({'close': range(5), 'high': range(5), 'low': range(5), 'volume': range(5)})
    validate_indicator_inputs(df, min_rows=5)
    assert '✅ ตรวจข้อมูลก่อนเข้า indicator' in capsys.readouterr().out


def test_validate_inputs_replace_inf(capsys):
    df = pd.DataFrame({
        'close': [1, np.inf, 2],
        'high': [1, 1, 1],
        'low': [1, 1, 1],
        'volume': [1, 1, 1]
    })
    validate_indicator_inputs(df, min_rows=2)
    out = capsys.readouterr().out
    assert 'เหลือ 2 row' in out
import pandas as pd
from nicegold_v5.entry import sanitize_price_columns


def test_sanitize_conversion_and_logging(capsys):
    df = pd.DataFrame({
        'close': ['1', '2'],
        'high': ['3', '4'],
        'low': ['5', '6'],
        'open': ['7', '8'],
        'volume': ['9', '10'],
    })
    out = sanitize_price_columns(df)
    assert out['close'].tolist() == [1.0, 2.0]
    assert pd.api.types.is_numeric_dtype(out['high'])
    log = capsys.readouterr().out
    assert 'Sanitize Columns' in log


def test_sanitize_nan_count(capsys):
    df = pd.DataFrame({
        'close': ['a', '1'],
        'high': ['b', '2'],
        'low': ['c', '3'],
        'open': ['d', '4'],
        'volume': ['e', '5'],
    })
    out = sanitize_price_columns(df)
    assert out['close'].isna().sum() == 1
    log = capsys.readouterr().out
    assert 'close: 1 NaN' in log


def test_sanitize_handle_commas():
    df = pd.DataFrame({
        'close': ['1,000', '2,000'],
        'high': ['3,000', '4,000'],
        'low': ['5,000', '6,000'],
        'open': ['7,000', '8,000'],
        'volume': ['9,000', '10,000'],
    })
    out = sanitize_price_columns(df)
    assert out['close'].iloc[0] == 1000.0
    assert out['volume'].iloc[1] == 10000.0

import pandas as pd
from nicegold_v5.entry import (
    apply_tp_logic,
    generate_entry_signal,
    session_filter,
    trade_log_fields,
)


def test_apply_tp_logic():
    tp1, tp2 = apply_tp_logic(100, 'buy', sl_distance=5)
    assert tp1 == 106.0
    assert tp2 == 110.0


def test_generate_entry_signal():
    logs = []
    row = {
        'rsi': 25,
        'pattern': 'inside_bar',
        'timestamp': pd.Timestamp('2025-01-01'),
        'close': 100.0,
        'session': 'London',
    }
    signal = generate_entry_signal(row, logs)
    assert signal == 'RSI_InsideBar'
    assert logs and logs[0]['signal'] == 'RSI_InsideBar'


def test_generate_entry_signal_sell():
    logs = []
    row = {
        'rsi': 75,
        'pattern': 'inside_bar',
        'timestamp': pd.Timestamp('2025-01-01'),
        'close': 100.0,
        'session': 'London',
    }
    signal = generate_entry_signal(row, logs)
    assert signal == 'RSI70_InsideBar'
    assert logs and logs[0]['signal'] == 'RSI70_InsideBar'


def test_generate_entry_signal_bearish_patterns():
    logs = []
    row_qm = {
        'pattern': 'qm_bearish',
        'timestamp': pd.Timestamp('2025-01-01'),
        'close': 100.0,
        'session': 'London',
    }
    assert generate_entry_signal(row_qm, logs) == 'QM_Bearish'

    logs.clear()
    row_engulf = {
        'pattern': 'bearish_engulfing',
        'timestamp': pd.Timestamp('2025-01-01'),
        'close': 100.0,
        'session': 'London',
    }
    assert generate_entry_signal(row_engulf, logs) == 'BearishEngulfing'

    logs.clear()
    row_fb = {
        'gain_z': -0.5,
        'ema_slope': -0.1,
        'timestamp': pd.Timestamp('2025-01-01'),
        'close': 100.0,
        'session': 'London',
    }
    assert generate_entry_signal(row_fb, logs) == 'fallback_sell'


def test_session_filter():
    row_block = {'session': 'NY', 'ny_sl_count': 4}
    row_allow = {'session': 'NY', 'ny_sl_count': 1}
    # [Patch v31.0.0] ปิด session_filter ชั่วคราว → ควรผ่านทุกกรณี
    assert session_filter(row_block)
    assert session_filter(row_allow)


def test_trade_log_fields():
    required = {
        'tp1_price',
        'tp2_price',
        'mfe',
        'duration_min',
        'entry_tier',
        'signal_name',
        'pnl',  # [Patch v12.8.3]
        'planned_risk',  # [Patch v12.8.3]
        'r_multiple',  # [Patch v12.8.3]
    }
    assert required.issubset(set(trade_log_fields))


def test_simulate_trades_with_tp():
    from nicegold_v5.entry import simulate_trades_with_tp

    df = pd.DataFrame([
        {
            'timestamp': pd.Timestamp('2025-01-01 00:00:00'),
            'close': 100.0,
            'high': 120.0,
            'low': 96.0,
            'signal': 'long',
            'session': 'London',
            'rsi': 25,
            'pattern': 'inside_bar',
        }
    ])

    trades, logs = simulate_trades_with_tp(df)
    assert len(trades) == 1
    assert len(logs) == 1
    trade = trades[0]
    assert trade['tp1_price'] > trade['entry_price']
    assert trade['tp2_price'] > trade['tp1_price']
    assert trade['tp2_price'] == 115.0
    assert trade['exit_reason'] == 'tp2'
    assert trade['entry_tier'] == 'C'
    assert trade['signal_name'] == 'RSI_InsideBar'
    assert isinstance(trade['mfe'], float)
    assert trade['pnl'] > 0  # [Patch v12.8.3]
    assert trade['planned_risk'] == 0.5  # [Patch v12.8.3]
    assert trade['r_multiple'] > 2.0  # [Patch v12.8.3]


def test_simulate_trades_with_tp_skip_tp1():
    from nicegold_v5.entry import simulate_trades_with_tp

    df = pd.DataFrame([
        {
            'timestamp': pd.Timestamp('2025-01-01 00:00:00'),
            'close': 100.0,
            'high': 120.0,
            'low': 96.0,
            'signal': 'long',
            'session': 'London',
            'rsi': 25,
            'pattern': 'inside_bar',
            'entry_score': 5.0,
            'mfe': 4.0,
        }
    ])

    trades, _ = simulate_trades_with_tp(df)
    assert trades[0]['exit_reason'] == 'tp2'


def test_simulate_trades_with_tp_dynamic_tsl():
    from nicegold_v5.entry import simulate_trades_with_tp

    data = [
        {
            'timestamp': pd.Timestamp('2025-01-01 00:00:00'),
            'close': 100.0,
            'high': 105.0,
            'low': 96.0,
            'signal': 'long',
            'session': 'London',
            'rsi': 25,
            'pattern': 'inside_bar',
            'atr': 1.0,
            'entry_score': 5.0,
            'mfe': 4.0,
        },
        {'timestamp': pd.Timestamp('2025-01-01 00:10:00'), 'close': 106.0, 'high': 107.0, 'low': 96.0, 'session': 'NY', 'ny_sl_count': 4},
        {'timestamp': pd.Timestamp('2025-01-01 00:20:00'), 'close': 110.0, 'high': 111.0, 'low': 101.0, 'session': 'NY', 'ny_sl_count': 4},
        {'timestamp': pd.Timestamp('2025-01-01 00:30:00'), 'close': 108.0, 'high': 114.9, 'low': 107.0, 'session': 'NY', 'ny_sl_count': 4},
        {'timestamp': pd.Timestamp('2025-01-01 00:40:00'), 'close': 114.0, 'high': 114.0, 'low': 110.0, 'session': 'NY', 'ny_sl_count': 4},
    ]
    df = pd.DataFrame(data)

    trades, _ = simulate_trades_with_tp(df)
    assert trades[0]['exit_reason'] == 'tsl_exit'


def test_parse_timestamp_safe_logs(capsys):
    from nicegold_v5.utils import parse_timestamp_safe

    series = pd.Series([
        'bad',
        '2025-01-01 00:00:00',
        '2025/01/01 01:00:00'
    ])
    result = parse_timestamp_safe(series, '%Y-%m-%d %H:%M:%S')
    out = capsys.readouterr().out
    assert 'parse_timestamp_safe()' in out
    assert result.notna().sum() == 1


def test_load_csv_safe_fallback():
    import importlib
    main = importlib.import_module('main')
    path = '/invalid/path/XAUUSD_M1.csv'
    df = main.load_csv_safe(path)
    assert not df.empty


def test_prepare_csv_auto(monkeypatch):
    import importlib
    main = importlib.reload(importlib.import_module('main'))
    utils = importlib.reload(importlib.import_module('nicegold_v5.utils'))
    df = pd.DataFrame({
        'timestamp': ['2025-01-01 00:00:00', '2025-01-01 01:00:00'],
        'open': ['1', '2'],
        'high': ['2', '3'],
        'low': ['0', '1'],
        'close': ['1', '2'],
        'volume': ['1', '1'],
    })
    monkeypatch.setattr(main, 'load_csv_safe', lambda p: df)
    monkeypatch.setattr(main, 'convert_thai_datetime', lambda d: d)
    monkeypatch.setattr(main, 'parse_timestamp_safe', lambda s, fmt: pd.to_datetime(s))
    monkeypatch.setattr(main, 'sanitize_price_columns', lambda d: d)
    monkeypatch.setattr(main, 'validate_indicator_inputs', lambda df, required_cols=None, min_rows=500: None)
    out = utils.prepare_csv_auto('dummy.csv')
    assert len(out) == 2
    assert pd.api.types.is_datetime64_any_dtype(out['timestamp'])

def test_detect_session_auto():
    from nicegold_v5.exit import detect_session_auto
    assert detect_session_auto(pd.Timestamp('2025-01-01 04:00')) == 'Asia'
    assert detect_session_auto(pd.Timestamp('2025-01-01 10:00')) == 'London'
    assert detect_session_auto(pd.Timestamp('2025-01-01 16:00')) == 'NY'
    assert detect_session_auto(pd.Timestamp('2025-01-01 02:00')) == 'Asia'
    # รองรับ dict input
    assert detect_session_auto({'timestamp': pd.Timestamp('2025-01-01 20:00')}) == 'NY'


def test_simulate_partial_tp_safe_session(monkeypatch):
    from nicegold_v5.exit import simulate_partial_tp_safe

    df = pd.DataFrame([
        {'timestamp': pd.Timestamp('2025-01-01 10:00'), 'close': 100.0, 'high': 100.6, 'low': 99.5, 'entry_signal': 'buy', 'atr': 1.0},
        {'timestamp': pd.Timestamp('2025-01-01 10:05'), 'close': 101.6, 'high': 101.6, 'low': 100.5, 'entry_signal': 'buy', 'atr': 1.0},
        {'timestamp': pd.Timestamp('2025-01-01 10:10'), 'close': 100.5, 'high': 100.7, 'low': 100.4, 'entry_signal': 'buy', 'atr': 1.0},
        {'timestamp': pd.Timestamp('2025-01-01 10:20'), 'close': 101.8, 'high': 101.9, 'low': 100.9, 'entry_signal': 'buy', 'atr': 1.0},
        {'timestamp': pd.Timestamp('2025-01-01 10:30'), 'close': 100.8, 'high': 101.2, 'low': 100.6, 'entry_signal': 'buy', 'atr': 1.0},
    ])

    trades = simulate_partial_tp_safe(df)
    assert not trades.empty
    assert 'session' in trades.columns
    assert trades['session'].iloc[0] == 'London'
    assert trades['exit_reason'].iloc[0] in {'tp1', 'tp2', 'sl'}


def test_entry_simulate_partial_tp_safe_basic():
    from nicegold_v5 import entry

    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=1, freq='min'),
        'close': [100.0],
        'high': [101.6],
        'low': [99.0],
        'entry_signal': ['buy'],
        'atr': [1.0],
        'ema_slope': [0.1],
    })

    trades = entry.simulate_partial_tp_safe(df)
    assert isinstance(trades, pd.DataFrame)
    assert len(trades) == 1
    assert trades['exit_reason'].iloc[0] == 'tp1'


def test_simulate_partial_tp_safe_low_atr_high_gainz():
    from nicegold_v5.exit import simulate_partial_tp_safe

    df = pd.DataFrame({
        'timestamp': [pd.Timestamp('2025-01-01 00:00'),
                     pd.Timestamp('2025-01-01 00:01'),
                     pd.Timestamp('2025-01-01 00:16')],
        'close': [100, 100.5, 99.0],
        'high': [100.2, 100.7, 100.0],
        'low': [99.8, 100.3, 98.8],
        'entry_signal': ['buy', 'buy', 'buy'],
        'atr': [0.1, 0.1, 0.1],
        'gain_z_entry': [0.5, 0.5, 0.5],
    })

    trades = simulate_partial_tp_safe(df)
    assert not trades.empty


def test_generate_signals_disable_buy_volume_guard(monkeypatch):
    """เงื่อนไข Ultra override sell ควรทำงานแม้ disable_buy"""
    from nicegold_v5.entry import generate_signals_v12_0
    monkeypatch.setattr('nicegold_v5.entry.validate_indicator_inputs', lambda df, required_cols=None, min_rows=500: None)

    rows = 25
    close = np.linspace(100, 125, rows)
    close[20] = close[19] - 5  # force negative gain_z at index 20
    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=rows, freq='min'),
        'close': close,
        'high': np.linspace(100.5, 125.5, rows),
        'low': np.linspace(99.5, 124.5, rows),
        'volume': [10.0] * rows,
        'pattern': ['inside_bar'] + [None]*19 + ['qm_bearish'] + [None]*(rows-21),
        'gain_z': [0.1]*20 + [-0.02] + [0.0]*(rows-21),
        'entry_score': [4.0]*20 + [4.5] + [4.0]*(rows-21),
    })

    config = {'disable_buy': True, 'min_volume': 0.5}
    out = generate_signals_v12_0(df, config)
    assert config['disable_buy'] is False
    assert out['entry_signal'].iloc[20] == 'sell'


def test_generate_signals_disable_sell_override(monkeypatch):
    """ควรบังคับเปิดฝั่ง SELL แม้ตั้ง disable_sell=True"""
    from nicegold_v5.entry import generate_signals_v12_0
    monkeypatch.setattr('nicegold_v5.entry.validate_indicator_inputs', lambda df, required_cols=None, min_rows=500: None)

    rows = 25
    close = np.linspace(100, 125, rows)
    close[20] = close[19] - 5
    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=rows, freq='min'),
        'close': close,
        'high': np.linspace(100.5, 125.5, rows),
        'low': np.linspace(99.5, 124.5, rows),
        'volume': [10.0] * rows,
        'pattern': ['inside_bar'] + [None]*19 + ['qm_bearish'] + [None]*(rows-21),
        'gain_z': [0.1]*20 + [-0.02] + [0.0]*(rows-21),
        'entry_score': [4.0]*20 + [4.5] + [4.0]*(rows-21),
    })

    config = {'disable_sell': True}
    out = generate_signals_v12_0(df, config)
    assert config['disable_sell'] is False
    assert out['entry_signal'].iloc[20] == 'sell'


def test_generate_signals_v12_0_ultra_override(monkeypatch):
    """ตรวจสอบ ultra override sell ทำงานเมื่อ gain_z ติดลบและ entry_score สูง"""
    from nicegold_v5.entry import generate_signals_v12_0
    monkeypatch.setattr(
        'nicegold_v5.entry.validate_indicator_inputs',
        lambda df, required_cols=None, min_rows=500: None,
    )

    rows = 25
    close2 = np.linspace(100, 125, rows)
    close2[20] = close2[19] - 5
    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=rows, freq='min'),
        'close': close2,
        'high': np.linspace(100.5, 125.5, rows),
        'low': np.linspace(99.5, 124.5, rows),
        'volume': [50]*20 + [60] + [50]*(rows-21),
        'pattern': [None]*20 + [None] + [None]*(rows-21),
        'gain_z': [0.0]*20 + [-0.05] + [0.0]*(rows-21),
        'entry_score': [4.0]*20 + [4.5] + [4.0]*(rows-21),
    })

    out = generate_signals_v12_0(df, {'volume_ratio': 0.05})
    assert out.loc[20, 'entry_signal'] == 'sell'


def test_generate_signals_v12_0_fallback_sell(monkeypatch):
    """ตรวจสอบ fallback momentum sell ถูกบล็อกเมื่อ RSI ต่ำ"""
    from nicegold_v5.entry import generate_signals_v12_0
    monkeypatch.setattr(
        'nicegold_v5.entry.validate_indicator_inputs',
        lambda df, required_cols=None, min_rows=500: None,
    )

    rows = 25
    close = np.linspace(130, 100, rows)
    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=rows, freq='min'),
        'close': close,
        'high': np.array(close) + 0.5,
        'low': np.array(close) - 0.5,
        'volume': [100] * rows,
        'pattern': [None] * rows,
        'rsi': [50] * rows,
        'confirm_zone': [True] * rows,
    })

    out = generate_signals_v12_0(df)
    assert out['entry_signal'].isnull().all()


def test_simulate_partial_tp_safe_be_exit():
    from nicegold_v5.exit import simulate_partial_tp_safe

    df = pd.DataFrame([
        {'timestamp': pd.Timestamp('2025-01-01 00:00'), 'close': 100.0, 'high': 100.2, 'low': 99.8, 'entry_signal': 'sell', 'atr': 1.0},
        {'timestamp': pd.Timestamp('2025-01-01 00:05'), 'close': 97.0, 'high': 97.5, 'low': 96.8, 'entry_signal': 'sell', 'atr': 1.0},
        {'timestamp': pd.Timestamp('2025-01-01 00:10'), 'close': 97.8, 'high': 98.0, 'low': 97.5, 'entry_signal': 'sell', 'atr': 1.0},
        {'timestamp': pd.Timestamp('2025-01-01 00:20'), 'close': 100.1, 'high': 100.1, 'low': 99.9, 'entry_signal': 'sell', 'atr': 1.0},
    ])

    trades = simulate_partial_tp_safe(df)
    assert not trades.empty


def test_generate_pattern_signals():
    from nicegold_v5.entry import generate_pattern_signals

    df = pd.DataFrame([
        {"open": 1.2, "close": 1.0, "high": 1.25, "low": 0.95},  # red
        {"open": 0.9, "close": 1.3, "high": 1.35, "low": 0.85},  # bullish engulf
        {"open": 1.4, "close": 1.5, "high": 1.55, "low": 1.35},  # green
        {"open": 1.6, "close": 1.1, "high": 1.65, "low": 1.05},  # bearish engulf
    ])

    out = generate_pattern_signals(df)
    assert out.loc[1, "pattern_signal"] == "bullish_engulfing"
    assert out.loc[1, "entry_signal"] == "buy"
    assert out.loc[3, "pattern_signal"] == "bearish_engulfing"
    assert out.loc[3, "entry_signal"] == "sell"


def test_simulate_partial_tp_safe_fallback_sl(monkeypatch):
    """ตรวจสอบ fallback SL เมื่อ tsl_activated แต่ sl เป็น None"""
    from nicegold_v5.exit import simulate_partial_tp_safe

    def fake_should_exit(trade, row):
        trade["sl"] = None
        trade["tsl_activated"] = True
        return True, "manual"

    monkeypatch.setattr("nicegold_v5.exit.should_exit", fake_should_exit)

    df = pd.DataFrame([
        {
            "timestamp": pd.Timestamp("2025-01-01 00:00"),
            "close": 100.0,
            "high": 100.3,
            "low": 99.7,
            "entry_signal": "buy",
            "atr": 1.0,
        },
        {
            "timestamp": pd.Timestamp("2025-01-01 00:01"),
            "close": 100.1,
            "high": 100.2,
            "low": 99.9,
            "entry_signal": "buy",
            "atr": 1.0,
        },
    ])

    trades = simulate_partial_tp_safe(df)
    assert not trades.empty
    assert trades.loc[0, "sl_price"] == 100.0 + 0.5


def test_should_exit_trailing_debug(caplog):
    trade = {
        'entry': 100,
        'type': 'buy',
        'lot': 0.1,
        'entry_time': pd.Timestamp('2025-01-01 00:00:00'),
        'tsl_activated': True,
        'trailing_sl': 99.5,
    }
    row = {
        'close': 104.0,
        'high': 105.0,
        'low': 103.5,
        'gain_z': 0.1,
        'atr': 1.0,
        'atr_ma': 1.0,
        'timestamp': pd.Timestamp('2025-01-01 00:05:00'),
    }
    with caplog.at_level(logging.DEBUG):
        should_exit(trade, row)
    assert any('Updated trail' in rec.message for rec in caplog.records)

def test_generate_signals_print_blocked_pct(capsys):
    df = sample_df()
    generate_signals(df)
    out = capsys.readouterr().out
    assert 'Entry Signal Blocked' in out


def test_simulate_trades_with_tp_sl_exit():
    from nicegold_v5.entry import simulate_trades_with_tp

    df = pd.DataFrame([
        {
            'timestamp': pd.Timestamp('2025-01-01 00:00:00'),
            'close': 100.0,
            'high': 101.0,
            'low': 94.0,
            'signal': 'long',
            'session': 'London',
            'rsi': 55,
            'pattern': 'inside_bar',
        }
    ])

    trades, _ = simulate_trades_with_tp(df)
    assert trades[0]['exit_reason'] == 'sl'


def test_simulate_trades_with_tp_tp1_exit():
    from nicegold_v5.entry import simulate_trades_with_tp

    df = pd.DataFrame([
        {
            'timestamp': pd.Timestamp('2025-01-01 00:00:00'),
            'close': 100.0,
            'high': 108.0,
            'low': 96.0,
            'signal': 'long',
            'session': 'London',
            'rsi': 30,
            'pattern': 'inside_bar',
        }
    ])

    trades, _ = simulate_trades_with_tp(df)
    assert trades[0]['exit_reason'] == 'tp1'


def test_simulate_trades_with_tp_zero_planned_risk():
    from nicegold_v5.entry import simulate_trades_with_tp

    df = pd.DataFrame([
        {
            'timestamp': pd.Timestamp('2025-01-01 00:00:00'),
            'close': 100.0,
            'high': 100.0,
            'low': 100.0,
            'signal': 'long',
            'session': 'London',
            'rsi': 50,
            'pattern': 'inside_bar',
        }
    ])

    trades, _ = simulate_trades_with_tp(df, sl_distance=0.0)
    assert trades[0]['planned_risk'] == 0.01

