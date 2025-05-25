import pandas as pd
from nicegold_v5.entry import generate_signals
from nicegold_v5.risk import calc_lot
from nicegold_v5.exit import should_exit
from nicegold_v5.backtester import run_backtest
from nicegold_v5.utils import summarize_results, run_auto_wfv
from nicegold_v5 import wfv


def sample_df():
    data = {
        'timestamp': pd.date_range('2024-01-01', periods=30, freq='D'),
        'open': pd.Series(range(30)) + 100,
        'high': pd.Series(range(30)) + 101,
        'low': pd.Series(range(30)) + 99,
        'close': pd.Series(range(30)) + 100,
        'gain_z': [0.0]*30,
        'atr_ma': [1.0]*30
    }
    return pd.DataFrame(data)


def sample_wfv_df():
    ts = pd.date_range('2024-01-01', periods=40, freq='H')
    data = {
        'Open': pd.Series(range(40)) + 100,
        'feat1': pd.Series(range(40)) * 0.1,
        'feat2': pd.Series(range(40)) * -0.1,
        'label': [0, 1] * 20,
        'EMA_50_slope': [1.0] * 40,
        'ATR_14': [1.0] * 40,
        'ATR_14_MA50': [0.5] * 40,
    }
    df = pd.DataFrame(data, index=ts)
    return df


def test_generate_signals():
    df = sample_df()
    out = generate_signals(df)
    assert 'entry_signal' in out.columns


def test_calc_lot():
    lot = calc_lot(100)
    assert lot >= 0.01


def test_should_exit():
    trade = {'entry': 100, 'type': 'buy', 'lot': 0.1}
    row = {'close': 101, 'gain_z': -1, 'atr': 1.0, 'atr_ma': 1.5}
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


def test_session_performance():
    df = sample_wfv_df()
    trades = wfv.run_walkforward_backtest(df, ['feat1', 'feat2'], 'label', n_folds=2)
    perf = wfv.session_performance(trades)
    assert 'sum' in perf.columns
