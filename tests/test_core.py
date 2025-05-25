import pandas as pd
from nicegold_v5.entry import generate_signals, rsi
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
    ts = pd.date_range('2024-01-01', periods=40, freq='h')
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

# New helper with lowercase 'open'
def sample_wfv_df_lower():
    df = sample_wfv_df().rename(columns={'Open': 'open'})
    return df


def test_rsi_vectorized():
    series = pd.Series(range(1, 16))
    result = rsi(series, period=14)
    assert round(result.iloc[-1], 2) == 100.0


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
        'commission': [0.02, 0.02]
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
