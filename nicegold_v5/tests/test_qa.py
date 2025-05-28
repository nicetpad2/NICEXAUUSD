import os
import pandas as pd
from nicegold_v5.qa import (
    detect_overfit_bias,
    detect_noise_exit,
    detect_leakage_columns,
    run_qa_guard,
    summarize_fold,
    compute_fold_bias,
    analyze_drawdown,
    export_fold_qa,
    detect_fold_drift,
    auto_qa_after_backtest,
    QA_BASE_PATH,
)
import nicegold_v5.qa as patch_g5_auto_qa


def sample_trades():
    return pd.DataFrame({
        'pnl': [1.0, -0.5, 0.2],
        'mfe': [1.0, 0.5, 5.0],
        'duration_min': [1, 3, 10],
        'exit_reason': ['sl', 'sl', 'tp'],
        '_id': [1, 2, 3],
        'entry_time': pd.date_range('2025-01-01', periods=3, freq='min'),
    })


def test_detect_overfit_bias():
    metrics = detect_overfit_bias(sample_trades())
    assert 'overfit_score' in metrics


def test_detect_noise_exit():
    noisy = detect_noise_exit(sample_trades())
    assert (noisy['pnl'] < 0).all()


def test_detect_leakage_columns():
    df = pd.DataFrame({'future_price': [1], 'next_val': [2], 'value_lead': [3], 'target': [0], 'keep': [1]})
    cols = detect_leakage_columns(df)
    assert set(cols) == {'future_price', 'next_val', 'value_lead', 'target'}


def test_run_qa_guard(capsys):
    trades = sample_trades()
    df = pd.DataFrame({'a': [1, 2, 3]})
    run_qa_guard(trades, df)
    out = capsys.readouterr().out
    assert 'QA Guard' in out


def test_summarize_fold():
    summary = summarize_fold(sample_trades(), 'X')
    assert summary['fold'] == 'X'
    assert 'winrate' in summary


def test_compute_fold_bias():
    bias = compute_fold_bias(sample_trades())
    assert isinstance(bias, float)


def test_analyze_drawdown():
    eq = pd.DataFrame({'equity': [100, 95, 110]})
    dd = analyze_drawdown(eq)
    assert 'max_drawdown_pct' in dd


def test_export_fold_qa(tmp_path):
    stats = summarize_fold(sample_trades(), 'X')
    bias = compute_fold_bias(sample_trades())
    dd = {'max_drawdown_pct': 5}
    export_fold_qa('X', stats, bias, dd, outdir=str(tmp_path))
    assert (tmp_path / 'fold_qa_x.json').exists()


def test_detect_fold_drift():
    stats1 = summarize_fold(sample_trades(), 'A')
    stats2 = summarize_fold(sample_trades(), 'B')
    df = detect_fold_drift([stats1, stats2])
    assert not df.empty


def test_auto_qa_after_backtest(tmp_path, monkeypatch):
    trades = sample_trades()
    equity = pd.DataFrame({'equity': [100, 101, 99]})
    qa_path = tmp_path / 'qa'
    monkeypatch.setattr(patch_g5_auto_qa, 'QA_BASE_PATH', str(qa_path))
    os.makedirs(qa_path, exist_ok=True)
    auto_qa_after_backtest(trades, equity, label='X')
    csv_files = list(qa_path.glob('fold_qa_x_*.csv'))
    json_files = list(qa_path.glob('fold_qa_x_*.json'))
    assert len(csv_files) == 1
    assert len(json_files) == 1


def test_default_qa_base_path():
    from nicegold_v5.qa import QA_BASE_PATH
    assert os.path.isabs(QA_BASE_PATH)
    assert QA_BASE_PATH == "/content/drive/MyDrive/NICEGOLD/logs/qa"
