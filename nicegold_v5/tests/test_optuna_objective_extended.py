import pandas as pd
import optuna
import nicegold_v5.optuna_tuner as tuner


def test_objective_empty_df():
    tuner.session_folds = {"London": pd.DataFrame()}
    trial = optuna.trial.FixedTrial({
        "gain_z_thresh": 0.1,
        "ema_slope_min": 0.1,
        "atr_thresh": 0.5,
        "sniper_risk_score_min": 5.5,
        "tp_rr_ratio": 4.5,
        "volume_ratio": 1.0,
    })
    assert tuner.objective(trial) == -999


def test_objective_score(monkeypatch):
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=3, freq="h"),
        "close": [1.0, 2.0, 3.0],
        "high": [1.0, 2.0, 3.0],
        "low": [0.5, 1.5, 2.5],
        "volume": [1.0, 1.0, 1.0],
    })
    tuner.session_folds = {"London": df}
    monkeypatch.setattr(tuner, "generate_signals", lambda d, config=None, **kw: d.assign(entry_signal="buy"))
    monkeypatch.setattr(
        tuner,
        "run_backtest",
        lambda d: (pd.DataFrame({"pnl": [1.0]}), pd.DataFrame({"equity": [100, 101]})),
    )
    monkeypatch.setattr(
        tuner, "print_qa_summary", lambda t, e: {"total_profit": 2, "max_drawdown": 0, "avg_pnl": 2}
    )
    trial = optuna.trial.FixedTrial({
        "gain_z_thresh": 0.1,
        "ema_slope_min": 0.1,
        "atr_thresh": 0.5,
        "sniper_risk_score_min": 5.5,
        "tp_rr_ratio": 4.5,
        "volume_ratio": 1.0,
    })
    score = tuner.objective(trial)
    assert score > 0
