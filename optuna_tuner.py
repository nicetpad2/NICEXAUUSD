import pandas as pd
import optuna
from nicegold_v5.entry import generate_signals
from nicegold_v5.backtester import run_backtest
from nicegold_v5.utils import print_qa_summary, split_by_session

session_folds: dict[str, pd.DataFrame] = {}


def objective(trial) -> float:
    config = {
        "gain_z_thresh": trial.suggest_float("gain_z_thresh", -0.1, 0.6),
        "ema_slope_min": trial.suggest_float("ema_slope_min", 0.0, 0.4),
        "atr_thresh": trial.suggest_float("atr_thresh", 0.5, 1.2),
        "sniper_risk_score_min": trial.suggest_float("sniper_risk_score_min", 5.0, 7.5),
        "tp_rr_ratio": trial.suggest_float("tp_rr_ratio", 4.0, 8.0),
        "volume_ratio": trial.suggest_float("volume_ratio", 0.7, 1.2),
    }

    df = session_folds.get("London", pd.DataFrame()).copy()
    if df.empty:
        return -999
    df = generate_signals(df, config=config)
    trades, equity = run_backtest(df)

    if trades.empty:
        return -999

    metrics = print_qa_summary(trades, equity)
    profit = metrics.get("total_profit", 0)
    drawdown = metrics.get("max_drawdown", 0)
    avg_pnl = metrics.get("avg_pnl", 0)

    score = profit - drawdown * 1.5 + avg_pnl * 5
    return score


def start_optimization(df: pd.DataFrame, n_trials: int = 50):
    global session_folds
    session_folds = split_by_session(df)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    print("✅ Best trial:")
    print(study.best_trial)
    return study
