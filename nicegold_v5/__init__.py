from .entry import (
    generate_signals_v8_0,
    generate_signals_v9_0,
    generate_signals_unblock_v9_1,
    generate_signals_profit_v10,
    generate_signals_v11_scalper_m1,
    generate_signals_v12_0,
    generate_signals,
    generate_pattern_signals,
    apply_tp_logic,
    generate_entry_signal,
    session_filter,
    trade_log_fields,
    rsi,
    simulate_trades_with_tp,
    simulate_partial_tp_safe,
)  # [Patch v10.0] expose latest logic
from .backtester import calc_lot
from .exit import should_exit
from .backtester import run_backtest
from .utils import (
    summarize_results,
    run_auto_wfv,
    auto_entry_config,
    print_qa_summary,
    export_chatgpt_ready_logs,
    create_summary_dict,
    safe_calculate_net_change,
    convert_thai_datetime,
    prepare_csv_auto,
    simulate_tp_exit,
    
)
from .wfv import (
    run_walkforward_backtest,
    session_performance,
    streak_summary,
    build_trade_log,
    inject_exit_variety,
)
# [Patch QA-FIX v28.2.5] ensure_buy_sell must be accessible (for QA)
def ensure_buy_sell(*args, **kwargs):
    from nicegold_v5.wfv import ensure_buy_sell as orig
    return orig(*args, **kwargs)

def inject_exit_variety(*args, **kwargs):
    from nicegold_v5.wfv import inject_exit_variety as orig
    return orig(*args, **kwargs)
from .config import ENTRY_CONFIG_PER_FOLD
from .optuna_tuner import start_optimization, objective
from .qa import (
    run_qa_guard,
    summarize_fold,
    compute_fold_bias,
    analyze_drawdown,
    export_fold_qa,
    detect_fold_drift,
    auto_qa_after_backtest,
    force_entry_stress_test,
)
from .rl_agent import RLScalper
from .meta_classifier import MetaClassifier
from .ml_dataset_m1 import generate_ml_dataset_m1
from .adaptive_threshold_dl import (
    ThresholdDataset,
    ThresholdPredictor,
    load_wfv_training_data,
    train_threshold_predictor,
    predict_thresholds,
)

try:
    from .deep_model_m1 import LSTMClassifier
    from .train_lstm_runner import load_dataset, train_lstm
except Exception:  # pragma: no cover - torch may be missing
    LSTMClassifier = None
    load_dataset = None
    train_lstm = None
