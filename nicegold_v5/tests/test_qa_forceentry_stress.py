import pandas as pd
import importlib
import qa


def sample_df():
    return pd.DataFrame({
        "timestamp": pd.date_range('2025-01-01', periods=3, freq='min'),
        "close": [1.0, 1.1, 1.2],
        "high": [1.1, 1.2, 1.3],
        "low": [0.9, 1.0, 1.1],
        "volume": [100, 100, 100],
    })


def test_force_entry_stress(tmp_path):
    df = sample_df()
    cfg = {"atr_thresh": 0.1, "tp_rr_ratio": 2.0, "version": "test", "commit_hash": "abc"}
    audit = qa.force_entry_stress_test(df, cfg, log_dir=str(tmp_path) + "/")
    csv_files = list(tmp_path.glob("QA_FORCEENTRY_*.csv"))
    json_files = list(tmp_path.glob("QA_FORCEENTRY_*.json"))
    assert audit["metrics"]["total_trades"] == 3
    assert len(csv_files) == 1
    assert len(json_files) == 1
