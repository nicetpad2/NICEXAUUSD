import os
import pandas as pd
import datetime
import json

from nicegold_v5.entry import validate_indicator_inputs, simulate_partial_tp_safe


def force_entry_stress_test(df: pd.DataFrame, config: dict, audit_prefix: str = "QA_FORCEENTRY", log_dir: str = "logs/") -> dict:
    """Run ForceEntry Stress Test by opening a trade on every bar.

    Parameters
    ----------
    df : pd.DataFrame
        Price data with columns 'timestamp', 'close', 'high', 'low'.
    config : dict
        Configuration parameters for SL/TP distance.
    audit_prefix : str
        Prefix for the output log filenames.
    log_dir : str
        Directory where audit CSV/JSON will be saved.
    """
    os.makedirs(log_dir, exist_ok=True)
    df = df.copy()
    validate_indicator_inputs(df, min_rows=min(10, len(df)))
    df["entry_signal"] = "force_buy"
    trades_df = simulate_partial_tp_safe(df)

    equity = trades_df["pnl"].cumsum()
    peak = equity.cummax()
    trades_df["drawdown"] = ((peak - equity) / peak.replace(0, pd.NA)).fillna(0)

    audit = {
        "run_type": "QA_FORCEENTRY",
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M"),
        "config": config,
        "metrics": {
            "total_trades": len(trades_df),
            "max_drawdown": float(trades_df["drawdown"].max()) if not trades_df.empty else 0.0,
            "total_pnl": float(trades_df["pnl"].sum()) if not trades_df.empty else 0.0,
            "sl_count": int((trades_df.get("exit_reason") == "sl").sum()),
        },
        "version": config.get("version", "dev"),
        "commit_hash": config.get("commit_hash", "manual"),
    }

    csv_path = os.path.join(log_dir, f"{audit_prefix}_{audit['timestamp']}.csv")
    json_path = os.path.join(log_dir, f"{audit_prefix}_{audit['timestamp']}.json")
    trades_df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"[QA] Exported audit files → {csv_path}, {json_path}")
    return audit
