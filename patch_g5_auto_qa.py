import os
import pandas as pd
from datetime import datetime
from patch_phase3_qa_guard import summarize_fold, compute_fold_bias, analyze_drawdown
from patch_g4_fold_export_drift import export_fold_qa, detect_fold_drift


def auto_qa_after_backtest(trades: pd.DataFrame, equity: pd.DataFrame, label: str = "Run"):
    """Automatically export QA summary after backtest."""
    if trades.empty:
        print("❌ ไม่มีข้อมูล trades สำหรับ QA")
        return

    stats = summarize_fold(trades, label)
    bias = compute_fold_bias(trades)
    dd = analyze_drawdown(equity)
    export_fold_qa(label, stats, bias, dd, outdir="logs/qa")

    save_csv_path = os.path.join("logs/qa", f"fold_qa_{label.lower()}.csv")
    pd.DataFrame([stats | {"bias_score": bias} | dd]).to_csv(save_csv_path, index=False)
    print(f"✅ QA Auto Export → {save_csv_path}")
