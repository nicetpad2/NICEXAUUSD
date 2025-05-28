import os
import json
import pandas as pd


def export_fold_qa(fold_name: str, stats: dict, bias_score: float, drawdown: dict, outdir: str = "logs"):
    """Export fold QA summary and related metrics as JSON."""
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"fold_qa_{fold_name.lower()}.json")
    payload = {
        "fold": fold_name,
        "qa_summary": stats,
        "overfit_bias_score": bias_score,
        "drawdown": drawdown,
    }
    with open(outpath, "w") as f:
        json.dump(payload, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else o)
    print(f"📁 Exported QA summary → {outpath}")


def detect_fold_drift(fold_stats: list[dict]) -> pd.DataFrame:
    """Detect metric variations across folds."""
    df = pd.DataFrame(fold_stats)
    if len(df) < 2:
        return pd.DataFrame()
    metrics = ["winrate", "avg_pnl", "sl_count"]
    drifts = []
    for m in metrics:
        std = df[m].std()
        mean = df[m].mean()
        pct_std = round((std / abs(mean)) * 100, 2) if mean != 0 else 0.0
        drifts.append({"metric": m, "std_dev": round(std, 2), "%_variation": pct_std})
    return pd.DataFrame(drifts)
