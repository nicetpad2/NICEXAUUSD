import pandas as pd
import numpy as np


def run_self_diagnostic(trades_df: pd.DataFrame, df: pd.DataFrame) -> dict:
    """ตรวจสอบคุณภาพการ simulate และวิเคราะห์สาเหตุเบื้องต้น"""
    exit_col = trades_df.get("exit_reason")
    summary = {
        "tp1_count": int((exit_col == "tp1").sum()) if exit_col is not None else 0,
        "tp2_count": int((exit_col == "tp2").sum()) if exit_col is not None else 0,
        "sl_count": int((exit_col == "sl").sum()) if exit_col is not None else 0,
        "be_count": int((exit_col == "be").sum()) if exit_col is not None else 0,
        "total_trades": len(trades_df),
        "avg_mfe": trades_df["mfe"].mean() if "mfe" in trades_df.columns else np.nan,
        "avg_duration": trades_df["duration_min"].mean() if "duration_min" in trades_df.columns else np.nan,
    }
    summary["tp_rate"] = (summary["tp1_count"] + summary["tp2_count"]) / (summary["total_trades"] + 1e-9)
    summary["sl_rate"] = summary["sl_count"] / (summary["total_trades"] + 1e-9)

    print("\n📊 [Self-Diagnostic Report] Summary:")
    for k, v in summary.items():
        print(f"   ▸ {k}: {v:.4f}" if isinstance(v, float) else f"   ▸ {k}: {v}")

    return summary


def auto_fix_logic(summary: dict, config: dict, session: str | None = None) -> dict:
    """แก้ไข config อัตโนมัติ หากเจอ TP = 0 หรือ SL เยอะ"""
    new_config = config.copy()

    if summary["tp1_count"] == 0 and summary["tp2_count"] == 0:
        print("\n[Patch Fix] ❗ TP1/TP2 = 0 → ลด TP1 RR จาก 1.5 → 1.2, TP2 delay เหลือ 10 นาที")
        new_config["tp1_rr_ratio"] = 1.2
        new_config["tp2_delay_min"] = 10
        new_config["atr_multiplier"] = 1.3

    elif summary["sl_rate"] > 0.3:
        print("\n[Patch Fix] ⚠️ SL > 30% → ขยับ SL ให้กว้างขึ้น")
        new_config["atr_multiplier"] = 1.6

    if session == "London" and summary["sl_rate"] > 0.25:
        print("\n[Patch Fix] 🔁 ปรับ SL multiplier สำหรับ session London")
        new_config["atr_multiplier"] = 1.7

    if summary.get("avg_mfe", 0) > 2.5 and summary["tp2_count"] == 0:
        print("\n[Patch Fix] 🧪 พบ MFE สูงแต่ไม่เกิด TP2 → ลด delay หรือเพิ่ม TP margin")
        new_config["tp2_delay_min"] = 5
        new_config["tp2_rr_ratio"] = 3.2

    if summary.get("avg_duration", 0) < 2.0 and summary["sl_rate"] > 0.2:
        print("\n[Patch Fix] ⛑ SL เกิดเร็ว → เพิ่ม minimum hold time ก่อน exit")
        new_config["min_hold_minutes"] = 10

    return new_config


def simulate_and_autofix(
    df: pd.DataFrame,
    simulate_fn,
    config: dict,
    session: str | None = None,
):
    """simulate แล้วรัน Self-Diagnostic + AutoFix พร้อมคืน config ที่ปรับแล้ว"""
    trades_df, equity_df = simulate_fn(df)
    summary = run_self_diagnostic(trades_df, df)
    config_adapted = auto_fix_logic(summary, config, session=session)
    return trades_df, equity_df, config_adapted
