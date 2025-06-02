"""Adaptive Fix Engine utilities."""

import pandas as pd
import numpy as np
import logging
import copy

logger = logging.getLogger(__name__)


def _exit_variety_insufficient(trades_df: pd.DataFrame,
                               require=("tp1", "tp2", "sl")) -> bool:
    """Check if exit_reason variety is insufficient."""
    reasons = trades_df.get("exit_reason", pd.Series(dtype=str)).astype(str).str.lower()
    found = set(reasons)
    if "tp" in found:
        found.update({"tp1", "tp2"})
    return not set(require).issubset(found)

# [Patch v12.3.7+] – รวม Unified Patch + AutoFix WFV + AutoRiskAdjust
# ----------------------------------------------------------------------
# ✅ รวม Patch v12.3.5–v12.3.7
# ✅ ฝัง AutoFix Logic เข้า WFV (ต่อ Fold)
# ✅ เพิ่ม AutoRiskAdjust: ปรับ TP1 RR / SL ตามผล Fold ก่อนหน้า


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
        "net_pnl": trades_df["pnl"].sum() if "pnl" in trades_df.columns else 0.0,
        "avg_pnl": trades_df["pnl"].mean() if "pnl" in trades_df.columns else 0.0,
    }
    summary["tp_rate"] = (summary["tp1_count"] + summary["tp2_count"]) / (summary["total_trades"] + 1e-9)
    summary["sl_rate"] = summary["sl_count"] / (summary["total_trades"] + 1e-9)
    summary["exit_variety_insufficient"] = _exit_variety_insufficient(trades_df)

    print("\n📊 [Self-Diagnostic Report] Summary:")
    for k, v in summary.items():
        print(f"   ▸ {k}: {v:.4f}" if isinstance(v, float) else f"   ▸ {k}: {v}")

    return summary


def auto_fix_logic(summary: dict, config: dict, session: str = None) -> dict:
    """แก้ไข config อัตโนมัติ หากเจอ TP = 0 หรือ SL เยอะ"""
    new_config = config.copy()

    # [Patch v32.1.0] เรียก override BUY/SELL safety ทุก config
    from .config import ensure_order_side_enabled
    new_config = ensure_order_side_enabled(new_config)

    if summary.get("exit_variety_insufficient", False):
        new_config["tp1_rr_ratio"] = 1.2
        logger.info("[auto_fix_logic] Adjusted tp1_rr_ratio → 1.2")

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

    if summary["avg_mfe"] > 2.5 and summary["tp2_count"] == 0:
        print("\n[Patch Fix] 🧪 พบ MFE สูงแต่ไม่เกิด TP2 → ลด delay หรือเพิ่ม TP margin")
        new_config["tp2_delay_min"] = 5
        new_config["tp2_rr_ratio"] = 3.2

    if summary["avg_duration"] < 2.0 and summary["sl_rate"] > 0.2:
        print("\n[Patch Fix] ⛑ SL เกิดเร็ว → เพิ่ม minimum hold time ก่อน exit")
        new_config["min_hold_minutes"] = 10

    if summary.get("net_pnl", 0) <= 0:
        print("\n[Patch Fix] 📉 Net PnL ติดลบ → ลด RR1 และเปิด Dynamic TSL")
        new_config["tp1_rr_ratio"] = min(new_config.get("tp1_rr_ratio", 1.2), 1.0)
        new_config["atr_multiplier"] = max(new_config.get("atr_multiplier", 1.6), 1.8)
        new_config["use_dynamic_tsl"] = True

    if summary["sl_rate"] > 0.5 and summary["avg_mfe"] < 1.0:
        print("\n[Patch Fix] 🛡️ SL สูงและ MFE ต่ำ → เพิ่มเวลาถือและขยาย SL")
        new_config["min_hold_minutes"] = max(new_config.get("min_hold_minutes", 10), 15)
        new_config["atr_multiplier"] = max(new_config.get("atr_multiplier", 1.8), 2.0)

    return new_config


def simulate_and_autofix(df: pd.DataFrame, simulate_fn, config: dict, session: str = None):
    """simulate แล้วรัน Self-Diagnostic + AutoFix พร้อมคืน config ที่ปรับแล้ว"""
    result = simulate_fn(df)
    if isinstance(result, tuple):
        trades_df, equity_df = result
    else:
        trades_df = result
        equity_df = pd.DataFrame()
    summary = run_self_diagnostic(trades_df, df)
    config_adapted = auto_fix_logic(summary, config, session=session)
    return trades_df, equity_df, config_adapted


# [Patch v32.1.0+] AutoFix per Fold (ผนวกใน WFV)
def autofix_fold_run(df_fold: pd.DataFrame, simulate_fn, config: dict, fold_name: str):
    print(f"\n🔁 [Fold: {fold_name}] Running simulation with AutoFix...")
    trades_df, equity_df, config_used = simulate_and_autofix(df_fold, simulate_fn, config)
    print(f"✅ [Fold: {fold_name}] Completed. Adjusted config:")
    for k, v in config_used.items():
        print(f"   ▸ {k}: {v}")
    return trades_df, config_used


# [Patch v32.1.0] AutoRiskAdjust (เรียกต่อเนื่องใน WFV)
def autorisk_adjust(prev_config: dict, prev_summary: dict) -> dict:
    config = copy.deepcopy(prev_config)
    if prev_summary.get("tp_rate", 0) < 0.2:
        config["tp1_rr_ratio"] = 1.2
        print("[AutoRiskAdjust] ลด RR1 → 1.2")
    return config
