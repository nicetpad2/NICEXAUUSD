# config.py – Fold-Based Entry Config

ENTRY_CONFIG_PER_FOLD = {
    1: {"gain_z_thresh": -0.05, "ema_slope_min": 0.0001},
    2: {"gain_z_thresh": -0.1, "ema_slope_min": 0.0},
    3: {"gain_z_thresh": -0.2, "ema_slope_min": -0.01},
    4: {"gain_z_thresh": -0.05, "ema_slope_min": 0.01},
    5: {"gain_z_thresh": -0.15, "ema_slope_min": 0.001},
}

# --- PATCH v26.0.1: Ensure BUY/SELL always enabled ---
def ensure_order_side_enabled(cfg: dict) -> dict:
    """Force disable_buy/disable_sell to False for safety."""
    if "disable_buy" in cfg:
        cfg["disable_buy"] = False
    if "disable_sell" in cfg:
        cfg["disable_sell"] = False
    return cfg

# [Patch v8.1.5] Default Sniper Config สำหรับใช้ใน main.py menu 4
SNIPER_CONFIG_DEFAULT = {
    "gain_z_thresh": -0.05,
    "ema_slope_min": 0.05,
    "atr_thresh": 0.5,
    "sniper_risk_score_min": 5.5,
    "tp_rr_ratio": 6.0,
    "volume_ratio": 0.8,
}

# [Patch v8.1.6] Relaxed sniper config สำหรับ fallback
SNIPER_CONFIG_RELAXED = {
    "gain_z_thresh": -0.3,
    "ema_slope_min": -0.01,
    "atr_thresh": 0.3,
    "sniper_risk_score_min": 4.5,
    "tp_rr_ratio": 5.0,
    "volume_ratio": 0.6,
}

# [Patch v8.1.7] Override sniper config สำหรับ fallback กรณีบล็อกสัญญาณ 100%
SNIPER_CONFIG_OVERRIDE = {
    "gain_z_thresh": -0.3,
    "ema_slope_min": -0.01,
    "atr_thresh": 0.2,
    "sniper_risk_score_min": 2.5,
    "tp_rr_ratio": 4.5,
    "volume_ratio": 0.4,
}

# [Patch v8.1.8] Ultra override config สำหรับ fallback แบบสุดขีด
SNIPER_CONFIG_ULTRA_OVERRIDE = {
    "gain_z_thresh": -9.9,
    "ema_slope_min": -9.9,
    "atr_thresh": 0.0,
    "sniper_risk_score_min": -1.0,
    "tp_rr_ratio": 3.5,
    "volume_ratio": 0.0,
}

# ค่าคงที่สำหรับโหมด Stable Gain
SNIPER_CONFIG_STABLE_GAIN = {
    "gain_z_thresh": 0.0,
    "ema_slope_min": 0.02,
    "atr_thresh": 0.15,
    "sniper_risk_score_min": 3.0,
    "tp_rr_ratio": 4.2,
    "tp1_rr_ratio": 1.5,
    "volume_ratio": 0.4,
}

# ค่าปรับอัตโนมัติเมื่อเน้นกำไรต่อไม้สูง
SNIPER_CONFIG_AUTO_GAIN = {
    "gain_z_thresh": 0.1,
    "ema_slope_min": 0.03,
    "atr_thresh": 0.2,
    "sniper_risk_score_min": 3.0,
    "tp_rr_ratio": 6.0,  # [Patch QA-P1] เพิ่มเป้าหมายกำไรเพื่อยกระดับ Avg Profit > 1 USD
    "tp1_rr_ratio": 0.0,
    "volume_ratio": 0.4,
}

# [Patch v9.0] Config ใหม่จากการ Re-Optimize หลัง Q3 เพื่อเพิ่มความทนทาน
SNIPER_CONFIG_Q3_TUNED = {
    "gain_z_thresh": -0.1,           # [Patch v9.0] ปรับเกณฑ์ Momentum ให้ยืดหยุ่นขึ้น
    "ema_slope_min": 0.04,           # [Patch v9.0] กรองสภาวะ Sideways เข้มขึ้น
    "atr_thresh": 0.3,               # [Patch v9.0] กรอง Spike และ Volatility ต่ำ
    "sniper_risk_score_min": 3.5,    # [Patch v9.0] เลือกเฉพาะเทรดคุณภาพสูงขึ้น
    "tp_rr_ratio": 5.5,              # [Patch v9.0] ลด TP ลงเล็กน้อย เพิ่ม Win Rate
    "tp1_rr_ratio": 1.5,             # [Patch v9.0] กำหนด TP1 ชัดเจน (ถ้าใช้)
    "volume_ratio": 0.3,             # [Patch v16.2.1] ลดเงื่อนไข volume guard
    "disable_buy": False,             # [Patch v16.0.2] ปิดฝั่ง Buy -> Force Enabled
    "min_volume": 0.05,              # [Patch v16.1.9] Volume filter
    "enable_be": True,               # [Patch v16.1.9] เปิด Breakeven
    "enable_trailing": True,         # [Patch v16.1.9] ใช้ Trailing SL
}

# [Patch v11.8] Relaxed fallback config หลัง Q3 ปรับลดเงื่อนไขให้ค้นหาสัญญาณได้กว้างขึ้น
RELAX_CONFIG_Q3 = {
    "gain_z_thresh": -0.3,
    "ema_slope_min": -0.02,
    "atr_thresh": 0.2,
    "sniper_risk_score_min": 2.0,
    "tp_rr_ratio": 4.5,
    "tp1_rr_ratio": 1.2,
    "volume_ratio": 0.4,
}

# [Patch QA-P11] Config สำหรับวินิจฉัย - ผ่อนปรนสูงสุดเพื่อแก้ปัญหา Signal Blocked 100%
SNIPER_CONFIG_DIAGNOSTIC = {
    "gain_z_thresh": -1.0,           # [Patch QA-P11] ยอมรับ Momentum เกือบทุกรูปแบบ
    "ema_slope_min": -0.1,           # [Patch QA-P11] ยอมรับ Trend เกือบทุกรูปแบบ
    "atr_thresh": 0.1,               # [Patch QA-P11] ยอมรับ Volatility เกือบทุกรูปแบบ
    "sniper_risk_score_min": 0.5,    # [Patch QA-P11] ลดเกณฑ์ Risk Score ลงต่ำสุด
    "tp_rr_ratio": 3.0,              # [Patch QA-P11] ลด TP เพื่อให้มีโอกาสชน TP ง่ายขึ้น
    "tp1_rr_ratio": 1.0,             # [Patch QA-P11] กำหนด TP1 ชัดเจน
    "volume_ratio": 0.1,             # [Patch QA-P11] ลดเกณฑ์ Volume ลงต่ำสุด
}

# [Patch v8.1.9] Relaxed AutoGain สำหรับ fallback ขั้นสุดท้าย
SNIPER_CONFIG_RELAXED_AUTOGAIN = {
    "gain_z_thresh": -0.5,
    "ema_slope_min": -0.05,
    "atr_thresh": 0.05,
    "sniper_risk_score_min": 0.0,
    "tp_rr_ratio": 3.0,
    "volume_ratio": 0.1,
}

SNIPER_CONFIG_UNBLOCK = {
    "gain_z_thresh": -0.2,
    "ema_slope_min": -0.01,
    "atr_thresh": 0.0,
    "sniper_risk_score_min": 2.0,
    "tp_rr_ratio": 4.0,
    "volume_ratio": 0.0,
}  # [Patch v9.1] โหมดปลดล็อกออเดอร์ พร้อมรับ market ทุกภาวะ

SNIPER_CONFIG_PROFIT = {
    "gain_z_thresh": -0.05,
    "ema_slope_min": 0.01,
    "atr_thresh": 0.15,
    "sniper_risk_score_min": 3.0,
    "tp_rr_ratio": 5.5,
    "volume_ratio": 0.5,
}  # [Patch v10.0] Profit Mode RR + filter slope logic
# ====== [Patch v26.0.0] Entry Filtering Config (Hedge Fund Mode) ======
HEDGEFUND_ENTRY_CONFIG = {
    "gain_z_thresh": -0.04,
    "ema_slope_min": 0.02,
    "atr_thresh": 0.5,
    "sniper_risk_score_min": 3.0,
    "tp_rr_ratio": 2.2,
    "tp1_rr_ratio": 1.15,
    "volume_ratio": 0.25,
    "disable_buy": False,
    "disable_sell": False,
    "min_volume": 0.02,
    "enable_be": True,
    "enable_trailing": True,
    "tp2_delay_min": 7,
    "atr_multiplier": 1.2,
    "use_dynamic_tsl": True,
    "dynamic_lot": True,
    "lot_win_multiplier": 1.18,
    "lot_loss_multiplier": 0.88,
    "session_adaptive": True,
}


# Meta ML Feature Config
META_CLASSIFIER_FEATURES = [
    'Gain_Z', 'MACD_hist', 'Candle_Speed', 'Candle_Ratio', 'Signal_Score', 'Wick_Ratio', 'Pattern_Label'
]
META_META_CLASSIFIER_FEATURES = [
    'meta_proba_tp', 'Signal_Score', 'Pattern_Label', 'tp_distance', 'sl_distance'
]

# ML Toggle Configs
USE_META_CLASSIFIER = True  # [Patch v9.0]
USE_META_META_CLASSIFIER = True  # [Patch v9.0]
FOLD_SPECIFIC_THRESHOLD_TUNING = True
ENABLE_AUTO_THRESHOLD_TUNING = True
TUNE_L2_THRESHOLD = True

# [Patch v16.2.2] AutoFix WFV Base Config เปิด BE และ Trailing SL
AUTOFIX_WFV_CONFIG = {
    "enable_be": True,
    "enable_trailing": True,
    "use_dynamic_tsl": True,
    "tp2_delay_min": 10,
    "breakeven_rr_trigger": 1.2,
    "trailing_rr_trigger": 0.9,
}
# [Patch v26.0.0] Adaptive Session Config
SESSION_CONFIG = {
    "Asia":   {"gain_z_thresh": -0.08, "tp1_rr_ratio": 1.10, "tp2_rr_ratio": 2.0,  "disable_buy": False, "disable_sell": False},
    "London": {"gain_z_thresh":  0.00, "tp1_rr_ratio": 1.15, "tp2_rr_ratio": 2.15, "disable_buy": False, "disable_sell": False},
    "NY":     {"gain_z_thresh":  0.03, "tp1_rr_ratio": 1.18, "tp2_rr_ratio": 2.30, "disable_buy": False, "disable_sell": False},
}

# [Patch v28.1.0] QA ForceEntry Config (for QA/backtest only)
QA_FORCE_ENTRY_CONFIG = {
    "force_entry": True,
    "force_entry_ratio": 0.02,
    "force_entry_side": "both",        # ["buy", "sell", "both"]
    "force_entry_session": "all",      # ["Asia", "London", "NY", "all"]
    "force_entry_type": "all",         # reserved for future use
    "force_entry_min_orders": 30,
    "force_entry_seed": 42,
}

# [Patch HEDGEFUND-NEXT] Compound/OMS parameters
COMPOUND_MILESTONES = [200, 500, 1000, 2000, 5000]
KILL_SWITCH_DD = 35
RECOVERY_SL_TRIGGER = 3
RECOVERY_LOT_MULT = 1.5

# --- PATCH v26.0.1: Apply safety check to all configs ---
for _cfg in [
    SNIPER_CONFIG_DEFAULT,
    SNIPER_CONFIG_RELAXED,
    SNIPER_CONFIG_OVERRIDE,
    SNIPER_CONFIG_ULTRA_OVERRIDE,
    SNIPER_CONFIG_STABLE_GAIN,
    SNIPER_CONFIG_AUTO_GAIN,
    SNIPER_CONFIG_Q3_TUNED,
    RELAX_CONFIG_Q3,
    SNIPER_CONFIG_DIAGNOSTIC,
    SNIPER_CONFIG_RELAXED_AUTOGAIN,
    SNIPER_CONFIG_UNBLOCK,
    SNIPER_CONFIG_PROFIT,
    HEDGEFUND_ENTRY_CONFIG,
    QA_FORCE_ENTRY_CONFIG,
    AUTOFIX_WFV_CONFIG,
]:
    ensure_order_side_enabled(_cfg)

for sess_cfg in SESSION_CONFIG.values():
    ensure_order_side_enabled(sess_cfg)
