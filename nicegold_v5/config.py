# config.py – Fold-Based Entry Config

ENTRY_CONFIG_PER_FOLD = {
    1: {"gain_z_thresh": -0.05, "ema_slope_min": 0.0001},
    2: {"gain_z_thresh": -0.1, "ema_slope_min": 0.0},
    3: {"gain_z_thresh": -0.2, "ema_slope_min": -0.01},
    4: {"gain_z_thresh": -0.05, "ema_slope_min": 0.01},
    5: {"gain_z_thresh": -0.15, "ema_slope_min": 0.001},
}

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
