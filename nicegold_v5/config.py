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
