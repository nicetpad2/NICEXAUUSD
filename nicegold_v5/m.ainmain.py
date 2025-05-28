from datetime import datetime
import numpy as np

# Patch C – Strategy Improvement Package
# --------------------------------------
# รวม Kill Switch, Recovery, Adaptive TP, Dynamic SL/TP และ Dynamic Lot

KILL_SWITCH_DD = 25  # %


def kill_switch(equity_curve):
    peak = equity_curve[0]
    for eq in equity_curve:
        dd = (peak - eq) / peak * 100
        if dd >= KILL_SWITCH_DD:
            print("[KILL SWITCH] Drawdown limit reached. Backtest halted.")
            return True
        peak = max(peak, eq)
    return False


def apply_recovery_lot(capital, sl_streak, base_lot=0.01):
    if sl_streak >= 2:
        factor = 1 + 0.5 * (sl_streak - 1)
        return round(base_lot * factor, 2)
    return base_lot


def adaptive_tp_multiplier(session):
    if session == "Asia":
        return 1.5
    elif session == "London":
        return 2.0
    elif session == "NY":
        return 2.5
    return 2.0


def get_sl_tp(price, atr, session, direction):
    multiplier = adaptive_tp_multiplier(session)
    sl = price - atr * 1.2 if direction == "buy" else price + atr * 1.2
    tp = price + atr * multiplier if direction == "buy" else price - atr * multiplier
    return sl, tp


def calc_lot_risk(capital, atr, risk_pct=1.5):
    pip_value = 10
    sl_pips = atr * 10
    risk_amount = capital * (risk_pct / 100)
    lot = risk_amount / (sl_pips * pip_value)
    return max(0.01, round(lot, 2))

