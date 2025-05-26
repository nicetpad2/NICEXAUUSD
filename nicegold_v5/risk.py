def calc_lot(capital: float, risk_pct: float = 1.5, sl_pips: float = 100) -> float:
    risk_amount = capital * (risk_pct / 100)
    pip_value = 10  # For gold 0.1/point × 100 = $10/lot
    lot = risk_amount / (sl_pips * pip_value)
    return max(0.01, round(lot, 2))


# --- Patch C: Advanced Risk Management ---

KILL_SWITCH_DD = 25  # % drawdown limit
MIN_TRADES_BEFORE_KILL = 100  # ต้องมีเทรดมากกว่า 100 ไม้ก่อนจึงเริ่มตรวจ DD


def kill_switch(equity_curve: list[float]) -> bool:
    """Return True if drawdown exceeds threshold (หลังจากเทรดครบ MIN_TRADES_BEFORE_KILL)"""
    if not equity_curve or len(equity_curve) < MIN_TRADES_BEFORE_KILL:
        return False  # ยังไม่ตรวจ drawdown
    peak = equity_curve[0]
    for eq in equity_curve:
        drawdown = (peak - eq) / peak * 100
        if drawdown >= KILL_SWITCH_DD:
            print("[KILL SWITCH] Drawdown limit reached. Backtest halted.")
            return True
        peak = max(peak, eq)
    return False


def apply_recovery_lot(capital: float, sl_streak: int, base_lot: float = 0.01) -> float:
    """Increase lot size after consecutive stop-losses."""
    if sl_streak >= 2:
        factor = 1 + 0.5 * (sl_streak - 1)
        return round(base_lot * factor, 2)
    return base_lot


def adaptive_tp_multiplier(session: str) -> float:
    if session == "Asia":
        return 1.5
    if session == "London":
        return 2.0
    if session == "NY":
        return 2.5
    return 2.0


def get_sl_tp(price: float, atr: float, session: str, direction: str) -> tuple[float, float]:
    multiplier = adaptive_tp_multiplier(session)
    sl = price - atr * 1.2 if direction == "buy" else price + atr * 1.2
    tp = price + atr * multiplier if direction == "buy" else price - atr * multiplier
    return sl, tp


def calc_lot_risk(capital: float, atr: float, risk_pct: float = 1.5) -> float:
    """True percent risk model using ATR as stop distance."""
    pip_value = 10  # XAUUSD standard
    sl_pips = atr * 10
    risk_amount = capital * (risk_pct / 100)
    lot = risk_amount / (sl_pips * pip_value)
    return max(0.01, round(lot, 2))


# --- Patch B.2: Recovery Mode Risk Logic ---
RECOVERY_SL_TRIGGER = 3  # SL สะสมกี่ครั้งจึงเข้าโหมด recovery


def calc_lot_recovery(capital: float, atr: float, risk_pct: float = 1.5) -> float:
    """Adaptive lot size เมื่ออยู่ใน Recovery Mode (lot × 1.5)."""
    base_lot = calc_lot_risk(capital, atr, risk_pct)
    recovery_lot = base_lot * 1.5
    return max(0.01, round(recovery_lot, 2))


def get_sl_tp_recovery(price: float, atr: float, direction: str) -> tuple[float, float]:
    """คำนวณ SL/TP แบบกว้างขึ้นสำหรับ Recovery Mode."""
    sl_multiplier = 1.4
    tp_multiplier = 1.8
    if direction == "buy":
        sl = price - atr * sl_multiplier
        tp = price + atr * tp_multiplier
    else:
        sl = price + atr * sl_multiplier
        tp = price - atr * tp_multiplier
    return sl, tp

