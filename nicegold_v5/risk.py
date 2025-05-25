def calc_lot(capital: float, risk_pct: float = 1.5, sl_pips: float = 100) -> float:
    risk_amount = capital * (risk_pct / 100)
    pip_value = 10  # For gold 0.1/point × 100 = $10/lot
    lot = risk_amount / (sl_pips * pip_value)
    return max(0.01, round(lot, 2))
