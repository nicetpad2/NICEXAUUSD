import pandas as pd
import pytest
from nicegold_v5.backtester import (
    kill_switch,
    apply_recovery_lot,
    adaptive_tp_multiplier,
    get_sl_tp_recovery,
    calculate_mfe,
    calc_lot,
)


def test_kill_switch_no_dd_after_trades():
    curve = [100] * 100 + [99]
    assert not kill_switch(curve)


def test_apply_recovery_lot_no_streak():
    lot = apply_recovery_lot(1000, sl_streak=1, base_lot=0.02)
    assert lot == 0.02


@pytest.mark.parametrize(
    "session, expected",
    [
        ("Asia", 1.5),
        ("London", 2.0),
        ("NY", 2.5),
        ("Other", 2.0),
    ],
)
def test_adaptive_tp_multiplier(session, expected):
    assert adaptive_tp_multiplier(session) == expected


def test_get_sl_tp_recovery_sell():
    sl, tp = get_sl_tp_recovery(100, 1.0, "sell")
    assert round(sl, 2) == 101.8
    assert round(tp, 2) == 98.2


def test_calculate_mfe_empty():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=3, freq="min"),
        "high": [1, 2, 3],
        "low": [0, 1, 2],
    })
    start = pd.Timestamp("2025-01-02 00:00:00")
    end = pd.Timestamp("2025-01-02 00:05:00")
    assert calculate_mfe(start, end, df, 1.5, "buy") == 0.0


def test_calc_lot_account_guard():
    account = {"equity": 1000, "risk_pct": 0.01, "init_lot": 0.05}
    lot = calc_lot(account, sl_pips=0)
    assert lot >= 0.05
