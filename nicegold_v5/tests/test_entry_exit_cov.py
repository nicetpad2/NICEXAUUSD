import pandas as pd
from nicegold_v5.exit import _rget, should_exit
from nicegold_v5.entry import generate_signals_v6_5, simulate_partial_tp_safe


def test_rget_default():
    assert _rget(object(), "foo", 42) == 42


def test_should_exit_be_sl():
    trade = {"entry": 100, "type": "buy", "entry_time": pd.Timestamp("2025-01-01"), "breakeven": True}
    row = {"close": 99, "timestamp": pd.Timestamp("2025-01-01 00:01:00")}
    exit_now, reason = should_exit(trade, row)
    assert exit_now and reason == "be_sl"


def test_generate_signals_v6_5_low_atr():
    ts = pd.date_range("2025-01-01", periods=5, freq="min")
    df = pd.DataFrame({
        "timestamp": ts,
        "open": 1.0,
        "high": 1.1,
        "low": 1.0,
        "close": 1.05,
        "volume": 100,
    })
    out = generate_signals_v6_5(df, fold_id=3)
    assert "entry_signal" in out.columns


def test_simulate_partial_tp_safe_branches():
    start = pd.Timestamp("2025-01-01 00:00:00")
    ts = [start, start + pd.Timedelta(minutes=1), start + pd.Timedelta(minutes=2), start + pd.Timedelta(minutes=3), start + pd.Timedelta(minutes=4)]
    df = pd.DataFrame({
        "timestamp": ts,
        "close": [100, 100, 100, 100, 100],
        "high": [100, 103, 100.1, 100, 100.1],
        "low": [100, 99, 98, 97, 99.9],
        "atr": [1, 1, 1, 1, 1],
        "ema_slope": [0, 0.5, 0.5, -0.5, 0.5],
        "entry_signal": [None, "buy", "buy", "sell", "buy"],
        "session": ["Asia"] * 5,
    })
    trades = simulate_partial_tp_safe(df)
    assert trades.exit_reason.tolist() == ["tp2", "sl", "tp2", "timeout_exit"]
