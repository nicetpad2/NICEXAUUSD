import pandas as pd
from nicegold_v5.backtester import run_backtest


def test_backtest_hit_sl_expected():
    """ทดสอบว่าไม้โดน SL และออกด้วยเหตุผล SL"""
    df = pd.DataFrame({
        "timestamp": pd.date_range(start="2025-01-01", periods=5, freq="min"),
        "open": [100, 100.5, 100.8, 101.0, 101.2],
        "high": [100, 100.6, 100.9, 101.1, 101.3],
        "low": [99.5, 99.8, 100.0, 100.2, 100.4],
        "close": [100.0, 99.0, 98.9, 99.1, 99.0],
        "entry_signal": ["buy", None, None, None, None],
        "atr": [0.5] * 5,
        "atr_ma": [0.6] * 5,
        "gain_z": [0.0] * 5,
    })

    trades, equity = run_backtest(df)
    assert not trades.empty
    assert any(trades["exit_reason"] == "SL")
    assert (trades["pnl"] < 0).any()
