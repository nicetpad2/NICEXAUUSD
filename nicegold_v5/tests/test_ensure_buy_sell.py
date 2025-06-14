import pandas as pd
from nicegold_v5.utils import ensure_buy_sell


def test_ensure_buy_sell_skip():
    df = pd.DataFrame({'Open': [1, 2]})
    trades = pd.DataFrame({'side': ['buy', 'sell']})
    result = ensure_buy_sell(trades, df, lambda d, percentile_threshold=75: trades, outdir=None)
    assert len(result) == 2


def test_ensure_buy_sell_force(monkeypatch):
    df = pd.DataFrame({'Open': [1, 2]})
    calls = {'cnt': 0}

    def fake_sim(data, percentile_threshold=75):
        calls['cnt'] += 1
        if percentile_threshold == 75:
            return pd.DataFrame({'side': ['buy']})
        return pd.DataFrame({'side': ['buy', 'sell']})

    trades = pd.DataFrame({'side': ['buy']})
    result = ensure_buy_sell(trades, df, fake_sim, outdir=None)
    assert {'buy', 'sell'} <= set(result['side'])
    assert calls['cnt'] >= 1


def test_ensure_buy_sell_no_percentile():
    df = pd.DataFrame({'Open': [1, 2]})
    calls = {'cnt': 0}

    def fake_sim(data):
        calls['cnt'] += 1
        return pd.DataFrame({'side': ['buy', 'sell']})

    trades = pd.DataFrame({'side': ['buy']})
    result = ensure_buy_sell(trades, df, fake_sim, outdir=None)
    assert {'buy', 'sell'} <= set(result['side'])
    assert calls['cnt'] >= 1


def test_ensure_buy_sell_dummy_pnl():
    df = pd.DataFrame({
        'timestamp': [pd.Timestamp('2025-01-01 00:00:00')],
        'Close': [2000.0],
        'ATR_14': [1.0],
    })

    def fake_sim(data, percentile_threshold=75):
        return pd.DataFrame({'side': ['buy']})

    trades = pd.DataFrame({'side': ['buy']})
    out = ensure_buy_sell(trades, df, fake_sim, outdir=None, fold=1)
    dummy = out[out['is_dummy'].fillna(False)]
    assert not dummy.empty
    assert dummy['pnl'].notna().all()
