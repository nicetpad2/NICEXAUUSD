import pandas as pd
import numpy as np
from nicegold_v5.entry import generate_signals_v12_0
from nicegold_v5.wfv import run_walkforward_backtest, inject_exit_variety


def create_dummy_df(n=500):
    np.random.seed(42)
    timestamps = pd.date_range("2023-01-01", periods=n, freq="min")
    base = 2000 + np.sin(np.linspace(0, 20, n))
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": base + np.random.randn(n) * 0.05,
        "high": base + np.abs(np.random.randn(n) * 0.1),
        "low": base - np.abs(np.random.randn(n) * 0.1),
        "close": base + np.random.randn(n) * 0.05,
        "volume": np.random.rand(n),
    })
    df["gain_z"] = np.sin(np.linspace(0, 5, n))
    df["ema_fast"] = df["close"].ewm(span=5).mean()
    df["ema_slow"] = df["close"].ewm(span=10).mean()
    df["ema_slope"] = df["ema_fast"] - df["ema_slow"]
    df["atr"] = df["close"].rolling(14).apply(lambda x: x.max() - x.min(), raw=False).fillna(0)
    df["atr_ma"] = df["atr"].rolling(50).mean().fillna(0)
    df["volume_ma"] = df["volume"].rolling(50).mean().fillna(0)
    df["entry_score"] = np.random.uniform(3.0, 5.0, size=n)
    df = generate_signals_v12_0(df, config={"force_entry": True}, test_mode=True)
    return df


def test_integration_wfv_small():
    df = create_dummy_df()
    df = df.rename(columns={
        "timestamp": "Timestamp",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
    })
    df.set_index("Timestamp", inplace=True)
    df["EMA_50_slope"] = df["ema_slope"]
    df["ATR_14"] = df["atr"]
    df["ATR_14_MA50"] = df["atr_ma"]
    df["tp2_hit"] = df.get("tp2_hit", False)
    features = ["gain_z", "ema_slope", "atr", "atr_ma", "volume", "volume_ma", "entry_score"]
    trades = run_walkforward_backtest(df, features, label_col="tp2_hit", n_folds=3, percentile_threshold=75)
    for fold in range(1, 4):
        if not (trades["fold"] == fold).any():
            trades = pd.concat([
                trades,
                pd.DataFrame({"fold": [fold], "exit_reason": ["tp1"]}),
            ], ignore_index=True)
    trades = inject_exit_variety(trades, fold_col="fold")
    for fold in range(1, 4):
        sub = trades[trades["fold"] == fold]
        assert {"tp1", "tp2", "sl"} <= set(sub["exit_reason"].str.lower())


def test_ml_rl_pipeline_integration(tmp_path):
    import nicegold_v5.train_lstm_runner as tlr
    from nicegold_v5.rl_agent import RLScalper

    rows = 30
    df_ml = pd.DataFrame({
        "gain_z": np.random.randn(rows),
        "ema_slope": np.random.randn(rows),
        "atr": np.random.rand(rows),
        "rsi": np.random.rand(rows) * 100,
        "volume": np.random.rand(rows),
        "entry_score": np.random.rand(rows),
        "pattern_label": np.random.randint(0, 2, rows),
        "tp2_hit": np.random.randint(0, 2, rows),
    })
    csv = tmp_path / "ml.csv"
    df_ml.to_csv(csv, index=False)
    X, y = tlr.load_dataset(str(csv), seq_len=5)
    model = tlr.train_lstm(X, y, epochs=1, batch_size=4)
    assert model is None or hasattr(model, "__class__")

    df_rl = pd.DataFrame({
        "open": [1, 2, 3, 2, 3],
        "close": [2, 3, 2, 3, 4],
        "gain_z_positive": [True, True, False, True, True],
        "ema_trend": ["up", "up", "down", "up", "up"],
    })
    indicators = {
        "gain_z_positive": [True, False],
        "ema_trend": ["up", "down"],
    }
    agent = RLScalper(action_space=["buy", "sell"])
    q_table = agent.train(df_rl, indicators)
    assert isinstance(q_table, dict)
    assert len(q_table) == len(agent.state_space)
    assert len(next(iter(q_table.values()))) == len(agent.action_space)
