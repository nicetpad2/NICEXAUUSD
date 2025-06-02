import pandas as pd
from nicegold_v5.rl_agent import RLScalper


def test_rl_training():
    df = pd.DataFrame(
        {
            "open": [1, 2, 3, 2, 3],
            "close": [2, 3, 2, 3, 4],
            "gain_z_positive": [True, True, False, True, True],
            "ema_trend": ["up", "up", "down", "up", "up"],
        }
    )
    indicators = {
        "gain_z_positive": [True, False],
        "ema_trend": ["up", "down"],
    }
    agent = RLScalper(action_space=["buy", "sell"])
    q_table = agent.train(df, indicators)
    assert isinstance(q_table, dict)
    assert len(q_table) == len(agent.state_space)


def test_rl_save_load(tmp_path):
    df = pd.DataFrame(
        {
            "open": [1, 2, 3],
            "close": [2, 3, 4],
            "gain_z_positive": [True, False, True],
            "ema_trend": ["up", "down", "up"],
        }
    )
    indicators = {
        "gain_z_positive": [True, False],
        "ema_trend": ["up", "down"],
    }
    agent = RLScalper(action_space=["buy", "sell"])
    agent.train(df, indicators)
    fpath = tmp_path / "qtable.json"
    agent.save_q_table(fpath)
    assert fpath.exists()
    loaded = RLScalper.load_q_table(fpath, ["buy", "sell"])
    assert loaded.q_table == agent.q_table

