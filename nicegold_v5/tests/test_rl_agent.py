import pandas as pd
from nicegold_v5.rl_agent import RLScalper


def test_rl_training():
    df = pd.DataFrame({
        'open': [1, 2, 3, 2, 3],
        'close': [2, 3, 2, 3, 4],
    })
    agent = RLScalper(lr=0.5, gamma=0.9, eps=0.2)
    q_table = agent.train(df)
    assert isinstance(q_table, dict)
    assert len(q_table) > 0

