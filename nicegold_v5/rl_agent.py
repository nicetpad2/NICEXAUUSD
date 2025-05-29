import numpy as np
import pandas as pd

class RLScalper:
    """Simple Q-learning agent for scalping."""

    def __init__(self, lr: float = 0.1, gamma: float = 0.95, eps: float = 0.1):
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.q_table: dict[int, np.ndarray] = {}

    def _state(self, row: pd.Series) -> int:
        # 0 = price down, 1 = price up
        return int(row.get("close", 0) > row.get("open", 0))

    def act(self, state: int) -> int:
        if np.random.rand() < self.eps or state not in self.q_table:
            return np.random.choice([0, 1])  # 0 buy, 1 sell
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float, next_state: int) -> None:
        self.q_table.setdefault(state, np.zeros(2))
        self.q_table.setdefault(next_state, np.zeros(2))
        best_next = np.max(self.q_table[next_state])
        old = self.q_table[state][action]
        self.q_table[state][action] = old + self.lr * (reward + self.gamma * best_next - old)

    def train(self, df: pd.DataFrame) -> dict:
        df = df.reset_index(drop=True)
        for i in range(len(df) - 1):
            row = df.iloc[i]
            next_row = df.iloc[i + 1]
            state = self._state(row)
            next_state = self._state(next_row)
            action = self.act(state)
            price_diff = next_row.get("close", 0) - row.get("close", 0)
            reward = price_diff if action == 0 else -price_diff
            self.update(state, action, reward, next_state)
        return self.q_table
