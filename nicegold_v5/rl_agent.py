import itertools
import numpy as np
import pandas as pd

class RLScalper:
    """Simple Q-learning agent for scalping."""

    def __init__(
        self,
        state_space: list | None = None,
        action_space: list | None = None,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
    ) -> None:
        """Initialize the RL agent.

        Parameters
        ----------
        state_space : list | None
            Iterable of possible states. Defaults to ``[0, 1]`` for backward
            compatibility.
        action_space : list | None
            Iterable of possible actions. Defaults to ``[0, 1]`` which maps to
            ``buy`` and ``sell``.
        alpha : float
            Learning rate.
        gamma : float
            Discount factor.
        epsilon : float
            Probability of choosing a random action.
        """

        if action_space is None:
            action_space = [0, 1]
        if state_space is None:
            state_space = [0, 1]

        self.action_space = action_space
        # Initialize Q-table for all states
        self.state_space = state_space
        self.q_table: dict[tuple, list[float]] = {
            state: [0.0 for _ in action_space] for state in state_space
        }
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def _state(self, row: pd.Series) -> int:
        # 0 = price down, 1 = price up
        return int(row.get("close", 0) > row.get("open", 0))

    @staticmethod
    def generate_all_states(indicators: dict) -> list:
        """Return all possible state tuples given indicator options.

        Parameters
        ----------
        indicators : dict
            Mapping of feature name to possible values.
        """
        keys = sorted(indicators.keys())
        values = [indicators[k] for k in keys]
        combinations = list(itertools.product(*values))
        return [tuple(comb) for comb in combinations]

    def choose_action(self, state) -> int:
        """Choose action using epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
        return self.action_space[int(np.argmax(self.q_table[state]))]

    def update_q(self, state, action, reward: float, next_state) -> None:
        """Update Q-table using the Q-learning update rule."""
        a_idx = self.action_space.index(action)
        next_max = max(self.q_table[next_state])
        current = self.q_table[state][a_idx]
        self.q_table[state][a_idx] = current + self.alpha * (
            reward + self.gamma * next_max - current
        )

    def _get_reward(self, row: pd.Series, action) -> float:
        """Simple reward function based on next price movement."""
        price_diff = row.get("close", 0) - row.get("open", 0)
        return price_diff if action == 0 else -price_diff

    def act(self, state):  # backward compatibility
        return self.choose_action(state)

    def update(self, state, action, reward, next_state):  # backward compatibility
        self.update_q(state, action, reward, next_state)

    def train(self, df: pd.DataFrame, indicators: dict | None = None) -> dict:
        """Train Q-table using provided dataframe."""
        df = df.reset_index(drop=True)
        if indicators:
            self.state_space = self.generate_all_states(indicators)
            self.q_table = {
                state: [0.0 for _ in self.action_space] for state in self.state_space
            }
            keys = sorted(indicators.keys())
            for _, row in df.iterrows():
                # build state tuple in same order as indicators.keys()
                state = tuple(row[k] for k in keys)
                action = self.choose_action(state)
                reward = self._get_reward(row, action)
                next_state = state
                self.update_q(state, action, reward, next_state)
        else:
            for i in range(len(df) - 1):
                row = df.iloc[i]
                next_row = df.iloc[i + 1]
                state = self._state(row)
                next_state = self._state(next_row)
                action = self.choose_action(state)
                price_diff = next_row.get("close", 0) - row.get("close", 0)
                reward = price_diff if action == 0 else -price_diff
                self.update_q(state, action, reward, next_state)
        return self.q_table

    def save_q_table(self, filepath: str) -> None:
        """Save Q-table to JSON file."""
        import json

        with open(filepath, "w") as f:
            json.dump({str(k): v for k, v in self.q_table.items()}, f)

    @classmethod
    def load_q_table(
        cls,
        filepath: str,
        action_space: list,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
    ) -> "RLScalper":
        """Load Q-table from JSON and return initialized agent."""
        import json

        with open(filepath, "r") as f:
            qd = json.load(f)
        agent = cls([tuple(eval(k)) for k in qd.keys()], action_space, alpha, gamma, epsilon)
        agent.q_table = {tuple(eval(k)): v for k, v in qd.items()}
        return agent
