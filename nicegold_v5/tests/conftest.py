import sys
from pathlib import Path

import numpy as np
import pytest

# Speed up tests by using a lightweight RandomForest stub
class _FastRF:
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, X, y):
        self.classes_ = [0, 1]
        return self

    def predict_proba(self, X):
        return np.tile([0.5, 0.5], (len(X), 1))

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))


@pytest.fixture(autouse=True)
def fast_rf(monkeypatch):
    """Patch RandomForestClassifier in wfv to avoid slow training."""
    monkeypatch.setattr("nicegold_v5.wfv.RandomForestClassifier", _FastRF)
