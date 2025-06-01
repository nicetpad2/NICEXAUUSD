import os
from typing import List
import numpy as np
import pandas as pd

try:  # pragma: no cover - optional torch
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
except Exception:  # pragma: no cover - lightweight stub for tests
    import types
    torch = types.ModuleType('torch')
    torch.float32 = 0.0
    torch.cat = lambda tensors, dim=0: np.concatenate([np.asarray(t) for t in tensors], axis=dim)
    torch.from_numpy = lambda arr: np.array(arr, dtype=np.float32)
    torch.no_grad = lambda: (yield)
    class _Tensor(np.ndarray):
        def __new__(cls, arr):
            obj = np.asarray(arr, dtype=np.float32).view(cls)
            return obj
        def to(self, device):
            return self
        def unsqueeze(self, dim):
            return np.expand_dims(self, dim).view(_Tensor)
    torch.tensor = lambda d, dtype=None: _Tensor(d)

    class _Module:
        def __call__(self, *a, **k):
            return self.forward(*a, **k)
        def forward(self, *a, **k):
            raise NotImplementedError
        def to(self, device):
            return self
        def eval(self):
            pass
        def train(self):
            pass
        def load_state_dict(self, state):
            pass
        def state_dict(self):  # pragma: no cover - unused
            return {}
    class _LSTM(_Module):
        def __init__(self, input_size, hidden_size, num_layers, batch_first=True, dropout=0.1):
            self.hidden_size = hidden_size
        def forward(self, x):
            batch, seq, _ = x.shape
            return np.zeros((batch, seq, self.hidden_size)), None
    class _Linear(_Module):
        def __init__(self, in_f, out_f):
            self.out_f = out_f
        def forward(self, x):
            batch = x.shape[0]
            return np.zeros((batch, self.out_f))
    class _ReLU(_Module):
        def forward(self, x):
            return x
    class _Sigmoid(_Module):
        def forward(self, x):
            return x
    class _Sequential(_Module):
        def __init__(self, *layers):
            self.layers = layers
        def forward(self, x):
            for l in self.layers:
                x = l(x)
            return x
    nn = types.SimpleNamespace(
        Module=_Module,
        LSTM=_LSTM,
        Linear=_Linear,
        ReLU=_ReLU,
        Sigmoid=_Sigmoid,
        Sequential=_Sequential,
    )
    Dataset = object
    DataLoader = lambda dataset, batch_size=1, shuffle=False, drop_last=False: [dataset[i] for i in range(len(dataset))]
    torch.nn = nn
    torch.utils = types.SimpleNamespace(data=types.SimpleNamespace(Dataset=Dataset, DataLoader=DataLoader))
    torch.save = lambda obj, path: None
    torch.load = lambda path, map_location=None: {}


class ThresholdDataset(Dataset):
    """Dataset สำหรับโมเดล Adaptive Threshold"""
    def __init__(self, seq_data: np.ndarray, feedback_data: np.ndarray, targets: np.ndarray):
        assert len(seq_data) == len(feedback_data) == len(targets)
        self.seq = seq_data.astype(np.float32)
        self.fb = feedback_data.astype(np.float32)
        self.tgt = targets.astype(np.float32)

    def __len__(self):
        return len(self.tgt)

    def __getitem__(self, idx):
        return {
            "seq": torch.from_numpy(self.seq[idx]),
            "feedback": torch.from_numpy(self.fb[idx]),
            "target": torch.from_numpy(self.tgt[idx]),
        }


class ThresholdPredictor(nn.Module):
    def __init__(self, num_indicators: int = 3, seq_len: int = 60, hidden_size: int = 64, num_layers: int = 2, feedback_dim: int = 3, fc_hidden: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=num_indicators, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.1)
        self.feedback_fc = nn.Sequential(
            nn.Linear(feedback_dim, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, fc_hidden),
            nn.ReLU(),
        )
        self.combine_fc = nn.Sequential(
            nn.Linear(hidden_size + fc_hidden, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, 3),
            nn.Sigmoid(),
        )

    def forward(self, seq_batch, feedback_batch):
        lstm_out, _ = self.lstm(seq_batch)
        lstm_last = lstm_out[:, -1, :]
        fb_out = self.feedback_fc(feedback_batch)
        combined = torch.cat([lstm_last, fb_out], dim=1)
        out = self.combine_fc(combined)
        return out


def load_wfv_training_data(logs_dir: str, seq_len: int = 60, lookback_step: int = 1) -> ThresholdDataset:
    seq_list: List[np.ndarray] = []
    fb_list: List[np.ndarray] = []
    target_list: List[np.ndarray] = []
    for fname in os.listdir(logs_dir):
        if not fname.startswith("wfv_results_fold") or not fname.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(logs_dir, fname), parse_dates=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        req = {"optimal_gain_z_thresh", "optimal_ema_slope_thresh", "optimal_atr_thresh"}
        if not req.issubset(df.columns):
            continue
        indicators = df[["gain_z", "ema_slope", "atr"]].values
        pnl = df["pnl"].values[-1]
        max_dd = df["max_dd"].values[-1]
        winrate = df["winrate"].values[-1]
        feedback = np.array([pnl, max_dd, winrate], dtype=np.float32)
        target = np.array([
            df["optimal_gain_z_thresh"].values[-1],
            df["optimal_ema_slope_thresh"].values[-1],
            df["optimal_atr_thresh"].values[-1],
        ], dtype=np.float32)
        N = len(indicators)
        for i in range(seq_len - 1, N, lookback_step):
            seq = indicators[i - seq_len + 1 : i + 1]
            seq_list.append(seq)
            fb_list.append(feedback)
            target_list.append(target)
    seq_arr = np.stack(seq_list, axis=0)
    fb_arr = np.stack(fb_list, axis=0)
    tgt_arr = np.stack(target_list, axis=0)
    return ThresholdDataset(seq_arr, fb_arr, tgt_arr)


def train_threshold_predictor(logs_dir: str, model_save_path: str = "model/threshold_predictor.pt", epochs: int = 50, batch_size: int = 128, lr: float = 1e-3, seq_len: int = 60, device: str = "cuda" if hasattr(torch, "cuda") and callable(getattr(torch.cuda, "is_available", lambda: False)) and torch.cuda.is_available() else "cpu"):
    dataset = load_wfv_training_data(logs_dir=logs_dir, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model = ThresholdPredictor(num_indicators=3, seq_len=seq_len, feedback_dim=3).to(device)
    criterion = nn.MSELoss()
    optimizer = getattr(torch.optim, "Adam", lambda params, lr: None)(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        total = 0.0
        for batch in dataloader:
            seq = batch["seq"].to(device)
            fb = batch["feedback"].to(device)
            tgt = batch["target"].to(device)
            pred = model(seq, fb)
            loss = criterion(pred, tgt)
            if hasattr(optimizer, "zero_grad"):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += float(loss) * len(seq)
        avg = total / len(dataset) if len(dataset) else 0
        print(f"[AdaptiveThreshold-DL] Epoch {epoch}/{epochs} – Loss: {avg:.6f}")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"[AdaptiveThreshold-DL] Saved model → {model_save_path}")


def predict_thresholds(seq_recent: np.ndarray, feedback_scalar: List[float], model_path: str = "model/threshold_predictor.pt", device: str = "cuda" if hasattr(torch, "cuda") and callable(getattr(torch.cuda, "is_available", lambda: False)) and torch.cuda.is_available() else "cpu") -> dict:
    model = ThresholdPredictor(num_indicators=3, seq_len=seq_recent.shape[1], feedback_dim=3).to(device)
    state = torch.load(model_path, map_location=device)
    if hasattr(model, "load_state_dict"):
        model.load_state_dict(state)
    model.eval()
    seq_tensor = torch.from_numpy(seq_recent.astype(np.float32))
    fb_tensor = torch.from_numpy(np.array(feedback_scalar, dtype=np.float32)).unsqueeze(0)
    with getattr(torch, "no_grad", lambda: (yield))():
        pred = model(seq_tensor, fb_tensor)
    pred = np.asarray(pred)
    return {
        "gain_z_thresh": float(pred.reshape(-1)[0]),
        "ema_slope_min": float(pred.reshape(-1)[1]),
        "atr_thresh": float(pred.reshape(-1)[2]),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Adaptive Threshold Model")
    parser.add_argument("--logs_dir", type=str, default="logs/", help="WFV logs directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq_len", type=int, default=60)
    args = parser.parse_args()
    train_threshold_predictor(args.logs_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seq_len=args.seq_len)
