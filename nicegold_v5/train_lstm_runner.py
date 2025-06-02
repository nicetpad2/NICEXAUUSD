import pandas as pd
import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    try:
        from torch.amp import autocast, GradScaler
        _AMP_MODE = "torch.amp"
    except Exception:  # pragma: no cover - fallback for older PyTorch
        from torch.cuda.amp import autocast, GradScaler
        _AMP_MODE = "torch.cuda.amp"
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    TORCH_AVAILABLE = False
    print("[Warning] PyTorch ไม่ถูกติดตั้ง – ข้ามการเทรน LSTM")
    import types
    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        tensor=lambda *a, **k: None,
        zeros=lambda shape, dtype=None: np.zeros(shape, dtype=dtype),
        float32=float,
        device=lambda x=None: 'cpu',
        save=lambda *a, **k: None,
    )
    nn = types.SimpleNamespace(Module=object)
    optim = types.SimpleNamespace(SGD=lambda *a, **k: None, Adam=lambda *a, **k: None)
    DataLoader = lambda *a, **k: []
    TensorDataset = lambda *a, **k: None
    GradScaler = lambda enabled=True: None
    autocast = lambda enabled=True: types.SimpleNamespace(__enter__=lambda self: None, __exit__=lambda self, exc_type, exc, tb: False)
    _AMP_MODE = None
LSTMClassifier = None  # patched in tests or loaded lazily
from .utils import autotune_resource, print_resource_status, dynamic_batch_scaler
import time


def load_dataset(path="data/ml_dataset_m1.csv", seq_len=10):
    df = pd.read_csv(path)
    required = ["tp2_hit"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"ML dataset missing column: {col}")
    label = "tp2_hit"
    features = [c for c in df.columns if c not in ["timestamp", label]]
    data = df[features].values
    labels = df[label].astype(int).values

    X_seq, y_seq = [], []
    for i in range(len(data) - seq_len):
        X_seq.append(data[i:i + seq_len])
        y_seq.append(labels[i + seq_len])

    X_arr = np.array(X_seq)
    y_arr = np.array(y_seq)
    if TORCH_AVAILABLE:
        X = torch.tensor(X_arr, dtype=torch.float32)
        y = torch.tensor(y_arr, dtype=torch.float32).unsqueeze(1)
    else:
        X, y = X_arr, y_arr.reshape(-1, 1)
    return X, y


def train_lstm(
    X,
    y,
    hidden_dim=64,
    epochs=10,
    lr=0.001,
    batch_size=64,
    optimizer_name="adam",
):
    """Train LSTM classifier with selectable optimizer."""
    if not TORCH_AVAILABLE or not getattr(torch, "nn", None) or not hasattr(torch.nn, "LSTM"):
        print("[train_lstm_runner] PyTorch ไม่ถูกติดตั้ง – exit อัตโนมัติ")
        return None
    from .deep_model_m1 import LSTMClassifier

    model = LSTMClassifier(X.shape[2], hidden_dim)
    use_amp = torch.cuda.is_available()
    if use_amp:
        model = model.cuda()
    else:
        print("⚠️ [AMP Warning] No GPU detected – switched to CPU mode with fallback config")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TensorDataset(X, y)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        prefetch_factor=2,
    )
    # [Patch v24.3.2] ⚡️ Use BCEWithLogitsLoss for amp+sigmoid safety
    criterion = nn.BCEWithLogitsLoss()
    optimizer = (
        optim.SGD(model.parameters(), lr=lr)
        if optimizer_name == "sgd"
        else optim.Adam(model.parameters(), lr=lr)
    )

    scaler = GradScaler(enabled=use_amp)

    for epoch in range(epochs):
        print_resource_status()
        model.train()
        total_loss = 0.0
        load_time = forward_time = backward_time = step_time = 0.0
        epoch_true_labels, epoch_preds = [], []
        for batch_x, batch_y in loader:
            t0 = time.time()
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            t1 = time.time()
            optimizer.zero_grad()
            with autocast(enabled=use_amp):
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
            epoch_true_labels.append(batch_y.detach().cpu().numpy())
            epoch_preds.append(preds.detach().cpu().numpy())
            t2 = time.time()
            scaler.scale(loss).backward()
            t3 = time.time()
            scaler.step(optimizer)
            scaler.update()
            t4 = time.time()
            total_loss += loss.item()

            load_time += t1 - t0
            forward_time += t2 - t1
            backward_time += t3 - t2
            step_time += t4 - t3
        print(f"Epoch {epoch+1}/{epochs} – Loss: {total_loss:.4f}")
        print(
            f"⏱️ Time Breakdown (sec): Load {load_time:.2f} | Forward {forward_time:.2f} | Backward {backward_time:.2f} | Step {step_time:.2f}"
        )
        bottlenecks = sorted(
            [
                ("DataLoad", load_time),
                ("Forward", forward_time),
                ("Backward", backward_time),
                ("Step", step_time),
            ],
            key=lambda x: -x[1],
        )
        print(f"🔥 Bottleneck: {bottlenecks[0][0]} ({bottlenecks[0][1]:.2f}s)")
    return model


if __name__ == "__main__":
    device, batch_size = autotune_resource()
    X, y = load_dataset()
    model = train_lstm(X, y, batch_size=batch_size)
    if model is not None:
        torch.save(model.state_dict(), "models/model_lstm_tp2.pth")
        print("✅ Model saved to models/model_lstm_tp2.pth")  # pragma: no cover
