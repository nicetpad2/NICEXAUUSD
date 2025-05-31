import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import time
from .deep_model_m1 import LSTMClassifier


def load_dataset(path="data/ml_dataset_m1.csv", seq_len=10):
    df = pd.read_csv(path)
    features = ["gain_z", "ema_slope", "atr", "rsi", "volume", "entry_score", "pattern_label"]
    data = df[features].values
    labels = df["tp2_hit"].values

    X_seq, y_seq = [], []
    for i in range(len(data) - seq_len):
        X_seq.append(data[i:i + seq_len])
        y_seq.append(labels[i + seq_len])

    X = torch.tensor(np.array(X_seq), dtype=torch.float32)
    y = torch.tensor(np.array(y_seq), dtype=torch.float32).unsqueeze(1)
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
        model.train()
        total_loss = 0.0
        load_time = forward_time = backward_time = step_time = 0.0
        for batch_x, batch_y in loader:
            t0 = time.time()
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            t1 = time.time()
            optimizer.zero_grad()
            with autocast(enabled=use_amp):
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
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
    X, y = load_dataset()
    model = train_lstm(X, y)
    torch.save(model.state_dict(), "models/model_lstm_tp2.pth")
    print("✅ Model saved to models/model_lstm_tp2.pth")
