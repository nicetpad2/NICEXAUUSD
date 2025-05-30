import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
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
    if torch.cuda.is_available():
        model = model.cuda()

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.BCELoss()
    if optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in loader:
            if torch.cuda.is_available():
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} – Loss: {total_loss:.4f}")
    return model


if __name__ == "__main__":
    X, y = load_dataset()
    model = train_lstm(X, y)
    torch.save(model.state_dict(), "models/model_lstm_tp2.pth")
    print("✅ Model saved to models/model_lstm_tp2.pth")
