import inspect
import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """Simple LSTM-based classifier."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if "dropout" in inspect.signature(nn.LSTM).parameters:
            self.lstm = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        else:  # pragma: no cover - legacy stub without dropout
            self.lstm = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
            )
        self.dropout = nn.Dropout(dropout) if hasattr(nn, "Dropout") else lambda x: x
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Return raw logits for a batch."""
        out, _ = self.lstm(x)  # (batch, seq_len, hidden)
        out = out[:, -1, :]
        out = self.dropout(out)
        logits = self.fc(out)
        return logits
