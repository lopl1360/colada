from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
except ImportError:  # pragma: no cover - torch may not be installed
    torch = None  # type: ignore
    nn = None  # type: ignore
    Dataset = object  # type: ignore
    DataLoader = object  # type: ignore


class LSTMPricePredictor(nn.Module):
    """Simple LSTM network for price prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        if torch is None:
            raise ImportError("PyTorch is required to use LSTMPricePredictor")
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


class PriceSentimentDataset(Dataset):
    """Dataset that creates sequences from price and sentiment features."""

    def __init__(
        self,
        csv_path: str,
        seq_len: int = 60,
        target_col: str = "target",
        feature_cols: Optional[List[str]] = None,
    ):
        if torch is None:
            raise ImportError("PyTorch is required to use PriceSentimentDataset")
        df = pd.read_csv(csv_path)
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c not in {target_col, "timestamp"}]

        data = df[feature_cols].values.astype(np.float32)
        target = df[target_col].values.astype(np.float32)

        sequences = []
        targets = []
        for i in range(len(df) - seq_len):
            sequences.append(data[i : i + seq_len])
            targets.append(target[i + seq_len])
        self.X = np.stack(sequences)
        self.y = np.array(targets)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_lstm(
    csv_path: str,
    seq_len: int = 60,
    batch_size: int = 32,
    num_epochs: int = 10,
    lr: float = 1e-3,
    target_col: str = "target",
    feature_cols: Optional[List[str]] = None,
    classification: bool = False,
    model_path: str = "lstm_model.pt",
):
    """Train the LSTM model using a CSV with price and sentiment features."""
    if torch is None:
        raise ImportError("PyTorch is required to train the LSTM model")

    dataset = PriceSentimentDataset(csv_path, seq_len, target_col, feature_cols)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = dataset.X.shape[2]
    model = LSTMPricePredictor(input_dim, output_dim=1)

    criterion = nn.BCEWithLogitsLoss() if classification else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            y_batch = y_batch.unsqueeze(-1)
            if classification:
                y_batch = (y_batch > 0).float()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)
        avg_loss = epoch_loss / len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")
    return model


__all__ = [
    "LSTMPricePredictor",
    "PriceSentimentDataset",
    "train_lstm",
]
