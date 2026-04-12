"""
Phase 0.3: DeepAR-Style Probabilistic Forecasting Example

A minimal PyTorch implementation that demonstrates the same concepts as GluonTS DeepAR:
- Autoregressive recurrent network (LSTM)
- Outputs Gaussian parameters (mean, variance) for probabilistic forecasts
- Trained on financial returns

Use this when GluonTS cannot be installed (e.g., Python 3.14 + scipy build issues).
For full GluonTS DeepAR, use Python 3.10-3.11 and: pip install gluonts[torch]

Usage:
    source venv/bin/activate
    python phase0_examples/deepar_style_example.py
"""

import os
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Use yfinance for sample data if available
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_config"
warnings.filterwarnings("ignore")


# ============== Probabilistic LSTM (DeepAR-style) ==============
class ProbabilisticLSTM(nn.Module):
    """
    Autoregressive LSTM that outputs Gaussian (mean, log_var) for each step.
    Similar to DeepAR: predicts distribution parameters rather than point estimates.
    """

    def __init__(self, input_dim=1, hidden_dim=32, num_layers=2, output_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True
        )
        self.fc_mean = nn.Linear(hidden_dim, output_dim)
        self.fc_logvar = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # Take last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_dim)
        mean = self.fc_mean(last_hidden)
        log_var = self.fc_logvar(last_hidden)
        return mean, log_var


def gaussian_nll_loss(y_true, mean, log_var):
    """Negative log-likelihood for Gaussian distribution."""
    var = torch.exp(log_var) + 1e-6
    nll = 0.5 * (torch.log(var) + (y_true - mean) ** 2 / var)
    return nll.mean()


def generate_synthetic_returns(n=500, seed=42):
    """Generate synthetic financial returns for demo."""
    np.random.seed(seed)
    # Random walk with drift and volatility
    returns = 0.0002 + 0.01 * np.random.randn(n)
    return returns.astype(np.float32)


def download_spy_returns(days=500):
    """Download S&P 500 ETF returns from Yahoo Finance."""
    if not HAS_YFINANCE:
        return None
    ticker = yf.Ticker("SPY")
    hist = ticker.history(period=f"{days}d")
    if hist is None or len(hist) < 100:
        return None
    closes = hist["Close"].values
    returns = np.diff(np.log(closes)).astype(np.float32)
    return returns


def create_sequences(data, seq_len=20):
    """Create (X, y) sequences for training."""
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)


def main():
    print("=" * 60)
    print("DeepAR-Style Probabilistic Forecasting Example")
    print("=" * 60)

    # Get data
    if HAS_YFINANCE:
        returns = download_spy_returns(500)
        data_source = "Yahoo Finance (SPY)"
    else:
        returns = generate_synthetic_returns(500)
        data_source = "synthetic"

    if returns is None:
        returns = generate_synthetic_returns(500)
        data_source = "synthetic"

    print(f"Data source: {data_source}")
    print(f"Number of observations: {len(returns)}")

    # Create sequences
    SEQ_LEN = 20
    X, y = create_sequences(returns, SEQ_LEN)
    X = X[:, :, np.newaxis]  # (N, seq_len, 1)
    y = y[:, np.newaxis]  # (N, 1)

    # To tensors
    X_t = torch.FloatTensor(X)
    y_t = torch.FloatTensor(y)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model
    device = torch.device("cpu")  # Use "cuda" if available
    model = ProbabilisticLSTM(
        input_dim=1, hidden_dim=32, num_layers=2, output_dim=1
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train
    model.train()
    n_epochs = 50
    for epoch in range(n_epochs):
        total_loss = 0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            mean, log_var = model(X_batch)
            loss = gaussian_nll_loss(y_batch, mean, log_var)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(loader):.6f}")

    # Inference: get mean and uncertainty (std) for last sequence
    model.eval()
    with torch.no_grad():
        last_seq = X_t[-1:].to(device)  # (1, seq_len, 1)
        mean, log_var = model(last_seq)
        std = torch.exp(0.5 * log_var).cpu().numpy()
        mean_np = mean.cpu().numpy()

    print("\n" + "-" * 40)
    print("Probabilistic forecast (last timestep):")
    print(f"  Mean (expected return): {mean_np[0, 0]:.6f}")
    print(f"  Std (uncertainty):      {std[0, 0]:.6f}")
    print(f"  95% interval: [{mean_np[0,0]-1.96*std[0,0]:.6f}, {mean_np[0,0]+1.96*std[0,0]:.6f}]")
    print("-" * 40)

    print("\n" + "=" * 60)
    print("Done! This demonstrates:")
    print("  - LSTM outputs (mean, variance) for aleatoric uncertainty")
    print("  - Same concept as DeepAR in GluonTS")
    print("  - Ready to integrate into PPO state space (Phase 2)")
    print("=" * 60)


if __name__ == "__main__":
    main()
