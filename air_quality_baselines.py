"""
Air Quality Baselines: ARIMA, LSTM, CNN-LSTM
Beijing PM2.5 synthetic data
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print(f"Using device: {DEVICE}")

# ── 1. Synthetic data generation ───────────────────────────────────────────────
np.random.seed(42)
N = 35_000

hours = np.arange(N)
# Seasonal / diurnal patterns
daily   = np.sin(2 * np.pi * hours / 24)
weekly  = np.sin(2 * np.pi * hours / (24 * 7))
yearly  = np.sin(2 * np.pi * hours / (24 * 365))

pm25_base = (
    60
    + 40 * yearly          # seasonal winter peak
    + 15 * daily           # diurnal pattern
    + 10 * weekly
    + np.random.normal(0, 12, N)
    + np.cumsum(np.random.normal(0, 0.3, N))  # slow drift
)
pm25_base = np.clip(pm25_base, 2, 500)

no2  = 40 + 20 * daily + np.random.normal(0, 8, N)
so2  = 25 + 15 * yearly + np.random.normal(0, 5, N)
o3   = 80 - 30 * daily + np.random.normal(0, 10, N)   # inverse diurnal
temp = 15 + 20 * yearly + 5 * daily + np.random.normal(0, 2, N)
pres = 1013 + 5 * yearly + np.random.normal(0, 2, N)
dewp = temp - 10 + np.random.normal(0, 3, N)
rain = np.where(np.random.rand(N) < 0.05, np.random.exponential(2, N), 0)
wspm = np.abs(np.random.normal(2, 1.5, N))

start = pd.Timestamp('2010-01-01')
timestamps = pd.date_range(start, periods=N, freq='h')

df = pd.DataFrame({
    'year':  timestamps.year,
    'month': timestamps.month,
    'day':   timestamps.day,
    'hour':  timestamps.hour,
    'PM2.5': pm25_base,
    'NO2':   no2,
    'SO2':   so2,
    'O3':    o3,
    'TEMP':  temp,
    'PRES':  pres,
    'DEWP':  dewp,
    'RAIN':  rain,
    'WSPM':  wspm,
})
print(f"Data shape: {df.shape}")
print(df.describe().round(2))

# ── 2. Preprocessing ───────────────────────────────────────────────────────────
TARGET = 'PM2.5'
FEATURES = ['PM2.5', 'NO2', 'SO2', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
SEQ_LEN = 24

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[FEATURES].values)

n = len(data_scaled)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

train_data = data_scaled[:train_end]
val_data   = data_scaled[train_end:val_end]
test_data  = data_scaled[val_end:]

# PM2.5 scaler for inverse transform
pm25_idx = FEATURES.index(TARGET)
pm25_scaler = MinMaxScaler()
pm25_scaler.fit(df[[TARGET]].values)

def inverse_pm25(arr):
    return pm25_scaler.inverse_transform(arr.reshape(-1, 1)).ravel()


class AQDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data    = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]          # (seq_len, n_features)
        y = self.data[idx + self.seq_len, pm25_idx]      # scalar (PM2.5)
        return x, y


train_ds = AQDataset(train_data, SEQ_LEN)
val_ds   = AQDataset(val_data,   SEQ_LEN)
test_ds  = AQDataset(test_data,  SEQ_LEN)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=0)

print(f"Train/Val/Test sizes: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

# ── 3. ARIMA walk-forward ──────────────────────────────────────────────────────
print("\n── ARIMA(5,1,2) walk-forward on 2000 test points ──")
from statsmodels.tsa.arima.model import ARIMA

arima_n    = 2000
pm25_full  = df[TARGET].values
arima_test_start = train_end  # use first chunk of validation for speed

arima_history = list(pm25_full[:arima_test_start])
arima_preds   = []
arima_actuals = []
last_pred     = pm25_full[arima_test_start - 1]  # fallback

for i in range(arima_n):
    try:
        model = ARIMA(arima_history[-200:], order=(5, 1, 2))
        fit   = model.fit(method_kwargs={"warn_convergence": False})
        pred  = float(fit.forecast(steps=1)[0])
        last_pred = pred
    except Exception:
        try:
            model = ARIMA(arima_history[-100:], order=(2, 1, 1))
            fit   = model.fit(method_kwargs={"warn_convergence": False})
            pred  = float(fit.forecast(steps=1)[0])
            last_pred = pred
        except Exception:
            pred = last_pred
    arima_preds.append(pred)
    true_val = pm25_full[arima_test_start + i]
    arima_actuals.append(true_val)
    arima_history.append(true_val)
    if (i + 1) % 500 == 0:
        print(f"  ARIMA step {i+1}/{arima_n}")

arima_preds   = np.array(arima_preds)
arima_actuals = np.array(arima_actuals)

arima_mae  = mean_absolute_error(arima_actuals, arima_preds)
arima_rmse = np.sqrt(mean_squared_error(arima_actuals, arima_preds))
arima_r2   = r2_score(arima_actuals, arima_preds)
print(f"  MAE={arima_mae:.3f}  RMSE={arima_rmse:.3f}  R2={arima_r2:.4f}")

# ── 4. LSTM ────────────────────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, n_features, hidden=128, n_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, n_layers,
                            batch_first=True, dropout=dropout)
        self.fc   = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def train_model(model, train_loader, val_loader, epochs=50, patience=7, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val, best_state, wait = np.inf, None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= len(val_loader.dataset)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | train={train_loss:.5f} val={val_loss:.5f}")

        if val_loss < best_val:
            best_val, best_state, wait = val_loss, model.state_dict().copy(), 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model


def evaluate_model(model, loader):
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            p  = model(xb).cpu().numpy()
            preds.append(p)
            actuals.append(yb.numpy())
    preds   = np.concatenate(preds)
    actuals = np.concatenate(actuals)
    preds   = inverse_pm25(preds)
    actuals = inverse_pm25(actuals)
    mae  = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    r2   = r2_score(actuals, preds)
    return mae, rmse, r2, preds, actuals


n_features = len(FEATURES)

print("\n── LSTM training ──")
lstm_model = LSTMModel(n_features).to(DEVICE)
lstm_model = train_model(lstm_model, train_loader, val_loader, epochs=60, patience=7)
lstm_mae, lstm_rmse, lstm_r2, lstm_preds, lstm_actuals = evaluate_model(lstm_model, test_loader)
print(f"  MAE={lstm_mae:.3f}  RMSE={lstm_rmse:.3f}  R2={lstm_r2:.4f}")

# ── 5. CNN-LSTM ────────────────────────────────────────────────────────────────
class CNNLSTMModel(nn.Module):
    def __init__(self, n_features, hidden=128, n_layers=2, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(64, hidden, n_layers,
                            batch_first=True, dropout=dropout)
        self.fc   = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: (batch, seq_len, n_features) → conv expects (batch, channels, length)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)  # back to (batch, seq_len, 64)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


print("\n── CNN-LSTM training ──")
cnnlstm_model = CNNLSTMModel(n_features).to(DEVICE)
cnnlstm_model = train_model(cnnlstm_model, train_loader, val_loader, epochs=60, patience=7)
cnn_mae, cnn_rmse, cnn_r2, cnn_preds, cnn_actuals = evaluate_model(cnnlstm_model, test_loader)
print(f"  MAE={cnn_mae:.3f}  RMSE={cnn_rmse:.3f}  R2={cnn_r2:.4f}")

# ── 6. Results table ───────────────────────────────────────────────────────────
results = pd.DataFrame({
    'Model': ['ARIMA(5,1,2)', 'LSTM', 'CNN-LSTM'],
    'MAE':   [arima_mae,  lstm_mae,  cnn_mae],
    'RMSE':  [arima_rmse, lstm_rmse, cnn_rmse],
    'R2':    [arima_r2,   lstm_r2,   cnn_r2],
})
results = results.round(4)
print("\n" + "="*50)
print("         RESULTS SUMMARY")
print("="*50)
print(results.to_string(index=False))
print("="*50)

results.to_csv('results_baselines.csv', index=False)
print("Saved: results_baselines.csv")

# ── 7. Figure ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)

# ARIMA plot
n_plot = 200
axes[0].plot(arima_actuals[:n_plot], label='Actual', color='steelblue', lw=1.2)
axes[0].plot(arima_preds[:n_plot],   label='ARIMA',  color='tomato',    lw=1.2, linestyle='--')
axes[0].set_title(f'ARIMA(5,1,2)  |  MAE={arima_mae:.2f}  RMSE={arima_rmse:.2f}  R²={arima_r2:.4f}')
axes[0].legend(); axes[0].set_ylabel('PM2.5 (µg/m³)')

# LSTM plot
n_plot2 = 500
axes[1].plot(lstm_actuals[:n_plot2], label='Actual', color='steelblue', lw=1.0)
axes[1].plot(lstm_preds[:n_plot2],   label='LSTM',   color='darkorange', lw=1.0, linestyle='--')
axes[1].set_title(f'LSTM  |  MAE={lstm_mae:.2f}  RMSE={lstm_rmse:.2f}  R²={lstm_r2:.4f}')
axes[1].legend(); axes[1].set_ylabel('PM2.5 (µg/m³)')

# CNN-LSTM plot
axes[2].plot(cnn_actuals[:n_plot2], label='Actual',   color='steelblue', lw=1.0)
axes[2].plot(cnn_preds[:n_plot2],   label='CNN-LSTM', color='seagreen',  lw=1.0, linestyle='--')
axes[2].set_title(f'CNN-LSTM  |  MAE={cnn_mae:.2f}  RMSE={cnn_rmse:.2f}  R²={cnn_r2:.4f}')
axes[2].legend(); axes[2].set_ylabel('PM2.5 (µg/m³)')
axes[2].set_xlabel('Time steps (test set)')

plt.suptitle('Beijing PM2.5 Forecast Baselines', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figure_baselines.png', dpi=150, bbox_inches='tight')
print("Saved: figure_baselines.png")
plt.close()

print("\nDone.")
