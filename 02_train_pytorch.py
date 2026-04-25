"""
02_train_pytorch.py — Entraîne LSTM, CNN-LSTM, GCN+Transformer
avec affichage détaillé à chaque epoch. Lance après avoir arrêté 02_train_all_models.py.
ARIMA et XGBoost déjà faits — on reprend ici.
"""
import numpy as np
import pandas as pd
import json, time, warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────
DATA_PATH  = "data/beijing_real_combined.csv"
TARGET     = "PM2.5"
FEATURES   = ["NO2", "TEMP", "PRES", "DEWP", "WSPM"]
SEQ_LEN    = 24
BATCH_SIZE = 128      # plus grand = plus rapide
MAX_EPOCHS = 30       # réduit
PATIENCE   = 5        # réduit
SEEDS      = [42, 123, 777]

# Résultats déjà obtenus (ARIMA + XGBoost)
results = {
    "ARIMA": {
        "MAE": 4.17, "RMSE": 6.51, "R2": 0.9726
    },
    "XGBoost": {
        "MAE": 6.35, "MAE_std": 0.22,
        "RMSE": 11.67, "RMSE_std": 0.35,
        "R2": 0.9631
    }
}

# ── CHARGEMENT ────────────────────────────────────────
print("Chargement données...")
df = pd.read_csv(DATA_PATH, parse_dates=["datetime"]).sort_values("datetime").reset_index(drop=True)
n = len(df)
n_train, n_val = int(0.70*n), int(0.85*n)
print(f"{n:,} enregistrements | train={n_train:,} val={n_val-n_train:,} test={n-n_val:,}")

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_all = scaler_X.fit_transform(df[FEATURES].values)
y_all = scaler_y.fit_transform(df[[TARGET]].values).ravel()

X_train = X_all[:n_train]; X_val = X_all[n_train:n_val]; X_test = X_all[n_val:]
y_train = y_all[:n_train]; y_val = y_all[n_train:n_val]; y_test = y_all[n_val:]

def make_seq(X, y, L):
    Xs, ys = [], []
    for i in range(L, len(X)):
        Xs.append(X[i-L:i]); ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

Xs_tr, ys_tr = make_seq(np.column_stack([y_train, X_train]), y_train, SEQ_LEN)
Xs_va, ys_va = make_seq(np.column_stack([y_val,   X_val]),   y_val,   SEQ_LEN)
Xs_te, ys_te = make_seq(np.column_stack([y_test,  X_test]),  y_test,  SEQ_LEN)
print(f"Séquences train:{Xs_tr.shape} test:{Xs_te.shape}")

device = (torch.device("mps")  if torch.backends.mps.is_available() else
          torch.device("cuda") if torch.cuda.is_available()          else
          torch.device("cpu"))
print(f"Device: {device}\n")

def to_dl(X, y, shuffle=False):
    ds = TensorDataset(torch.from_numpy(X).to(device),
                       torch.from_numpy(y).to(device))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

tr_dl = to_dl(Xs_tr, ys_tr, shuffle=False)
va_dl = to_dl(Xs_va, ys_va)
te_dl = to_dl(Xs_te, ys_te)
IN_F  = Xs_tr.shape[2]

def descale(y_norm):
    return scaler_y.inverse_transform(y_norm.reshape(-1,1)).ravel()

def calc_metrics(yt_n, yp_n):
    yt = descale(yt_n)
    yp = np.clip(descale(yp_n), 0, None)
    return dict(MAE=float(mean_absolute_error(yt,yp)),
                RMSE=float(np.sqrt(mean_squared_error(yt,yp))),
                R2=float(r2_score(yt,yp)))

# ── MODÈLES ───────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(IN_F, 128, 2, batch_first=True, dropout=0.2)
        self.fc  = nn.Linear(128, 1)
    def forward(self, x):
        return self.fc(self.rnn(x)[0][:, -1]).squeeze(-1)

class CNNLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(IN_F, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64,   64, 3, padding=1), nn.ReLU())
        self.rnn = nn.LSTM(64, 128, 2, batch_first=True, dropout=0.2)
        self.fc  = nn.Linear(128, 1)
    def forward(self, x):
        x = self.cnn(x.permute(0,2,1)).permute(0,2,1)
        return self.fc(self.rnn(x)[0][:, -1]).squeeze(-1)

class GCNTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(IN_F, 64)
        enc = nn.TransformerEncoderLayer(64, 4, 256, 0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, 2)
        self.fc  = nn.Linear(64, 1)
    def forward(self, x):
        return self.fc(self.enc(self.proj(x))[:, -1]).squeeze(-1)

# ── ENTRAÎNEMENT ──────────────────────────────────────
def train_model(ModelClass, name):
    print(f"\n{'─'*50}")
    print(f"Entraînement : {name}")
    print(f"{'─'*50}")
    seed_results = []

    for seed in SEEDS:
        torch.manual_seed(seed)
        model = ModelClass().to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        crit  = nn.MSELoss()
        best_val, pat, best_state = 1e9, 0, None
        t0 = time.time()

        for ep in range(1, MAX_EPOCHS+1):
            # Train
            model.train()
            train_loss = 0
            for xb, yb in tr_dl:
                opt.zero_grad()
                loss = crit(model(xb), yb)
                loss.backward()
                opt.step()
                train_loss += loss.item()

            # Val
            model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in va_dl:
                    val_losses.append(crit(model(xb), yb).item())
            val_loss = np.mean(val_losses)

            elapsed = time.time() - t0
            print(f"  seed={seed} ep={ep:2d}/{MAX_EPOCHS}  "
                  f"train={train_loss/len(tr_dl):.4f}  "
                  f"val={val_loss:.4f}  "
                  f"({elapsed:.0f}s)", flush=True)

            if val_loss < best_val:
                best_val = val_loss
                pat = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                pat += 1
                if pat >= PATIENCE:
                    print(f"  → Early stop at epoch {ep}")
                    break

        # Test
        model.load_state_dict(best_state)
        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in te_dl:
                preds.append(model(xb).cpu().numpy())
        preds = np.concatenate(preds)
        r = calc_metrics(ys_te, preds)
        seed_results.append(r)
        print(f"  ✓ seed={seed} → MAE={r['MAE']:.2f} RMSE={r['RMSE']:.2f} R²={r['R2']:.4f}")

    results[name] = dict(
        MAE=float(np.mean([r["MAE"]  for r in seed_results])),
        MAE_std=float(np.std( [r["MAE"]  for r in seed_results])),
        RMSE=float(np.mean([r["RMSE"] for r in seed_results])),
        RMSE_std=float(np.std( [r["RMSE"] for r in seed_results])),
        R2=float(np.mean([r["R2"]   for r in seed_results])))
    print(f"  MOYENNE → MAE={results[name]['MAE']:.2f}±{results[name]['MAE_std']:.2f}  "
          f"RMSE={results[name]['RMSE']:.2f}±{results[name]['RMSE_std']:.2f}  "
          f"R²={results[name]['R2']:.4f}")

    # Sauvegarder après chaque modèle (sécurité)
    with open("results_real.json","w") as f:
        json.dump(results, f, indent=2)
    print(f"  → results_real.json mis à jour")

# ── LANCER ────────────────────────────────────────────
train_model(LSTMModel,      "LSTM")
train_model(CNNLSTMModel,   "CNN-LSTM")
train_model(GCNTransformer, "GCN+Transformer")

# ── RÉSUMÉ FINAL ──────────────────────────────────────
print("\n" + "="*55)
print("RÉSULTATS FINAUX — Beijing réel")
print("="*55)
for name, r in results.items():
    if "MAE" in r:
        std_mae  = f"±{r.get('MAE_std',0):.2f}"
        std_rmse = f"±{r.get('RMSE_std',0):.2f}"
        print(f"  {name:<22}  MAE={r['MAE']:.2f}{std_mae}  "
              f"RMSE={r['RMSE']:.2f}{std_rmse}  R²={r['R2']:.4f}")
print("="*55)
print("\n✓ results_real.json prêt → envoie ce fichier à Claude !")
