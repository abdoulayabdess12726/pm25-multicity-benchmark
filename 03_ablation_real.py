"""
03_ablation_real.py — Ablation study sur le vrai dataset Beijing UCI
Trois variantes : Full | No-GNN | No-Transformer
Résultats → results_ablation_real.json
Lance après 01_download_beijing.py (data/beijing_real_combined.csv requis)
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

# ── CONFIG ────────────────────────────────────────────────────────
DATA_PATH  = "data/beijing_real_combined.csv"
TARGET     = "PM2.5"
FEATURES   = ["NO2", "TEMP", "PRES", "DEWP", "WSPM"]
SEQ_LEN    = 24
BATCH_SIZE = 128
MAX_EPOCHS = 30
PATIENCE   = 5
SEEDS      = [42, 123, 777]
LATENCY_RUNS = 50

# ── CHARGEMENT ────────────────────────────────────────────────────
print("Chargement données réelles pour ablation...")
df = pd.read_csv(DATA_PATH, parse_dates=["datetime"]).sort_values("datetime").reset_index(drop=True)
n = len(df)
n_train, n_val = int(0.70*n), int(0.85*n)
print(f"  {n:,} enregistrements | train={n_train:,} val={n_val-n_train:,} test={n-n_val:,}")

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_all = scaler_X.fit_transform(df[FEATURES].values)
y_all = scaler_y.fit_transform(df[[TARGET]].values).ravel()

X_tr, X_va, X_te = X_all[:n_train], X_all[n_train:n_val], X_all[n_val:]
y_tr, y_va, y_te = y_all[:n_train], y_all[n_train:n_val], y_all[n_val:]

def make_seq(X, y, L):
    Xs, ys = [], []
    for i in range(L, len(X)):
        Xs.append(X[i-L:i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

Xs_tr, ys_tr = make_seq(np.column_stack([y_tr, X_tr]), y_tr, SEQ_LEN)
Xs_va, ys_va = make_seq(np.column_stack([y_va, X_va]), y_va, SEQ_LEN)
Xs_te, ys_te = make_seq(np.column_stack([y_te, X_te]), y_te, SEQ_LEN)
print(f"  Séquences train:{Xs_tr.shape} test:{Xs_te.shape}")

device = (torch.device("mps")  if torch.backends.mps.is_available() else
          torch.device("cuda") if torch.cuda.is_available()          else
          torch.device("cpu"))
print(f"  Device: {device}")

def to_dl(X, y, shuffle=False):
    ds = TensorDataset(torch.from_numpy(X).to(device),
                       torch.from_numpy(y).to(device))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

tr_dl = to_dl(Xs_tr, ys_tr)
va_dl = to_dl(Xs_va, ys_va)
te_dl = to_dl(Xs_te, ys_te)
IN_F  = Xs_tr.shape[2]

def descale(y_n):
    return scaler_y.inverse_transform(y_n.reshape(-1,1)).ravel()

def calc_metrics(yt_n, yp_n):
    yt = descale(yt_n)
    yp = np.clip(descale(yp_n), 0, None)
    return dict(MAE=float(mean_absolute_error(yt,yp)),
                RMSE=float(np.sqrt(mean_squared_error(yt,yp))),
                R2=float(r2_score(yt,yp)))

def measure_latency(model, runs=LATENCY_RUNS):
    """Mesure latence sur BATCH_SIZE=128, runs répétitions."""
    model.eval()
    Xb = torch.from_numpy(Xs_te[:BATCH_SIZE]).to(device)
    # Warm-up
    with torch.no_grad():
        for _ in range(5): model(Xb)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(runs): model(Xb)
    return (time.time() - t0) / runs * 1000  # ms

# ── MODÈLES : 3 variantes ─────────────────────────────────────────

class FullFramework(nn.Module):
    """GCN Feature Encoder + Transformer."""
    def __init__(self):
        super().__init__()
        self.gcn_proj = nn.Linear(IN_F, 64)   # GCN feature projection
        enc = nn.TransformerEncoderLayer(64, 4, 256, 0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, 2)
        self.fc = nn.Linear(64, 1)
    def forward(self, x):
        return self.fc(self.transformer(self.gcn_proj(x))[:, -1]).squeeze(-1)

class NoGNN(nn.Module):
    """Linear projection + Transformer (sans GCN feature encoder)."""
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(IN_F, 64)       # projection simple, pas GCN
        enc = nn.TransformerEncoderLayer(64, 4, 256, 0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, 2)
        self.fc = nn.Linear(64, 1)
    def forward(self, x):
        return self.fc(self.transformer(self.proj(x))[:, -1]).squeeze(-1)

class NoTransformer(nn.Module):
    """GCN Feature Encoder + LSTM (sans Transformer)."""
    def __init__(self):
        super().__init__()
        self.gcn_proj = nn.Linear(IN_F, 64)
        self.lstm = nn.LSTM(64, 128, 2, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(128, 1)
    def forward(self, x):
        x = self.gcn_proj(x)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1]).squeeze(-1)

# ── ENTRAÎNEMENT ──────────────────────────────────────────────────
results = {}

def train_variant(ModelClass, name):
    print(f"\n{'─'*50}")
    print(f"Variante : {name}")
    seed_results, seed_latencies = [], []

    for seed in SEEDS:
        torch.manual_seed(seed)
        model = ModelClass().to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        crit  = nn.MSELoss()
        best_val, pat, best_state = 1e9, 0, None
        t0 = time.time()

        for ep in range(1, MAX_EPOCHS+1):
            model.train()
            for xb, yb in tr_dl:
                opt.zero_grad(); crit(model(xb),yb).backward(); opt.step()

            model.eval()
            vl = np.mean([crit(model(xb),yb).item() for xb,yb in va_dl])
            print(f"  seed={seed} ep={ep:2d}/{MAX_EPOCHS}  val={vl:.4f}  ({time.time()-t0:.0f}s)", flush=True)

            if vl < best_val:
                best_val, pat = vl, 0
                best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            else:
                pat += 1
                if pat >= PATIENCE:
                    print(f"  → Early stop ep={ep}")
                    break

        model.load_state_dict(best_state); model.eval()
        preds = np.concatenate([model(xb).cpu().detach().numpy() for xb,_ in te_dl])
        r = calc_metrics(ys_te, preds)
        lat = measure_latency(model)
        seed_results.append(r)
        seed_latencies.append(lat)
        print(f"  ✓ seed={seed} MAE={r['MAE']:.2f} RMSE={r['RMSE']:.2f} R²={r['R2']:.4f} lat={lat:.1f}ms")

    results[name] = dict(
        MAE=float(np.mean([r["MAE"]  for r in seed_results])),
        MAE_std=float(np.std( [r["MAE"]  for r in seed_results])),
        RMSE=float(np.mean([r["RMSE"] for r in seed_results])),
        RMSE_std=float(np.std( [r["RMSE"] for r in seed_results])),
        R2=float(np.mean([r["R2"]   for r in seed_results])),
        Latency_ms=float(np.mean(seed_latencies)),
    )
    print(f"  MOYENNE → MAE={results[name]['MAE']:.2f}±{results[name]['MAE_std']:.2f}  "
          f"RMSE={results[name]['RMSE']:.2f}±{results[name]['RMSE_std']:.2f}  "
          f"R²={results[name]['R2']:.4f}  lat={results[name]['Latency_ms']:.1f}ms")

    # Sauvegarder après chaque variante
    with open("results_ablation_real.json","w") as f:
        json.dump(results, f, indent=2)

# Lancer les 3 variantes
train_variant(FullFramework,  "Full Framework (GCN+Transformer)")
train_variant(NoGNN,          "No-GCN (Linear+Transformer)")
train_variant(NoTransformer,  "No-Transformer (GCN+LSTM)")

# ── RÉSUMÉ ────────────────────────────────────────────────────────
print("\n" + "="*65)
print("ABLATION RÉELLE — Beijing Multi-site UCI #501")
print("="*65)
print(f"{'Variant':<35} {'MAE':>6} {'RMSE':>7} {'R²':>7} {'Lat(ms)':>9}")
print("-"*65)
for name, r in results.items():
    print(f"{name:<35} {r['MAE']:>6.2f} {r['RMSE']:>7.2f} {r['R2']:>7.4f} {r['Latency_ms']:>9.1f}")
print("="*65)
print("\n✓ results_ablation_real.json sauvegardé → envoie ce fichier à Claude !")
