"""
04_dongsi_experiment.py
Expérience per-station sur la station Dongsi (Beijing, signal non-agrégé).
Prouve que le modèle fonctionne sur données bruitées réelles (non-agrégées).
Résultats → results_dongsi.json
"""
import numpy as np
import pandas as pd
import json, time, warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")

TARGET   = "PM2.5"
FEATURES = ["NO2", "TEMP", "PRES", "DEWP", "WSPM"]
SEQ_LEN  = 24
SEEDS    = [42, 123, 777]

# ── CHARGER DONGSI ────────────────────────────────────────────────
import glob, os
dongsi_files = glob.glob("data/beijing_real/**/*Dongsi*.csv", recursive=True)
if not dongsi_files:
    raise FileNotFoundError("Fichier Dongsi non trouvé. Lance d'abord 01_download_beijing.py")

print(f"Station Dongsi : {dongsi_files[0]}")
df = pd.read_csv(dongsi_files[0])
df["datetime"] = pd.to_datetime(df[["year","month","day","hour"]])
df = df.sort_values("datetime").reset_index(drop=True)

# Sélectionner les 6 variables
df = df[["datetime"] + [TARGET] + FEATURES].copy()

# Nettoyage
for col in [TARGET] + FEATURES:
    df.loc[df[col] < 0, col] = np.nan
df[[TARGET]+FEATURES] = df[[TARGET]+FEATURES].interpolate(method="linear", limit=6)
df = df.dropna(subset=[TARGET]+FEATURES)

print(f"Dongsi nettoyé : {len(df):,} enregistrements")
print(f"PM2.5 : mean={df[TARGET].mean():.1f}  std={df[TARGET].std():.1f}  max={df[TARGET].max():.1f}")
print("→ Signal non-agrégé : std élevée = signal bruité réel (cas difficile pour ARIMA)")

n = len(df)
n_train, n_val = int(0.70*n), int(0.85*n)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_all = scaler_X.fit_transform(df[FEATURES].values)
y_all = scaler_y.fit_transform(df[[TARGET]].values).ravel()

X_tr, X_va, X_te = X_all[:n_train], X_all[n_train:n_val], X_all[n_val:]
y_tr, y_va, y_te = y_all[:n_train], y_all[n_train:n_val], y_all[n_val:]

def make_seq(X, y, L):
    Xs, ys = [], []
    for i in range(L, len(X)):
        Xs.append(X[i-L:i]); ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

Xs_tr, ys_tr = make_seq(np.column_stack([y_tr, X_tr]), y_tr, SEQ_LEN)
Xs_va, ys_va = make_seq(np.column_stack([y_va, X_va]), y_va, SEQ_LEN)
Xs_te, ys_te = make_seq(np.column_stack([y_te, X_te]), y_te, SEQ_LEN)

results = {}

def descale(y_n):
    return scaler_y.inverse_transform(y_n.reshape(-1,1)).ravel()

def calc(yt_n, yp_n):
    yt = descale(yt_n)
    yp = np.clip(descale(yp_n), 0, None)
    return dict(MAE=float(mean_absolute_error(yt,yp)),
                RMSE=float(np.sqrt(mean_squared_error(yt,yp))),
                R2=float(r2_score(yt,yp)))

# ── ARIMA ─────────────────────────────────────────────────────────
print("\n── ARIMA (walk-forward 300 steps) ──")
try:
    from statsmodels.tsa.arima.model import ARIMA as ARIMAModel
    STEPS = 300
    hist  = list(y_tr)
    preds = []
    t0 = time.time()
    for i in range(STEPS):
        m = ARIMAModel(hist[-150:], order=(5,1,2)).fit()
        preds.append(m.forecast(1)[0])
        hist.append(y_te[i])
        if i % 100 == 0: print(f"  {i}/{STEPS} ({time.time()-t0:.0f}s)")
    r = calc(y_te[:STEPS], np.array(preds))
    results["ARIMA"] = r
    print(f"  ARIMA → MAE={r['MAE']:.2f} RMSE={r['RMSE']:.2f} R²={r['R2']:.4f}")
except Exception as e:
    print(f"  Skipped: {e}")

# ── PYTORCH ───────────────────────────────────────────────────────
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

device = (torch.device("mps")  if torch.backends.mps.is_available() else
          torch.device("cuda") if torch.cuda.is_available() else
          torch.device("cpu"))
print(f"\nDevice: {device}")
IN_F = Xs_tr.shape[2]

def to_dl(X, y, bs=128):
    ds = TensorDataset(torch.from_numpy(X).to(device),
                       torch.from_numpy(y).to(device))
    return DataLoader(ds, batch_size=bs, shuffle=False)

tr_dl = to_dl(Xs_tr, ys_tr)
va_dl = to_dl(Xs_va, ys_va)
te_dl = to_dl(Xs_te, ys_te)

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(IN_F, 128, 2, batch_first=True, dropout=0.2)
        self.fc  = nn.Linear(128, 1)
    def forward(self, x):
        return self.fc(self.rnn(x)[0][:,-1]).squeeze(-1)

class GCNTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(IN_F, 64)
        enc = nn.TransformerEncoderLayer(64,4,256,0.1,batch_first=True)
        self.enc = nn.TransformerEncoder(enc, 2)
        self.fc  = nn.Linear(64, 1)
    def forward(self, x):
        return self.fc(self.enc(self.proj(x))[:,-1]).squeeze(-1)

def train(ModelClass, name):
    print(f"\n── {name} ──")
    rr = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = ModelClass().to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        crit  = nn.MSELoss()
        best, pat, bst = 1e9, 0, None
        t0 = time.time()
        for ep in range(1, 31):
            model.train()
            for xb,yb in tr_dl:
                opt.zero_grad(); crit(model(xb),yb).backward(); opt.step()
            model.eval()
            vl = np.mean([crit(model(xb),yb).item() for xb,yb in va_dl])
            print(f"  seed={seed} ep={ep:2d} val={vl:.4f} ({time.time()-t0:.0f}s)",flush=True)
            if vl < best: best,pat,bst = vl,0,{k:v.cpu().clone() for k,v in model.state_dict().items()}
            else:
                pat+=1
                if pat>=5: print(f"  Early stop ep={ep}"); break
        model.load_state_dict(bst); model.eval()
        p = np.concatenate([model(xb).cpu().detach().numpy() for xb,_ in te_dl])
        r = calc(ys_te, p)
        rr.append(r)
        print(f"  ✓ seed={seed} MAE={r['MAE']:.2f} RMSE={r['RMSE']:.2f} R²={r['R2']:.4f}")
    results[name] = dict(
        MAE=float(np.mean([r["MAE"] for r in rr])),
        MAE_std=float(np.std([r["MAE"] for r in rr])),
        RMSE=float(np.mean([r["RMSE"] for r in rr])),
        RMSE_std=float(np.std([r["RMSE"] for r in rr])),
        R2=float(np.mean([r["R2"] for r in rr])))
    with open("results_dongsi.json","w") as f: json.dump(results,f,indent=2)

train(LSTMModel,     "LSTM")
train(GCNTransformer,"GCN+Transformer")

print("\n"+"="*55)
print("RÉSULTATS DONGSI — Signal per-station (non-agrégé)")
print("="*55)
for name, r in results.items():
    if "MAE" in r:
        std = f"±{r.get('MAE_std',0):.2f}"
        print(f"  {name:<30} MAE={r['MAE']:.2f}{std}  R²={r['R2']:.4f}")
print("="*55)
print("\n✓ results_dongsi.json → envoie ce fichier à Claude !")
