"""
02_train_all_models.py — Entraîne tous les modèles sur le vrai dataset Beijing
Résultats → results_real.json (envoie ce fichier à Claude pour maj du docx)
"""
import numpy as np
import pandas as pd
import json, time, warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────
DATA_PATH  = "data/beijing_real_combined.csv"
TARGET     = "PM2.5"
FEATURES   = ["NO2", "TEMP", "PRES", "DEWP", "WSPM"]
SEQ_LEN    = 24
BATCH_SIZE = 64
MAX_EPOCHS = 50
PATIENCE   = 7
SEEDS      = [42, 123, 777]

# ── CHARGEMENT ────────────────────────────────────────────────────
print("Chargement données réelles...")
df = pd.read_csv(DATA_PATH, parse_dates=["datetime"]).sort_values("datetime").reset_index(drop=True)
n = len(df)
n_train, n_val = int(0.70*n), int(0.85*n)
print(f"{n:,} enregistrements | train={n_train:,} val={n_val-n_train:,} test={n-n_val:,}")

# Normalisation
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_all = scaler_X.fit_transform(df[FEATURES].values)
y_all = scaler_y.fit_transform(df[[TARGET]].values).ravel()

X_train, X_val, X_test = X_all[:n_train], X_all[n_train:n_val], X_all[n_val:]
y_train, y_val, y_test = y_all[:n_train], y_all[n_train:n_val], y_all[n_val:]

def make_seq(X, y, seq_len):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i-seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

Xs_tr, ys_tr = make_seq(np.column_stack([y_train, X_train]), y_train, SEQ_LEN)
Xs_va, ys_va = make_seq(np.column_stack([y_val,   X_val]),   y_val,   SEQ_LEN)
Xs_te, ys_te = make_seq(np.column_stack([y_test,  X_test]),  y_test,  SEQ_LEN)
print(f"Séquences → train:{Xs_tr.shape} test:{Xs_te.shape}")

def metrics(yt_n, yp_n):
    yt = scaler_y.inverse_transform(yt_n.reshape(-1,1)).ravel()
    yp = np.clip(scaler_y.inverse_transform(yp_n.reshape(-1,1)).ravel(), 0, None)
    return dict(MAE=float(mean_absolute_error(yt,yp)),
                RMSE=float(np.sqrt(mean_squared_error(yt,yp))),
                R2=float(r2_score(yt,yp)))

results = {}

# ── ARIMA ─────────────────────────────────────────────────────────
print("\n── ARIMA (walk-forward 500 steps) ──")
try:
    from statsmodels.tsa.arima.model import ARIMA
    STEPS = 500
    hist  = list(y_train)
    preds = []
    t0    = time.time()
    for i in range(STEPS):
        m = ARIMA(hist[-200:], order=(5,1,2)).fit()
        preds.append(m.forecast(1)[0])
        hist.append(y_test[i])
        if i % 100 == 0: print(f"  {i}/{STEPS} ({time.time()-t0:.0f}s)")
    r = metrics(y_test[:STEPS], np.array(preds))
    results["ARIMA"] = r
    print(f"  MAE={r['MAE']:.2f} RMSE={r['RMSE']:.2f} R²={r['R2']:.4f}")
except Exception as e:
    print(f"  Skipped: {e}")
    results["ARIMA"] = {"note": str(e)}

# ── XGBOOST ───────────────────────────────────────────────────────
print("\n── XGBoost ──")
try:
    from xgboost import XGBRegressor
    rr = []
    Xtr_f = np.column_stack([y_train[:-1], X_train[1:]])
    ytr_f = y_train[1:]
    Xva_f = np.column_stack([y_val[:-1],   X_val[1:]])
    Xte_f = np.column_stack([y_test[:-1],  X_test[1:]])
    for s in SEEDS:
        m = XGBRegressor(n_estimators=500,max_depth=6,learning_rate=0.05,
                         subsample=0.8,colsample_bytree=0.8,
                         early_stopping_rounds=20,eval_metric="rmse",
                         random_state=s,verbosity=0)
        m.fit(Xtr_f,ytr_f,eval_set=[(Xva_f,y_val[1:])],verbose=False)
        rr.append(metrics(y_test[1:], m.predict(Xte_f)))
        print(f"  seed={s} MAE={rr[-1]['MAE']:.2f} RMSE={rr[-1]['RMSE']:.2f} R²={rr[-1]['R2']:.4f}")
    results["XGBoost"] = dict(
        MAE=float(np.mean([r["MAE"] for r in rr])),  MAE_std=float(np.std([r["MAE"] for r in rr])),
        RMSE=float(np.mean([r["RMSE"] for r in rr])),RMSE_std=float(np.std([r["RMSE"] for r in rr])),
        R2=float(np.mean([r["R2"] for r in rr])))
except Exception as e:
    print(f"  Skipped: {e}")
    results["XGBoost"] = {"note": str(e)}

# ── PYTORCH MODELS ────────────────────────────────────────────────
print("\n── Modèles PyTorch ──")
try:
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    device = (torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available()          else
              torch.device("cpu"))
    print(f"  Device: {device}")

    def tens(*a): return [torch.tensor(x,dtype=torch.float32).to(device) for x in a]
    Xt,yt = tens(Xs_tr,ys_tr); Xv,yv = tens(Xs_va,ys_va); Xe,ye = tens(Xs_te,ys_te)
    tr_dl = DataLoader(TensorDataset(Xt,yt), batch_size=BATCH_SIZE, shuffle=False)
    va_dl = DataLoader(TensorDataset(Xv,yv), batch_size=BATCH_SIZE)
    te_dl = DataLoader(TensorDataset(Xe,ye), batch_size=BATCH_SIZE)
    IN_F  = Xs_tr.shape[2]

    class LSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.LSTM(IN_F,128,2,batch_first=True,dropout=0.2)
            self.fc  = nn.Linear(128,1)
        def forward(self,x): return self.fc(self.rnn(x)[0][:,-1]).squeeze(-1)

    class CNNLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = nn.Sequential(nn.Conv1d(IN_F,64,3,padding=1),nn.ReLU(),
                                     nn.Conv1d(64,64,3,padding=1),nn.ReLU())
            self.rnn = nn.LSTM(64,128,2,batch_first=True,dropout=0.2)
            self.fc  = nn.Linear(128,1)
        def forward(self,x):
            x=self.cnn(x.permute(0,2,1)).permute(0,2,1)
            return self.fc(self.rnn(x)[0][:,-1]).squeeze(-1)

    class GCNTrans(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(IN_F,64)
            enc = nn.TransformerEncoderLayer(64,4,256,0.1,batch_first=True)
            self.enc = nn.TransformerEncoder(enc,2)
            self.fc  = nn.Linear(64,1)
        def forward(self,x): return self.fc(self.enc(self.proj(x))[:,-1]).squeeze(-1)

    def run(ModelClass, name):
        rr = []
        for s in SEEDS:
            torch.manual_seed(s)
            m   = ModelClass().to(device)
            opt = torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-5)
            crit= nn.MSELoss()
            best, pat, bst = 1e9, 0, None
            for ep in range(MAX_EPOCHS):
                m.train()
                for xb,yb in tr_dl:
                    opt.zero_grad(); crit(m(xb),yb).backward(); opt.step()
                m.eval()
                vl = np.mean([crit(m(xb),yb).item() for xb,yb in va_dl])
                if vl < best:
                    best,pat = vl,0
                    bst = {k:v.cpu().clone() for k,v in m.state_dict().items()}
                else:
                    pat += 1
                    if pat >= PATIENCE: break
            m.load_state_dict(bst); m.eval()
            p = np.concatenate([m(xb).cpu().detach().numpy() for xb,_ in te_dl])
            r = metrics(ys_te, p)
            rr.append(r)
            print(f"  {name} seed={s} ep={ep+1} MAE={r['MAE']:.2f} RMSE={r['RMSE']:.2f} R²={r['R2']:.4f}")
        results[name] = dict(
            MAE=float(np.mean([r["MAE"] for r in rr])),  MAE_std=float(np.std([r["MAE"] for r in rr])),
            RMSE=float(np.mean([r["RMSE"] for r in rr])),RMSE_std=float(np.std([r["RMSE"] for r in rr])),
            R2=float(np.mean([r["R2"] for r in rr])))

    run(LSTM,    "LSTM")
    run(CNNLSTM, "CNN-LSTM")
    run(GCNTrans,"GCN+Transformer")

except Exception as e:
    print(f"  PyTorch erreur: {e}")

# ── RÉSUMÉ ────────────────────────────────────────────────────────
print("\n" + "="*55)
print("RÉSULTATS FINAUX — Beijing réel")
print("="*55)
for name, r in results.items():
    if "MAE" in r and r["MAE"] is not None:
        print(f"  {name:<22} MAE={r['MAE']:.2f}±{r.get('MAE_std',0):.2f}"
              f"  RMSE={r['RMSE']:.2f}±{r.get('RMSE_std',0):.2f}"
              f"  R²={r['R2']:.4f}")

with open("results_real.json","w") as f: json.dump(results,f,indent=2)
print("\n✓ results_real.json sauvegardé → envoie ce fichier à Claude !")
