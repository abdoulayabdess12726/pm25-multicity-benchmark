#!/usr/bin/env python3
"""
10_external_baselines.py  —  Expérience E1 (baselines externes)
==========================================================================
ARIMA / XGBoost / LSTM sous le PROTOCOLE IDENTIQUE à 06_train_multistation.py :
  - splits chronologiques 70/15/15 (reuse split_and_scale de 06)
  - mêmes 5 features, horizon 1 h (cible = point t+1)
  - fenêtres SEQ_LEN=24 -> cibles test = test[24:] (comme MultiStationDataset)
  - métriques RMSE/MAE/R² per-station ET agrégées (R² global sur toutes les
    stations x temps, dénormalisé, exactement comme evaluate() de 06)
  - seeds 42/123/777 pour le LSTM ; XGBoost seed fixe 42 ; ARIMA déterministe

Baselines :
  - ARIMA (statsmodels, per-station, PM2.5 univarié, 1-step-ahead sans refit)
  - XGBoost (per-station : lags 1..24 de PM2.5 + 4 covariables météo à t-1)
  - LSTM (2 couches, hidden 64, temporel poolé par nœud, early-stopping de 06)

Sortie : results/external_baselines.csv (ARIMA/XGBoost/LSTM, per-station + agrégat)
         + tableau markdown par ville (ARIMA/XGBoost/LSTM/Linear-Tr/GCN-Tr).

Usage :
    python 10_external_baselines.py --cities madrid            # validation
    python 10_external_baselines.py --cities beijing london    # après feu vert
"""
import argparse
import importlib.util
import io
import json
import random
import sys
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import xgboost  # noqa: F401,E402 — DOIT être importé AVANT torch (conflit libomp macOS -> segfault)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# XGBoost (libomp) + torch coexistent : sans ceci, les ops torch se bloquent
# (deadlock OpenMP) après un fit XGBoost. 1 thread torch CPU suffit (le LSTM
# tourne sur MPS ; le dispatch reste rapide).
torch.set_num_threads(1)

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
SEQ_LEN = 24
SEEDS = [42, 123, 777]
ARIMA_ORDER = (2, 1, 2)          # ordre fixe per-station (fallback plus bas)
CSV = ROOT / "results" / "external_baselines.csv"


def load_bench():
    spec = importlib.util.spec_from_file_location(
        "bench", str(ROOT / "06_train_multistation.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench"] = mod
    spec.loader.exec_module(mod)
    return mod


def get_city(b, city):
    """Charge la ville, renvoie splits bruts + scaler + méta, alignés sur 06."""
    with redirect_stdout(io.StringIO()):
        if city == "beijing":
            ret = b.load_beijing_data(
                str(ROOT / "data/beijing_real/PRSA_Data_20130301-20170228"))
        elif city == "london":
            ret = b.load_london_data()
        else:
            ret = b.load_madrid_data()
        data = ret[0] if isinstance(ret, (tuple, list)) else ret
        data = np.asarray(data, dtype=np.float32)
        _, _, _, scaler = b.split_and_scale(data)

    names = list(b.STATION_NAMES)
    feats = list(b.FEATURES)
    pm = feats.index("PM2.5")
    wx = [i for i in range(len(feats)) if i != pm]     # 4 covariables météo
    T = len(data)
    t1, t2 = int(0.70 * T), int(0.85 * T)
    raw = dict(train=data[:t1], val=data[t1:t2], test=data[t2:])
    return dict(city=city, names=names, feats=feats, pm=pm, wx=wx,
                raw=raw, scaler=scaler)


# --------------------------------------------------------------------------- #
#  Métriques (per-station + agrégat global, dénormalisé) — identiques à 06
# --------------------------------------------------------------------------- #
def metrics_rows(city, model, seed, names, Y, P):
    """Y, P : (n_targets, n_stations) dénormalisés. Renvoie lignes tidy."""
    rows = []
    for i, name in enumerate(names):
        yt, pt = Y[:, i], P[:, i]
        rows.append(dict(city=city, model=model, station=name, seed=seed,
                         MAE=mean_absolute_error(yt, pt),
                         RMSE=float(np.sqrt(mean_squared_error(yt, pt))),
                         R2=r2_score(yt, pt)))
    yf, pf = Y.reshape(-1), P.reshape(-1)                # agrégat global (comme evaluate)
    rows.append(dict(city=city, model=model, station="__aggregate__", seed=seed,
                     MAE=mean_absolute_error(yf, pf),
                     RMSE=float(np.sqrt(mean_squared_error(yf, pf))),
                     R2=r2_score(yf, pf)))
    return rows


def true_targets(c):
    """Cibles test dénormalisées, alignées SEQ_LEN:  (n, n_stations)."""
    return c["raw"]["test"][SEQ_LEN:, :, c["pm"]]


# --------------------------------------------------------------------------- #
#  ARIMA (per-station, univarié, one-step-ahead sans refit)
# --------------------------------------------------------------------------- #
def run_arima(c):
    from statsmodels.tsa.arima.model import ARIMA
    pm = c["pm"]
    P = np.zeros((c["raw"]["test"].shape[0] - SEQ_LEN, len(c["names"])))
    for i in range(len(c["names"])):
        tr = c["raw"]["train"][:, i, pm].astype(float)
        ext = np.concatenate([c["raw"]["val"][:, i, pm],
                              c["raw"]["test"][:, i, pm]]).astype(float)
        pred = None
        for order in (ARIMA_ORDER, (1, 1, 1), (1, 1, 0), (0, 1, 1)):
            try:
                res = ARIMA(tr, order=order).fit()
                res2 = res.append(ext, refit=False)
                start = len(tr) + len(c["raw"]["val"])
                pred = np.asarray(res2.predict(start=start,
                                               end=start + len(c["raw"]["test"]) - 1))
                break
            except Exception:
                continue
        if pred is None:                                  # dernier recours : persistance
            pred = c["raw"]["test"][:, i, pm].astype(float)
            pred[1:] = pred[:-1]
        P[:, i] = pred[SEQ_LEN:]
    return metrics_rows(c["city"], "ARIMA", "-", c["names"], true_targets(c), P)


# --------------------------------------------------------------------------- #
#  XGBoost (per-station : lags 1..24 PM2.5 + 4 météo à t-1), seed 42
# --------------------------------------------------------------------------- #
def _xgb_design(arr, pm, wx):
    """arr (T,N,F) -> par station : X (T-24, 24+4), y (T-24,) pour cibles t>=24."""
    T = arr.shape[0]
    out = {}
    for i in range(arr.shape[1]):
        pm_series = arr[:, i, pm]
        X, y = [], []
        for t in range(SEQ_LEN, T):
            lags = pm_series[t - SEQ_LEN:t][::-1]           # PM2.5[t-1..t-24]
            wxt = arr[t - 1, i, wx]                         # météo à t-1
            X.append(np.concatenate([lags, wxt]))
            y.append(pm_series[t])
        out[i] = (np.asarray(X, dtype=float), np.asarray(y, dtype=float))
    return out


def run_xgboost(c):
    from xgboost import XGBRegressor
    pm, wx = c["pm"], c["wx"]
    tr = _xgb_design(c["raw"]["train"], pm, wx)
    te = _xgb_design(c["raw"]["test"], pm, wx)
    Y = true_targets(c)
    P = np.zeros_like(Y, dtype=float)
    for i in range(len(c["names"])):
        Xtr, ytr = tr[i]
        Xte, _ = te[i]
        m = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=42,
                         n_jobs=4, objective="reg:squarederror")
        m.fit(Xtr, ytr)
        P[:, i] = m.predict(Xte)
    return metrics_rows(c["city"], "XGBoost", 42, c["names"], Y, P)


# --------------------------------------------------------------------------- #
#  LSTM (2 couches, hidden 64, temporel poolé par nœud) — early stop de 06
# --------------------------------------------------------------------------- #
class LSTMReg(nn.Module):
    """LSTM 2 couches (hidden 64) avec SKIP DE PERSISTANCE : le réseau prédit la
    correction par rapport à la dernière valeur observée (pred = PM2.5[t-1] +
    LSTM(...)). Paramétrisation résiduelle standard en prévision : sans elle, un
    LSTM vanilla régresse vers la moyenne et sous-performe la persistance triviale
    (R² 0.17 vs 0.80 sur Madrid) — voir note dans le rapport."""
    def __init__(self, in_dim, pm_idx, hidden=64, layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, 1)
        self.pm_idx = pm_idx

    def forward(self, x):                                  # (B, SEQ, F)
        out, _ = self.lstm(x)
        return x[:, -1, self.pm_idx] + self.head(out[:, -1, :]).squeeze(-1)


def _windows(arr, pm):
    X, Y = [], []
    for j in range(SEQ_LEN, len(arr)):
        X.append(arr[j - SEQ_LEN:j]); Y.append(arr[j, :, pm])
    return np.asarray(X, np.float32), np.asarray(Y, np.float32)   # (n,SEQ,N,F),(n,N)


def _predict_batched(model, X_t, device, bs=2048):
    """Forward par blocs (MPS segfault sur de très grands batchs LSTM)."""
    outs = []
    with torch.no_grad():
        for k in range(0, X_t.shape[0], bs):
            outs.append(model(X_t[k:k + bs].to(device)).cpu())
    return torch.cat(outs)


def run_lstm(c, device):
    b_split = _split_scaled(c)                             # scaled splits alignés 06
    pm = c["pm"]
    dmin = c["scaler"].data_min_[pm]; dmax = c["scaler"].data_max_[pm]
    Xtr, ytr = _windows(b_split["train"], pm)
    Xva, yva = _windows(b_split["val"], pm)
    Xte, yte = _windows(b_split["test"], pm)
    N, F = Xtr.shape[2], Xtr.shape[3]

    def flat(X, Y):                                        # pool nœuds -> (n*N,SEQ,F),(n*N,)
        n = X.shape[0]
        return (torch.tensor(X.transpose(0, 2, 1, 3).reshape(n * N, SEQ_LEN, F)),
                torch.tensor(Y.reshape(n * N)))
    Xtr_t, ytr_t = flat(Xtr, ytr)
    Xva_t, yva_t = flat(Xva, yva)
    Y_true = true_targets(c)                               # (n_test, N) dénorm

    rows = []
    for seed in SEEDS:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        model = LSTMReg(F, pm).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        crit = nn.MSELoss()
        best, best_state, bad = np.inf, None, 0
        ntr = Xtr_t.shape[0]
        for epoch in range(1, 51):
            model.train()
            perm = torch.randperm(ntr)
            for k in range(0, ntr, 64):
                idx = perm[k:k + 64]
                xb = Xtr_t[idx].to(device); yb = ytr_t[idx].to(device)
                opt.zero_grad()
                loss = crit(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.eval()
            vpred = _predict_batched(model, Xva_t, device)
            vl = crit(vpred, yva_t).item()
            if vl < best:
                best, best_state, bad = vl, {k: v.cpu().clone()
                                             for k, v in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 8:
                    break
        model.load_state_dict(best_state)
        model.eval()
        # prédiction test, nœud par nœud
        nte = Xte.shape[0]
        Xte_t = torch.tensor(Xte.transpose(0, 2, 1, 3).reshape(nte * N, SEQ_LEN, F))
        pred = _predict_batched(model, Xte_t, device).numpy().reshape(nte, N)
        P = pred * (dmax - dmin) + dmin                    # dénormalisation PM2.5
        rows += metrics_rows(c["city"], "LSTM", seed, c["names"], Y_true, P)
    return rows


def _split_scaled(c):
    """Rejoue split_and_scale pour obtenir les splits SCALED (pour le LSTM)."""
    b = sys.modules["bench"]
    data = np.concatenate([c["raw"]["train"], c["raw"]["val"], c["raw"]["test"]], 0)
    with redirect_stdout(io.StringIO()):
        tr, va, te, _ = b.split_and_scale(data)
    return dict(train=tr, val=va, test=te)


# --------------------------------------------------------------------------- #
#  Table markdown (ARIMA/XGBoost/LSTM/Linear-Tr/GCN-Tr)
# --------------------------------------------------------------------------- #
def agg_from_rows(df, city, model):
    sub = df[(df.city == city) & (df.model == model) & (df.station == "__aggregate__")]
    if sub.empty:
        return None
    return {m: (sub[m].mean(), sub[m].std(ddof=0) if len(sub) > 1 else 0.0)
            for m in ("MAE", "RMSE", "R2")}


def agg_from_json(city, model_key):
    p = ROOT / f"results/{city}/multistation_results.json"
    if not p.exists():
        return None
    g = json.loads(p.read_text())["graphs"]
    topo = "distance" if "distance" in g else list(g)[0]
    d = g[topo].get(model_key)
    if not d:
        return None
    return {m: (float(np.mean(d[m])), float(np.std(d[m]))) for m in ("MAE", "RMSE", "R2")}


def markdown_table(city, df):
    def cell(a, m):
        return "—" if a is None else f"{a[m][0]:.3f} ± {a[m][1]:.3f}"
    order = [("ARIMA", lambda: agg_from_rows(df, city, "ARIMA")),
             ("XGBoost", lambda: agg_from_rows(df, city, "XGBoost")),
             ("LSTM", lambda: agg_from_rows(df, city, "LSTM")),
             ("Linear-Transformer", lambda: agg_from_json(city, "Linear+Transformer")),
             ("GCN-Transformer", lambda: agg_from_json(city, "GCN+Transformer"))]
    lines = [f"\n### {city.capitalize()} — agrégat (test, dénormalisé, moyenne ± SD)",
             "| Model | MAE | RMSE | R² |", "|---|---|---|---|"]
    for name, fn in order:
        a = fn()
        lines.append(f"| {name} | {cell(a,'MAE')} | {cell(a,'RMSE')} | {cell(a,'R2')} |")
    lines.append("\n_GCN-Transformer: topologie distance. Linear-Transformer: "
                 "temporel pur (identique aux 2 topologies)._")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", required=True,
                    choices=["beijing", "london", "madrid"])
    args = ap.parse_args()
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    b = load_bench()

    all_rows = []
    for city in args.cities:
        print(f"\n[{city}] baselines externes (device={device})...", file=sys.stderr)
        c = get_city(b, city)
        print(f"  ARIMA...", file=sys.stderr);   all_rows += run_arima(c)
        print(f"  XGBoost...", file=sys.stderr); all_rows += run_xgboost(c)
        print(f"  LSTM (3 seeds)...", file=sys.stderr); all_rows += run_lstm(c, device)

    new = pd.DataFrame(all_rows, columns=["city", "model", "station", "seed",
                                          "MAE", "RMSE", "R2"])
    # append-merge : remplace uniquement les villes recalculées
    if CSV.exists():
        old = pd.read_csv(CSV)
        old = old[~old.city.isin(args.cities)]
        full = pd.concat([old, new], ignore_index=True)
    else:
        full = new
    full.to_csv(CSV, index=False)
    print(f"\nCSV : {CSV}  ({len(full)} lignes ; +{len(new)} cette exécution)")

    for city in args.cities:
        print(markdown_table(city, new))


if __name__ == "__main__":
    main()
