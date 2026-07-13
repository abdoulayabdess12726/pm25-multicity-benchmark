#!/usr/bin/env python3
"""
oversmoothing_gat_control.py
============================================================
Reviewer-rebuttal experiment for the multi-city PM2.5 benchmark
(Badouch & Belhoucine, "Graph Encoding for PM2.5 Forecasting...").

Produces the two pieces of evidence an IoT/MDPI reviewer will ask for:

  (1) OVER-SMOOTHING CONTROL
      - 1-layer GCN-Transformer  vs  2-layer GCN-Transformer  vs  Linear-Transformer
      - layer-wise Dirichlet energy of GCN node embeddings (lower = more smoothed)
      If the 1-layer GCN STILL underperforms Linear in London/Madrid, the result
      is NOT a pure over-smoothing artifact -> rebuttal complete.

  (2) GAT BASELINE
      - GAT-Transformer (graph attention) vs Linear-Transformer
      Shows whether the failure is specific to GCN aggregation or generalizes,
      neutralizing the "single architecture" objection.

It reuses the SAME protocol as the paper: 3 seeds (42,123,777), both topologies
(distance k-NN, correlation k-NN), chronological 70/15/15, MinMax on train,
Adam 1e-3 / wd 1e-5, batch 64, seq 24h, MSE, early stop patience 8, max 50 epochs.

REQUIREMENTS
    pip install torch torch_geometric numpy pandas scipy scikit-learn

WIRE YOUR DATA  ->  edit load_city() below (one TODO marked).
RUN
    python oversmoothing_gat_control.py --city beijing --topology distance
    python oversmoothing_gat_control.py --city london  --topology distance
    python oversmoothing_gat_control.py --city madrid  --topology distance
    # repeat with --topology correlation
Outputs a results table per run; aggregate the printed mean +/- std into Table VI / VI.B.
============================================================
"""

import argparse, math, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv
from scipy.stats import wilcoxon
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

FEATURES = ["PM2.5", "TEMP", "PRES", "DEWP", "WSPM"]   # harmonized 5-feature set
TARGET   = "PM2.5"
SEQ_LEN  = 24
SEEDS    = [42, 123, 777]
K        = 5


# --------------------------------------------------------------------------- #
# 1. DATA
# --------------------------------------------------------------------------- #
BEIJING_DIR = "data/beijing_real/PRSA_Data_20130301-20170228"   # override via --data_dir
_BENCH = None


def _bench():
    """Import the user's 06_train_multistation.py once (filename starts with a
    digit, so importlib is required). Reuses their exact loaders + station
    coords so this control is consistent with the canonical benchmark."""
    global _BENCH
    if _BENCH is not None:
        return _BENCH
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("bench", "06_train_multistation.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench"] = mod
    spec.loader.exec_module(mod)            # safe: their executable code is under __main__
    _BENCH = mod
    return mod


def load_city(city):
    """Return data (T, n_stations, n_features) and coords (n_stations, 2),
    using the project's own loaders and FEATURES ordering."""
    global FEATURES
    b = _bench()
    if city == "beijing":
        ret = b.load_beijing_data(BEIJING_DIR)
    elif city == "london":
        ret = b.load_london_data()
    elif city == "madrid":
        fn = getattr(b, "load_madrid_data", None)
        if fn is None:
            raise RuntimeError("load_madrid_data not found in 06_train_multistation.py")
        ret = fn()
    else:
        raise ValueError(city)

    # their loaders return the data array (possibly inside a tuple) and set globals
    data = ret[0] if isinstance(ret, (tuple, list)) else ret
    data = np.asarray(data, dtype=np.float32)

    # align FEATURES / target ordering to theirs (PM2.5 must be column 0)
    FEATURES = list(getattr(b, "FEATURES", FEATURES))
    if FEATURES[0] != "PM2.5" and "PM2.5" in FEATURES:
        j = FEATURES.index("PM2.5")
        order = [j] + [i for i in range(len(FEATURES)) if i != j]
        data = data[:, :, order]
        FEATURES = [FEATURES[i] for i in order]

    # station coordinates (dict name->(lat,lon) for Beijing, array for LAQN/OpenAQ)
    coords_raw = b.STATION_COORDS
    names = getattr(b, "STATION_NAMES", None)
    if isinstance(coords_raw, dict):
        coords = (np.array([coords_raw[n] for n in names], dtype=float)
                  if names else np.array(list(coords_raw.values()), dtype=float))
    else:
        coords = np.asarray(coords_raw, dtype=float)

    assert coords.shape[0] == data.shape[1], \
        f"coords {coords.shape} vs data stations {data.shape[1]} mismatch"
    return data, coords


def haversine(a, b):
    R = 6371.0
    la1, lo1, la2, lo2 = map(math.radians, [a[0], a[1], b[0], b[1]])
    d = (math.sin((la2-la1)/2)**2 +
         math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2)
    return 2*R*math.asin(math.sqrt(d))


def build_edges(coords, pm_train, topology, k=K):
    """Return edge_index (2,E) and edge_weight (E,) for the chosen topology."""
    n = coords.shape[0]
    src, dst, w = [], [], []
    if topology == "distance":
        D = np.array([[haversine(coords[i], coords[j]) for j in range(n)]
                      for i in range(n)])
        sim = 1.0 / (D + 1e-6); np.fill_diagonal(sim, 0)
    else:  # correlation, computed on TRAIN split only (no leakage)
        sim = np.corrcoef(pm_train.T); np.fill_diagonal(sim, 0)
        sim = np.clip(sim, 0, None)
    for i in range(n):
        nbrs = np.argsort(-sim[i])[:k]
        for j in nbrs:
            src.append(i); dst.append(int(j)); w.append(float(sim[i, j]))
    w = np.array(w); w = (w - w.min()) / (w.max() - w.min() + 1e-9)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(w, dtype=torch.float32)
    return edge_index, edge_weight


# --------------------------------------------------------------------------- #
# 2. MODELS  (shared Transformer temporal backbone)
# --------------------------------------------------------------------------- #
def positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    pos = torch.arange(0, seq_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
    pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)


class TemporalTransformer(nn.Module):
    def __init__(self, d_model=64, heads=4, layers=2, dropout=0.1):
        super().__init__()
        enc = nn.TransformerEncoderLayer(d_model, heads, d_model*2,
                                         dropout, batch_first=True)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.register_buffer("pe", positional_encoding(SEQ_LEN, d_model))
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):                 # x: (B, SEQ, d_model)
        x = x + self.pe
        x = self.tr(x)
        return self.head(x[:, -1, :]).squeeze(-1)


class SpatialEncoder(nn.Module):
    """kind in {'linear','gcn','gat'}; n_layers controls GCN depth (1 or 2)."""
    def __init__(self, in_dim, d_model, kind, n_layers=2, heads=4):
        super().__init__()
        self.kind, self.n_layers = kind, n_layers
        if kind == "linear":
            self.lin = nn.Linear(in_dim, d_model)
        elif kind == "gcn":
            self.g1 = GCNConv(in_dim, d_model)
            self.g2 = GCNConv(d_model, d_model) if n_layers == 2 else None
        elif kind == "gat":
            self.a1 = GATConv(in_dim, d_model // heads, heads=heads)
            self.a2 = GATConv(d_model, d_model // heads, heads=heads) if n_layers == 2 else None
        self.relu = nn.ReLU()
        self._dirichlet = []             # filled during forward for logging

    def forward(self, x, edge_index, edge_weight):
        # x: (n_nodes, in_dim) for one timestep
        self._dirichlet = []
        if self.kind == "linear":
            return self.relu(self.lin(x))
        if self.kind == "gcn":
            h = self.relu(self.g1(x, edge_index, edge_weight))
            self._dirichlet.append(dirichlet_energy(h, edge_index, edge_weight))
            if self.g2 is not None:
                h = self.relu(self.g2(h, edge_index, edge_weight))
                self._dirichlet.append(dirichlet_energy(h, edge_index, edge_weight))
            return h
        if self.kind == "gat":
            h = self.relu(self.a1(x, edge_index))
            self._dirichlet.append(dirichlet_energy(h, edge_index, edge_weight))
            if self.a2 is not None:
                h = self.relu(self.a2(h, edge_index))
                self._dirichlet.append(dirichlet_energy(h, edge_index, edge_weight))
            return h


def dirichlet_energy(h, edge_index, edge_weight):
    """E = 0.5 * sum_(i,j) w_ij ||h_i - h_j||^2 ; lower => more over-smoothed."""
    src, dst = edge_index
    diff = h[src] - h[dst]
    e = 0.5 * (edge_weight * (diff * diff).sum(dim=1)).sum()
    return float(e.detach().cpu())


class Model(nn.Module):
    def __init__(self, in_dim, kind, n_layers=2, d_model=64):
        super().__init__()
        self.spatial = SpatialEncoder(in_dim, d_model, kind, n_layers)
        self.temporal = TemporalTransformer(d_model)

    def forward(self, x_seq, edge_index, edge_weight):
        # x_seq: (B, SEQ, n_nodes, F) -> encode each (t) spatially, predict per node
        B, S, N, F = x_seq.shape
        enc = []
        for t in range(S):
            xt = x_seq[:, t].reshape(B*N, F)
            ei = edge_index; ew = edge_weight
            # batch the graph by offsetting node indices
            ei_b = torch.cat([edge_index + n*N for n in range(B)], dim=1)
            ew_b = edge_weight.repeat(B)
            h = self.spatial(xt, ei_b, ew_b)        # (B*N, d_model)
            enc.append(h.reshape(B, N, -1))
        enc = torch.stack(enc, dim=1)               # (B, S, N, d_model)
        # predict each node independently from its temporal sequence
        out = []
        for n in range(N):
            out.append(self.temporal(enc[:, :, n, :]))   # (B,)
        return torch.stack(out, dim=1)              # (B, N)


# --------------------------------------------------------------------------- #
# 3. TRAIN / EVAL
# --------------------------------------------------------------------------- #
def make_windows(data):                              # (T,N,F) -> X,(y)
    X, y = [], []
    for t in range(SEQ_LEN, data.shape[0]-1):
        X.append(data[t-SEQ_LEN:t]); y.append(data[t+1, :, 0])   # PM2.5 target
    return np.stack(X), np.stack(y)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def run(city, topology, kind, n_layers, seed, device):
    set_seed(seed)
    data, coords = load_city(city)
    T = data.shape[0]; tr, va = int(0.70*T), int(0.85*T)
    # MinMax fit on train only
    sc = MinMaxScaler().fit(data[:tr].reshape(-1, data.shape[2]))
    data_n = sc.transform(data.reshape(-1, data.shape[2])).reshape(data.shape)
    pm_train = data[:tr, :, 0]
    edge_index, edge_weight = build_edges(coords, pm_train, topology)
    edge_index, edge_weight = edge_index.to(device), edge_weight.to(device)

    X, y = make_windows(data_n)
    n_train = tr - SEQ_LEN
    Xtr, ytr = X[:n_train], y[:n_train]
    Xva, yva = X[n_train:va-SEQ_LEN], y[n_train:va-SEQ_LEN]
    Xte, yte = X[va-SEQ_LEN:], y[va-SEQ_LEN:]

    def to_t(a): return torch.tensor(a, dtype=torch.float32, device=device)
    Xtr, ytr, Xva, yva, Xte, yte = map(to_t, [Xtr, ytr, Xva, yva, Xte, yte])

    model = Model(data.shape[2], kind, n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    best, best_state, patience = 1e9, None, 0
    for epoch in range(50):
        model.train()
        perm = torch.randperm(Xtr.shape[0])
        for i in range(0, Xtr.shape[0], 64):
            idx = perm[i:i+64]
            opt.zero_grad()
            pred = model(Xtr[idx], edge_index, edge_weight)
            loss = loss_fn(pred, ytr[idx]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(Xva, edge_index, edge_weight), yva).item()
        if vloss < best - 1e-5:
            best, best_state, patience = vloss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 8:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xte, edge_index, edge_weight).cpu().numpy()
    yte_np = yte.cpu().numpy()
    # per-station R^2 (denormalize PM2.5 only)
    pm_min, pm_max = sc.data_min_[0], sc.data_max_[0]
    def denorm(a): return a*(pm_max-pm_min)+pm_min
    r2_per = [r2_score(denorm(yte_np[:, n]), denorm(pred[:, n]))
              for n in range(yte_np.shape[1])]
    dirichlet = model.spatial._dirichlet      # last forward's per-layer energies
    return np.array(r2_per), dirichlet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=["beijing", "london", "madrid"])
    ap.add_argument("--topology", default="distance", choices=["distance", "correlation"])
    ap.add_argument("--data_dir", default=None, help="Beijing PRSA dir (optional override)")
    args = ap.parse_args()
    if args.data_dir:
        global BEIJING_DIR
        BEIJING_DIR = args.data_dir
    device = "mps" if torch.backends.mps.is_available() else (
             "cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== {args.city.upper()} / {args.topology} / device={device} ===")

    variants = [("linear", 1), ("gcn", 1), ("gcn", 2), ("gat", 2)]
    agg = {}
    for kind, nl in variants:
        per_seed = []
        for s in SEEDS:
            r2_per, dir_e = run(args.city, args.topology, kind, nl, s, device)
            per_seed.append(r2_per.mean())
        name = f"{kind}{nl}L"
        agg[name] = (np.mean(per_seed), np.std(per_seed), r2_per, dir_e)

    lin = agg["linear1L"][0]
    print(f"\n{'variant':14s} {'R2(mean+/-std)':20s} {'dR2 vs Linear':14s} {'Dirichlet(per layer)'}")
    for name, (m, sd, r2_per, dir_e) in agg.items():
        d = "" if name == "linear1L" else f"{m-lin:+.3f}"
        de = ",".join(f"{x:.3f}" for x in dir_e) if dir_e else "-"
        print(f"{name:14s} {m:.4f} +/- {sd:.4f}     {d:14s} {de}")

    # Wilcoxon: does 1-layer GCN still underperform Linear per station? (rebuttal core)
    lin_r2 = agg["linear1L"][2]; g1_r2 = agg["gcn1L"][2]
    try:
        w = wilcoxon(g1_r2, lin_r2, alternative="less")
        print(f"\n[Over-smoothing control] 1-layer GCN < Linear  Wilcoxon p = {w.pvalue:.4f}")
        print("  -> if p<0.05 AND dR2<0, the underperformance is NOT a 2-layer over-smoothing artifact.")
    except Exception as e:
        print("Wilcoxon skipped:", e)


if __name__ == "__main__":
    main()
