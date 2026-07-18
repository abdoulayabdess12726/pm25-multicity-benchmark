"""
06_train_multistation.py — Multi-station benchmark GCN vs Linear+Transformer
=============================================================================
Adapté de solution_A_multinode_gcn.py pour supporter Beijing ET London.

Usage:
    # Beijing (12 stations UCI)
    python 06_train_multistation.py --city beijing \
        --data_dir data/beijing_real/PRSA_Data_20130301-20170228

    # London (9 stations LAQN)
    python 06_train_multistation.py --city london

    # Mode rapide pour tester
    python 06_train_multistation.py --city beijing --quick \
        --data_dir data/beijing_real/PRSA_Data_20130301-20170228
    python 06_train_multistation.py --city london --quick

Sorties:
    results/<city>/multistation_results.json
"""

import argparse
import os
import json
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────

SEEDS       = [42, 123, 777]
SEQ_LEN     = 24
BATCH_SIZE  = 64
D_MODEL     = 64
N_HEADS     = 4
N_LAYERS    = 2
DROPOUT     = 0.1
LR          = 1e-3
WD          = 1e-5
PATIENCE    = 8
MAX_EPOCHS  = 50
K_NEIGHBORS = 5

# Coordonnées Beijing (12 stations UCI)
BEIJING_COORDS = {
    'Aotizhongxin':  (39.982, 116.397),
    'Changping':     (40.218, 116.231),
    'Dingling':      (40.292, 116.220),
    'Dongsi':        (39.929, 116.417),
    'Guanyuan':      (39.929, 116.339),
    'Gucheng':       (39.914, 116.184),
    'Huairou':       (40.328, 116.628),
    'Nongzhanguan':  (39.937, 116.461),
    'Shunyi':        (40.127, 116.655),
    'Tiantan':       (39.886, 116.407),
    'Wanliu':        (39.987, 116.287),
    'Wanshouxigong': (39.878, 116.352),
}

# Variables globales (mises à jour par load_*_data)
STATION_COORDS = BEIJING_COORDS
STATION_NAMES  = list(STATION_COORDS.keys())
N_STATIONS     = len(STATION_NAMES)
FEATURES       = ['PM2.5', 'TEMP', 'PRES', 'DEWP', 'WSPM']  # Beijing default
TARGET         = 'PM2.5'

# ─────────────────────────────────────────────
# 2. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ─────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def build_graph(k=K_NEIGHBORS):
    """Construit edge_index et edge_weight (k-NN basé sur distance géographique)."""
    n = N_STATIONS
    dist = np.zeros((n, n))
    for i, si in enumerate(STATION_NAMES):
        for j, sj in enumerate(STATION_NAMES):
            if i != j:
                dist[i, j] = haversine_km(*STATION_COORDS[si], *STATION_COORDS[sj])
            else:
                dist[i, j] = np.inf

    src, dst, wts = [], [], []
    k_eff = min(k, n - 1)
    for i in range(n):
        neighbors = np.argsort(dist[i])[:k_eff]
        for j in neighbors:
            src.append(i); dst.append(j)
            wts.append(1.0 / dist[i, j])

    edge_index  = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(wts,        dtype=torch.float32)
    edge_weight = (edge_weight - edge_weight.min()) / (edge_weight.max() - edge_weight.min() + 1e-8)
    return edge_index, edge_weight


def build_correlation_graph(data, k=K_NEIGHBORS, threshold=0.0):
    """Construit un graphe basé sur la corrélation PM2.5 entre stations."""
    feat_idx = FEATURES.index('PM2.5')
    pm25 = data[:, :, feat_idx]

    corr = np.corrcoef(pm25.T)
    np.fill_diagonal(corr, -np.inf)

    print("\n  Matrice de corrélation PM2.5 entre stations :")
    print("  " + "  ".join(f"{n[:6]:>6}" for n in STATION_NAMES))
    for i, name in enumerate(STATION_NAMES):
        row = "  ".join(f"{corr[i,j]:6.3f}" if corr[i,j] > -np.inf else "  ---  "
                        for j in range(N_STATIONS))
        print(f"  {name[:12]:<12} {row}")

    src, dst, wts = [], [], []
    k_eff = min(k, N_STATIONS - 1)
    for i in range(N_STATIONS):
        neighbors = np.argsort(corr[i])[::-1][:k_eff]
        for j in neighbors:
            if corr[i, j] > threshold:
                src.append(i); dst.append(j)
                wts.append(float(corr[i, j]))

    if len(wts) == 0:
        raise ValueError("Aucun edge créé — threshold trop élevé.")

    edge_index  = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(wts,        dtype=torch.float32)
    edge_weight = (edge_weight - edge_weight.min()) / \
                  (edge_weight.max() - edge_weight.min() + 1e-8)

    print(f"\n  Graphe corrélation : {N_STATIONS} nœuds, "
          f"{edge_index.shape[1]} edges (k={k_eff}, seuil={threshold})")
    return edge_index, edge_weight


def load_beijing_data(data_dir):
    """Charge les CSV UCI Beijing et retourne un tenseur [T, N, F]."""
    global STATION_COORDS, STATION_NAMES, N_STATIONS, FEATURES
    STATION_COORDS = BEIJING_COORDS
    STATION_NAMES  = list(STATION_COORDS.keys())
    N_STATIONS     = len(STATION_NAMES)
    FEATURES       = ['PM2.5', 'TEMP', 'PRES', 'DEWP', 'WSPM']

    dfs = []
    for name in STATION_NAMES:
        pattern = f"PRSA_Data_{name}_20130301-20170228.csv"
        path = os.path.join(data_dir, pattern)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fichier manquant : {path}")
        df = pd.read_csv(path)
        df['datetime'] = pd.to_datetime(df[['year','month','day','hour']])
        df = df.set_index('datetime').sort_index()
        df = df[FEATURES].copy()
        df = df.interpolate(method='linear').ffill().bfill()
        dfs.append(df)

    common_idx = dfs[0].index
    for df in dfs[1:]:
        common_idx = common_idx.intersection(df.index)

    data = np.stack([df.loc[common_idx].values for df in dfs], axis=1)
    print(f"Beijing chargé : {data.shape[0]} timesteps, {N_STATIONS} stations, "
          f"{len(FEATURES)} features")
    return data


def load_london_data():
    """Charge London depuis le parquet enrichi (PM2.5 + TEMP + PRES + DEWP + WSPM).
    Format identique à Beijing : [T, N, 6]"""
    global STATION_COORDS, STATION_NAMES, N_STATIONS, FEATURES

    parquet_path = "data/london_processed/london_full_hourly.parquet"
    coords_path  = "data/london_laqn/station_coords.csv"

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f"Missing {parquet_path}. Run 01c_preprocess_london.py first.\n"
            f"Pipeline required: 01b_download_london.py -> 01d_download_london_weather.py -> 01c_preprocess_london.py"
        )

    full_df = pd.read_parquet(parquet_path)
    coords_df = pd.read_csv(coords_path)

    coord_map = {row['station']: (float(row['lat']), float(row['lon']))
                 for _, row in coords_df.iterrows()}

    FEATURES = ['PM2.5', 'TEMP', 'PRES', 'DEWP', 'WSPM']
    
    # Liste des stations présentes
    valid_stations = sorted(full_df['station'].unique())
    valid_stations = [s for s in valid_stations if s in coord_map]
    STATION_NAMES  = valid_stations
    STATION_COORDS = {s: coord_map[s] for s in valid_stations}
    N_STATIONS     = len(valid_stations)
    
    # Construire le tenseur [T, N, F]
    # Pivot par station, conserver l'ordre des features
    arrs = []
    common_idx = None
    for station in valid_stations:
        sub = full_df[full_df['station'] == station].set_index('datetime').sort_index()
        # Drop missing
        sub = sub[FEATURES].interpolate(method='linear', limit=6).ffill().bfill()
        if common_idx is None:
            common_idx = sub.index
        else:
            common_idx = common_idx.intersection(sub.index)
    
    for station in valid_stations:
        sub = full_df[full_df['station'] == station].set_index('datetime').sort_index()
        sub = sub[FEATURES].interpolate(method='linear', limit=6).ffill().bfill()
        sub = sub.loc[common_idx]
        arrs.append(sub.values.astype(np.float32))
    
    data = np.stack(arrs, axis=1)  # [T, N, F]
    print(f"London chargé : {data.shape[0]} timesteps, {N_STATIONS} stations, "
          f"{data.shape[2]} features physiques (PM2.5 + TEMP + PRES + DEWP + WSPM)")
    return data


def load_madrid_data():
    """Charge Madrid depuis le parquet enrichi."""
    global STATION_COORDS, STATION_NAMES, N_STATIONS, FEATURES

    parquet_path = "data/madrid_processed/madrid_full_hourly.parquet"
    coords_path  = "data/madrid_openaq/station_coords_valid.csv"

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f"Missing {parquet_path}. Run 01f then 01g first."
        )

    full_df = pd.read_parquet(parquet_path)
    coords_df = pd.read_csv(coords_path)

    coord_map = {row['station']: (float(row['lat']), float(row['lon']))
                 for _, row in coords_df.iterrows()}

    FEATURES = ['PM2.5', 'TEMP', 'PRES', 'DEWP', 'WSPM']
    
    valid_stations = sorted(full_df['station'].unique())
    valid_stations = [s for s in valid_stations if s in coord_map]
    STATION_NAMES  = valid_stations
    STATION_COORDS = {s: coord_map[s] for s in valid_stations}
    N_STATIONS     = len(valid_stations)
    
    arrs = []
    common_idx = None
    for station in valid_stations:
        sub = full_df[full_df['station'] == station].set_index('datetime').sort_index()
        sub = sub[FEATURES].interpolate(method='linear', limit=6).ffill().bfill()
        if common_idx is None:
            common_idx = sub.index
        else:
            common_idx = common_idx.intersection(sub.index)
    
    for station in valid_stations:
        sub = full_df[full_df['station'] == station].set_index('datetime').sort_index()
        sub = sub[FEATURES].interpolate(method='linear', limit=6).ffill().bfill()
        sub = sub.loc[common_idx]
        arrs.append(sub.values.astype(np.float32))
    
    data = np.stack(arrs, axis=1)
    
    # Sanity checks
    assert not np.isnan(data).any(), "NaN detected in Madrid data"
    assert not np.isinf(data).any(), "Inf detected in Madrid data"
    pm25_idx = FEATURES.index('PM2.5')
    n_neg = (data[:, :, pm25_idx] < 0).sum()
    if n_neg > 0:
        data[:, :, pm25_idx] = np.clip(data[:, :, pm25_idx], 0, None)
    
    print(f"Madrid charge : {data.shape[0]} timesteps, {N_STATIONS} stations, "
          f"{data.shape[2]} features (NO2 is per-station constant proxy).")
    return data



def generate_synthetic_data(n_hours=21994):
    """Données synthétiques pour démo / test."""
    global FEATURES
    FEATURES = ['PM2.5', 'TEMP', 'PRES', 'DEWP', 'WSPM']
    np.random.seed(42)
    t = np.arange(n_hours)
    base = (30 + 20*np.sin(2*np.pi*t/(24*365))
            + 8*np.sin(2*np.pi*t/24)
            + np.random.normal(0, 5, n_hours))

    data_list = []
    for i, name in enumerate(STATION_NAMES):
        lat, lon = STATION_COORDS[name]
        spatial_shift = (lat - 39.9) * 15 + (lon - 116.3) * 10
        pm25  = np.clip(base + spatial_shift + np.random.normal(0, 8, n_hours), 2, 400)
        no2   = np.clip(30 + 0.4*pm25 + np.random.normal(0, 10, n_hours), 5, 150)
        temp  = 12 + 18*np.sin(2*np.pi*(t - 24*90)/(24*365)) + np.random.normal(0,3,n_hours)
        pres  = 1010 + 5*np.sin(2*np.pi*t/(24*365)) + np.random.normal(0,2,n_hours)
        dewp  = temp - 10 + np.random.normal(0, 3, n_hours)
        wspm  = np.abs(2 + np.random.normal(0, 1.5, n_hours))
        station_data = np.stack([pm25, no2, temp, pres, dewp, wspm], axis=1)
        data_list.append(station_data)

    data = np.stack(data_list, axis=1)
    print(f"Synthétique : {data.shape[0]} timesteps, {N_STATIONS} stations, "
          f"{len(FEATURES)} features")
    return data


def split_and_scale(data):
    """Split 70/15/15 chronologique + MinMax fit sur train uniquement."""
    T = data.shape[0]
    t1 = int(0.70 * T)
    t2 = int(0.85 * T)
    train_raw, val_raw, test_raw = data[:t1], data[t1:t2], data[t2:]

    scaler = MinMaxScaler()
    T_tr, N, F = train_raw.shape
    scaler.fit(train_raw.reshape(-1, F))

    def scale(arr):
        t, n, f = arr.shape
        return scaler.transform(arr.reshape(-1, f)).reshape(t, n, f)

    return scale(train_raw), scale(val_raw), scale(test_raw), scaler


# ─────────────────────────────────────────────
# 3. DATASET PYTORCH
# ─────────────────────────────────────────────

class MultiStationDataset(Dataset):
    def __init__(self, data, seq_len=SEQ_LEN):
        self.data    = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len
        self.feat_idx = FEATURES.index(TARGET)

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len, :, self.feat_idx]
        return x, y


# ─────────────────────────────────────────────
# 4. MODÈLES
# ─────────────────────────────────────────────

class GCNEncoder(nn.Module):
    """GCN vectorisé — n_gcn_layers ∈ {1, 2}.
    n_gcn_layers=1 sert de contrôle anti-over-smoothing (une seule agrégation,
    l'over-smoothing exigeant l'empilement de couches)."""
    def __init__(self, in_dim, out_dim, n_nodes, n_gcn_layers=2):
        super().__init__()
        self.n_gcn_layers = n_gcn_layers
        self.W1   = nn.Linear(in_dim,  out_dim, bias=False)
        self.W2   = nn.Linear(out_dim, out_dim, bias=False) if n_gcn_layers == 2 else None
        self.norm = nn.LayerNorm(out_dim)
        self.act  = nn.ReLU()
        self._a_hat = None
        self.n_nodes = n_nodes

    def _build_a_hat(self, edge_index, edge_weight, device):
        N = self.n_nodes
        A = torch.zeros(N, N, device=device)
        A[edge_index[0], edge_index[1]] = edge_weight
        A = A + torch.eye(N, device=device)
        D_inv_sqrt = torch.diag(A.sum(1).pow(-0.5))
        return D_inv_sqrt @ A @ D_inv_sqrt

    def forward(self, x, edge_index, edge_weight, return_layers=False):
        if self._a_hat is None or self._a_hat.device != x.device:
            self._a_hat = self._build_a_hat(edge_index, edge_weight, x.device)
        A_hat = self._a_hat
        layers = []

        h = self.W1(x)
        h = torch.einsum('nm,bmf->bnf', A_hat, h)
        layers.append(h)                       # embedding après couche 1

        if self.n_gcn_layers == 2:
            h = self.act(h)
            h = self.W2(h)
            h = torch.einsum('nm,bmf->bnf', A_hat, h)
            layers.append(h)                   # embedding après couche 2

        h = self.norm(h)
        if return_layers:
            return h, layers
        return h


class GATEncoder(nn.Module):
    """GAT dense multi-têtes (même formalisme matriciel que GCNEncoder), pour
    tester si le failure mode dépasse le GCN. Les poids d'arête sont appris par
    attention ; edge_weight n'est utilisé que pour le masque de voisinage."""
    def __init__(self, in_dim, out_dim, n_nodes, n_gat_layers=2, heads=4):
        super().__init__()
        assert out_dim % heads == 0, "out_dim doit être divisible par heads"
        self.n_gat_layers = n_gat_layers
        self.heads = heads
        self.dh = out_dim // heads
        self.n_nodes = n_nodes
        self.W1 = nn.Linear(in_dim, out_dim, bias=False)
        self.a1 = nn.Parameter(torch.empty(heads, 2 * self.dh)); nn.init.xavier_uniform_(self.a1)
        if n_gat_layers == 2:
            self.W2 = nn.Linear(out_dim, out_dim, bias=False)
            self.a2 = nn.Parameter(torch.empty(heads, 2 * self.dh)); nn.init.xavier_uniform_(self.a2)
        self.norm  = nn.LayerNorm(out_dim)
        self.leaky = nn.LeakyReLU(0.2)
        self.act   = nn.ReLU()
        self._mask = None

    def _build_mask(self, edge_index, device):
        N = self.n_nodes
        M = torch.zeros(N, N, device=device)
        M[edge_index[0], edge_index[1]] = 1.0
        M = M + torch.eye(N, device=device)         # self-loops
        return M

    def _gat_layer(self, x, W, a, mask):
        B, N, _ = x.shape
        h = W(x).view(B, N, self.heads, self.dh)            # (B,N,H,dh)
        a_src, a_dst = a[:, :self.dh], a[:, self.dh:]
        e_src = torch.einsum('bnhd,hd->bnh', h, a_src)       # (B,N,H)
        e_dst = torch.einsum('bnhd,hd->bnh', h, a_dst)
        e = self.leaky(e_src.unsqueeze(2) + e_dst.unsqueeze(1))   # (B,Ni,Nj,H)
        e = e.masked_fill(mask.unsqueeze(0).unsqueeze(-1) == 0, float('-inf'))
        alpha = torch.softmax(e, dim=2)                      # sur les voisins j
        out = torch.einsum('bijh,bjhd->bihd', alpha, h)      # (B,N,H,dh)
        return out.reshape(B, N, self.heads * self.dh)

    def forward(self, x, edge_index, edge_weight, return_layers=False):
        if self._mask is None or self._mask.device != x.device:
            self._mask = self._build_mask(edge_index, x.device)
        layers = []
        h = self.act(self._gat_layer(x, self.W1, self.a1, self._mask)); layers.append(h)
        if self.n_gat_layers == 2:
            h = self._gat_layer(h, self.W2, self.a2, self._mask);       layers.append(h)
        h = self.norm(h)
        if return_layers:
            return h, layers
        return h


def build_a_hat_ref(edge_index, edge_weight, n_nodes, device):
    """Adjacence normalisée (avec self-loops) — géométrie de référence commune
    pour mesurer la Dirichlet energy de tous les encodeurs."""
    A = torch.zeros(n_nodes, n_nodes, device=device)
    A[edge_index[0], edge_index[1]] = edge_weight
    A = A + torch.eye(n_nodes, device=device)
    D_inv_sqrt = torch.diag(A.sum(1).pow(-0.5))
    return D_inv_sqrt @ A @ D_inv_sqrt


def dirichlet_energy(H, A_hat):
    """E(H) = 0.5 · Σ_ij Â_ij ‖h_i − h_j‖²  (moyenne sur le batch).
    Plus la valeur baisse vers 0 à mesure qu'on empile les couches, plus il y a
    over-smoothing. H: (B, N, d) ; A_hat: (N, N)."""
    diff = H.unsqueeze(2) - H.unsqueeze(1)            # (B,N,N,d)
    sq = diff.pow(2).sum(-1)                          # (B,N,N)
    E = 0.5 * (A_hat.unsqueeze(0) * sq).sum(dim=(1, 2))
    return float(E.mean().detach().cpu())


class LinearEncoder(nn.Module):
    """Projection linéaire simple — pas de communication inter-stations."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index=None, edge_weight=None, return_layers=False):
        h = self.norm(self.proj(x))
        if return_layers:
            return h, []
        return h


class TemporalTransformer(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, dropout):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*4, dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.register_buffer('pos_enc', self._make_pe(500, d_model))

    @staticmethod
    def _make_pe(max_len, d_model):
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)

    def forward(self, x):
        B, N, T, D = x.shape
        x = x.reshape(B * N, T, D)
        x = x + self.pos_enc[:, :T, :]
        out = self.transformer(x)
        out = out[:, -1, :]
        return out.reshape(B, N, D)


class SpatioTemporalModel(nn.Module):
    def __init__(self, in_features, d_model, n_heads, n_layers, dropout,
                 n_nodes, use_gcn=True, encoder_type=None):
        super().__init__()
        # backward compat: use_gcn bool maps to gcn2/linear if encoder_type unset
        if encoder_type is None:
            encoder_type = 'gcn2' if use_gcn else 'linear'
        self.encoder_type = encoder_type
        if encoder_type == 'linear':
            self.encoder = LinearEncoder(in_features, d_model)
        elif encoder_type == 'gcn1':
            self.encoder = GCNEncoder(in_features, d_model, n_nodes, n_gcn_layers=1)
        elif encoder_type == 'gcn2':
            self.encoder = GCNEncoder(in_features, d_model, n_nodes, n_gcn_layers=2)
        elif encoder_type == 'gat':
            self.encoder = GATEncoder(in_features, d_model, n_nodes, n_gat_layers=2,
                                      heads=n_heads)
        else:
            raise ValueError(f"encoder_type inconnu: {encoder_type}")
        self.temporal = TemporalTransformer(d_model, n_heads, n_layers, dropout)
        self.head     = nn.Linear(d_model, 1)

    def forward(self, x, edge_index, edge_weight):
        B, T, N, F = x.shape
        x_flat = x.reshape(B * T, N, F)
        enc    = self.encoder(x_flat, edge_index, edge_weight)
        enc    = enc.reshape(B, T, N, -1).permute(0, 2, 1, 3)
        out    = self.temporal(enc)
        return self.head(out).squeeze(-1)

    def encode_layers(self, x, edge_index, edge_weight):
        """Renvoie les embeddings nodaux par couche pour la Dirichlet energy."""
        B, T, N, F = x.shape
        _, layers = self.encoder(x.reshape(B * T, N, F), edge_index, edge_weight,
                                 return_layers=True)
        return layers


# ─────────────────────────────────────────────
# 5. ENTRAÎNEMENT
# ─────────────────────────────────────────────

def train_model(model, train_loader, val_loader,
                edge_index, edge_weight, device,
                max_epochs=MAX_EPOCHS, patience=PATIENCE):

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    criterion = nn.MSELoss()
    best_val, best_state, no_improve = np.inf, None, 0
    ei = edge_index.to(device)
    ew = edge_weight.to(device)

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb, ei, ew)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb, ei, ew)
                val_losses.append(criterion(pred, yb).item())
        val_loss = np.mean(val_losses)

        if epoch % 5 == 0:
            print(f"     epoch {epoch:3d} | val_loss={val_loss:.5f}")

        if val_loss < best_val:
            best_val  = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"   Early stop epoch {epoch} (val_loss={best_val:.5f})")
                break

    model.load_state_dict(best_state)
    return model


def evaluate(model, loader, edge_index, edge_weight, scaler, device):
    model.eval()
    ei = edge_index.to(device)
    ew = edge_weight.to(device)
    feat_idx = FEATURES.index(TARGET)

    all_pred, all_true = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb, ei, ew).cpu().numpy()
            true = yb.numpy()
            all_pred.append(pred)
            all_true.append(true)

    pred = np.concatenate(all_pred, axis=0)
    true = np.concatenate(all_true, axis=0)

    T, N = pred.shape
    F    = len(FEATURES)

    def denorm(arr):
        tmp = np.zeros((T * N, F))
        tmp[:, feat_idx] = arr.reshape(-1)
        inv = scaler.inverse_transform(tmp)
        return inv[:, feat_idx].reshape(T, N)

    pred_dn = denorm(pred)
    true_dn = denorm(true)

    p_flat = pred_dn.reshape(-1)
    t_flat = true_dn.reshape(-1)
    mae  = mean_absolute_error(t_flat, p_flat)
    rmse = np.sqrt(mean_squared_error(t_flat, p_flat))
    r2   = r2_score(t_flat, p_flat)

    per_station = {}
    for i, name in enumerate(STATION_NAMES):
        per_station[name] = {
            'MAE':  float(mean_absolute_error(true_dn[:, i], pred_dn[:, i])),
            'RMSE': float(np.sqrt(mean_squared_error(true_dn[:, i], pred_dn[:, i]))),
            'R2':   float(r2_score(true_dn[:, i], pred_dn[:, i])),
        }

    return mae, rmse, r2, per_station


# ─────────────────────────────────────────────
# 6. BOUCLE PRINCIPALE
# ─────────────────────────────────────────────

def compute_dirichlet_for_model(model, loader, edge_index, edge_weight, device, n_nodes):
    """Mesure la Dirichlet energy par couche sur le 1er batch de test."""
    model.eval()
    A_hat = build_a_hat_ref(edge_index.to(device), edge_weight.to(device), n_nodes, device)
    with torch.no_grad():
        for xb, _ in loader:
            layers = model.encode_layers(xb.to(device), edge_index.to(device),
                                         edge_weight.to(device))
            return [dirichlet_energy(h, A_hat) for h in layers]
    return []


# default = comportement original (GCN2 + Linear) ; --control ajoute gcn1 + gat
DEFAULT_VARIANTS = [('GCN+Transformer', 'gcn2'), ('Linear+Transformer', 'linear')]
CONTROL_VARIANTS = [('Linear+Transformer', 'linear'),
                    ('GCN1+Transformer',  'gcn1'),
                    ('GCN+Transformer',   'gcn2'),
                    ('GAT+Transformer',   'gat')]


def run_experiment(data, edge_index, edge_weight, device, variants=None):
    if variants is None:
        variants = DEFAULT_VARIANTS
    train_d, val_d, test_d, scaler = split_and_scale(data)

    results = {name: [] for name, _ in variants}
    per_station_results = {}                 # seed primaire (SEEDS[0]) — pour les prints
    per_station_all = {}                     # {model: {seed: {station: metrics}}} — persisté
    dirichlet_results = {}

    for model_name, enc_type in variants:
        print(f"\n{'='*55}")
        print(f"  Modèle : {model_name}  (encoder={enc_type})")
        print(f"{'='*55}")
        seed_metrics = []

        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            print(f"  -> Seed {seed}...")

            train_ds = MultiStationDataset(train_d)
            val_ds   = MultiStationDataset(val_d)
            test_ds  = MultiStationDataset(test_d)

            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
            test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

            model = SpatioTemporalModel(
                in_features=len(FEATURES), d_model=D_MODEL,
                n_heads=N_HEADS, n_layers=N_LAYERS,
                dropout=DROPOUT, n_nodes=N_STATIONS, encoder_type=enc_type
            ).to(device)

            model = train_model(model, train_loader, val_loader,
                                edge_index, edge_weight, device,
                                max_epochs=MAX_EPOCHS, patience=PATIENCE)

            mae, rmse, r2, per_st = evaluate(
                model, test_loader, edge_index, edge_weight, scaler, device)

            print(f"     MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.4f}")
            seed_metrics.append((mae, rmse, r2))

            per_station_all.setdefault(model_name, {})[seed] = per_st

            if seed == SEEDS[0]:
                per_station_results[model_name] = per_st
                if enc_type in ('gcn1', 'gcn2', 'gat'):
                    de = compute_dirichlet_for_model(
                        model, test_loader, edge_index, edge_weight, device, N_STATIONS)
                    dirichlet_results[model_name] = de
                    print(f"     Dirichlet/couche : {['%.3f' % e for e in de]}")

        results[model_name] = seed_metrics

    return results, per_station_results, scaler, dirichlet_results, per_station_all


def print_results_table(results, per_station_results):
    print("\n" + "="*65)
    print("  TABLEAU 1 -- Résultats globaux (moyenne +/- SD)")
    print("="*65)
    print(f"  {'Modèle':<25} {'MAE':>8} {'RMSE':>10} {'R2':>10}")
    print("-"*65)

    for name, metrics in results.items():
        maes  = [m[0] for m in metrics]
        rmses = [m[1] for m in metrics]
        r2s   = [m[2] for m in metrics]
        print(f"  {name:<25} "
              f"{np.mean(maes):6.2f}+/-{np.std(maes):.2f}  "
              f"{np.mean(rmses):7.2f}+/-{np.std(rmses):.2f}  "
              f"{np.mean(r2s):.4f}+/-{np.std(r2s):.4f}")

    print("\n" + "="*65)
    print("  TABLEAU 2 -- Résultats par station (seed primaire)")
    print("="*65)
    print(f"  {'Station':<20} {'GCN R2':>9} {'Lin R2':>9}  {'Delta':>7}")
    print("-"*65)

    gcn_ps = per_station_results.get('GCN+Transformer', {})
    lin_ps = per_station_results.get('Linear+Transformer', {})

    for s in STATION_NAMES:
        gcn_r2 = gcn_ps.get(s, {}).get('R2', float('nan'))
        lin_r2 = lin_ps.get(s, {}).get('R2', float('nan'))
        delta  = gcn_r2 - lin_r2
        marker = " GCN+" if delta > 0 else " egal" if abs(delta) < 1e-4 else " Lin+"
        print(f"  {s:<20}  {gcn_r2:7.4f}   {lin_r2:7.4f}  {delta:+.4f}{marker}")

    all_gcn = [v['R2'] for v in gcn_ps.values()]
    all_lin = [v['R2'] for v in lin_ps.values()]
    print("-"*65)
    if all_gcn and all_lin:
        gcn_wins = sum(g > l for g, l in zip(all_gcn, all_lin))
        print(f"  GCN supérieur sur {gcn_wins}/{N_STATIONS} stations")
        print(f"  Delta R2 moyen = {np.mean([g-l for g,l in zip(all_gcn,all_lin)]):+.4f}")
    print("="*65)


# ─────────────────────────────────────────────
# 7. ENTRÉE PRINCIPALE
# ─────────────────────────────────────────────

def print_control_summary(results, per_station, dirichlet):
    """Synthèse de la rébuttal reviewer : ΔR² vs Linear, Dirichlet, Wilcoxon."""
    print("\n" + "="*72)
    print("  CONTRÔLE OVER-SMOOTHING & ARCHITECTURE (rébuttal reviewer)")
    print("="*72)
    lin = results.get('Linear+Transformer', [])
    lin_r2 = float(np.mean([m[2] for m in lin])) if lin else float('nan')
    print(f"  {'Variante':<22} {'R2 (mean±sd)':<20} {'ΔR2 vs Lin':<12} {'Dirichlet/couche'}")
    print("-"*72)
    for name, metrics in results.items():
        r2s = [m[2] for m in metrics]
        m, sd = np.mean(r2s), np.std(r2s)
        d = "" if name == 'Linear+Transformer' else f"{m-lin_r2:+.3f}"
        de = dirichlet.get(name, [])
        de_s = ", ".join(f"{x:.2f}" for x in de) if de else "-"
        print(f"  {name:<22} {m:.4f} ± {sd:.4f}     {d:<12} {de_s}")
    print("-"*72)

    # Wilcoxon par station : variante graphe < Linear ?
    try:
        from scipy.stats import wilcoxon
        lin_ps = per_station.get('Linear+Transformer', {})
        lin_arr = np.array([lin_ps[s]['R2'] for s in STATION_NAMES])
        for gname in ('GCN1+Transformer', 'GCN+Transformer', 'GAT+Transformer'):
            ps = per_station.get(gname)
            if not ps:
                continue
            arr = np.array([ps[s]['R2'] for s in STATION_NAMES])
            try:
                w = wilcoxon(arr, lin_arr, alternative='less')
                worse = int((arr < lin_arr).sum())
                print(f"  {gname:<22} sous-performe Linear sur {worse}/{len(arr)} stations "
                      f"| Wilcoxon p={w.pvalue:.4f}")
            except Exception as e:
                print(f"  {gname}: Wilcoxon ignoré ({e})")
    except ImportError:
        print("  (scipy absent — Wilcoxon non calculé)")
    print("  → GCN1 sous-performant + Dirichlet non-effondrée = PAS un artefact d'over-smoothing.")
    print("  → GAT sous-performant aussi = failure mode non spécifique au GCN.")
    print("="*72)


def main():
    parser = argparse.ArgumentParser(
        description='Multi-station benchmark Beijing/London')
    parser.add_argument('--city', type=str, default='beijing',
                    choices=['beijing', 'london', 'madrid'],
                        help='Ville à analyser')
    parser.add_argument('--data_dir',  type=str, default=None,
                        help='Dossier CSV (Beijing uniquement)')
    parser.add_argument('--synthetic', action='store_true',
                        help='Données synthétiques (démo)')
    parser.add_argument('--quick', action='store_true',
                        help='Mode rapide : 1 seed, 10 epochs, modèle réduit')
    parser.add_argument('--graph', type=str, default='both',
                        choices=['distance', 'correlation', 'both'])
    parser.add_argument('--control', action='store_true',
                        help='Ajoute GCN1 + GAT + Dirichlet (rébuttal over-smoothing)')
    args = parser.parse_args()

    # Device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Device : Apple MPS")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Device : CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device('cpu')
        print("Device : CPU")

    # Données
  # Données
      # Données
    if args.city == 'london':
        data = load_london_data()
    elif args.city == 'madrid':
        data = load_madrid_data()
    elif args.synthetic or args.data_dir is None:
        print("\n[Mode synthétique. Utilisez --data_dir pour vraies données.]")
        n_hours = 5000 if args.quick else 21994
        data = generate_synthetic_data(n_hours)
    else:
        data = load_beijing_data(args.data_dir)

    # Mode rapide
    global SEEDS, MAX_EPOCHS, PATIENCE, D_MODEL, N_HEADS, N_LAYERS, BATCH_SIZE, SEQ_LEN
    if args.quick:
        SEEDS      = [42]
        MAX_EPOCHS = 10
        PATIENCE   = 5
        D_MODEL    = 32
        N_HEADS    = 2
        N_LAYERS   = 1
        BATCH_SIZE = 128
        SEQ_LEN    = 12
        print("[Mode --quick activé]")

    # Graphe(s)
    graphs_to_run = []
    if args.graph in ('distance', 'both'):
        ei_dist, ew_dist = build_graph(k=K_NEIGHBORS)
        graphs_to_run.append(('distance', ei_dist.to(device), ew_dist.to(device)))
        print(f"\nGraphe DISTANCE : {N_STATIONS} nœuds, {ei_dist.shape[1]} edges")

    if args.graph in ('correlation', 'both'):
        print("\nGraphe CORRÉLATION sur train...")
        train_len = int(0.70 * len(data))
        ei_corr, ew_corr = build_correlation_graph(data[:train_len], k=K_NEIGHBORS)
        graphs_to_run.append(('correlation', ei_corr.to(device), ew_corr.to(device)))

    print(f"\nFeatures : {FEATURES}")
    print(f"SEQ_LEN={SEQ_LEN}h | BATCH={BATCH_SIZE} | D_MODEL={D_MODEL} | "
          f"HEADS={N_HEADS} | LAYERS={N_LAYERS}")

    # Boucle topologies
    variants = CONTROL_VARIANTS if args.control else DEFAULT_VARIANTS
    all_graph_results = {}
    t0 = time.time()

    for graph_name, ei_dev, ew_dev in graphs_to_run:
        print(f"\n{'#'*60}")
        print(f"  TOPOLOGIE : {graph_name.upper()}")
        print(f"{'#'*60}")
        results, per_station_results, scaler, dirichlet_results, per_station_all = run_experiment(
            data, ei_dev, ew_dev, device, variants=variants)
        all_graph_results[graph_name] = {
            'results': results,
            'per_station': per_station_results,
            'per_station_all': per_station_all,
            'dirichlet': dirichlet_results,
        }
        print_results_table(results, per_station_results)
        if args.control:
            print_control_summary(results, per_station_results, dirichlet_results)

    print(f"\nTemps total : {(time.time()-t0)/60:.1f} min")

    # Comparaison finale
    if len(all_graph_results) > 1:
        print("\n" + "="*70)
        print("  TABLEAU COMPARATIF -- GCN selon topologie vs Linear+Transformer")
        print("="*70)
        print(f"  {'Topologie':<25} {'MAE GCN':>9} {'R2 GCN':>9} "
              f"{'R2 Lin':>10} {'Delta':>8}")
        print("-"*70)

        for gname, gdata in all_graph_results.items():
            gcn = gdata['results']['GCN+Transformer']
            lin = gdata['results']['Linear+Transformer']
            gcn_mae  = np.mean([m[0] for m in gcn])
            gcn_r2   = np.mean([m[2] for m in gcn])
            lin_r2   = np.mean([m[2] for m in lin])
            delta    = gcn_r2 - lin_r2
            marker   = "GCN+" if delta > 0 else "Lin+"
            print(f"  {gname:<25}  {gcn_mae:7.2f}   {gcn_r2:7.4f}   "
                  f"{lin_r2:7.4f}  {delta:+.4f}  {marker}")
        print("="*70)

    # ── SAUVEGARDE JSON ──
    out_dir = Path(f"results/{args.city}")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_results = {
        'city': args.city,
        'n_stations': N_STATIONS,
        'station_names': STATION_NAMES,
        'features': FEATURES,
        'seeds': SEEDS,
        'config': {
            'seq_len': SEQ_LEN, 'batch_size': BATCH_SIZE,
            'd_model': D_MODEL, 'n_heads': N_HEADS, 'n_layers': N_LAYERS,
            'max_epochs': MAX_EPOCHS, 'patience': PATIENCE,
            'k_neighbors': K_NEIGHBORS,
        },
        'graphs': {},
    }

    for gname, gdata in all_graph_results.items():
        json_results['graphs'][gname] = {}
        for model_name, seed_metrics in gdata['results'].items():
            json_results['graphs'][gname][model_name] = {
                'MAE':  [float(m[0]) for m in seed_metrics],
                'RMSE': [float(m[1]) for m in seed_metrics],
                'R2':   [float(m[2]) for m in seed_metrics],
                'MAE_mean':  float(np.mean([m[0] for m in seed_metrics])),
                'MAE_std':   float(np.std([m[0] for m in seed_metrics])),
                'RMSE_mean': float(np.mean([m[1] for m in seed_metrics])),
                'RMSE_std':  float(np.std([m[1] for m in seed_metrics])),
                'R2_mean':   float(np.mean([m[2] for m in seed_metrics])),
                'R2_std':    float(np.std([m[2] for m in seed_metrics])),
            }
        json_results['graphs'][gname]['per_station'] = gdata['per_station']
        json_results['graphs'][gname]['per_station_all_seeds'] = gdata.get('per_station_all', {})
        json_results['graphs'][gname]['dirichlet'] = gdata.get('dirichlet', {})

    out_file = out_dir / "multistation_results.json"
    with open(out_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nRésultats sauvegardés : {out_file}")


if __name__ == '__main__':
    main()
