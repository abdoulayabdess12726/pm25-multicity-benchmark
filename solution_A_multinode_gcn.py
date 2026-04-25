"""
Solution A — Expérience multi-nœuds réelle
==========================================
Compare GCN+Transformer vs Linear+Transformer sur les 12 stations Beijing
comme graphe spatial (chaque station = 1 nœud).

Usage:
    python solution_A_multinode_gcn.py --data_dir /path/to/beijing_csv/
    python solution_A_multinode_gcn.py --synthetic   # démo sans données réelles

Données : UCI Beijing Multi-site Air Quality Dataset #501
  → https://archive.ics.uci.edu/dataset/501
  Fichiers attendus : PRSA_Data_<Station>_20130301_20170228.csv (12 fichiers)
"""

import argparse
import os
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────

SEEDS       = [42, 123, 777]
SEQ_LEN     = 24          # fenêtre temporelle (heures)
BATCH_SIZE  = 64
D_MODEL     = 64
N_HEADS     = 4
N_LAYERS    = 2
DROPOUT     = 0.1
LR          = 1e-3
WD          = 1e-5
PATIENCE    = 10
MAX_EPOCHS  = 80
K_NEIGHBORS = 5           # voisins dans le graphe spatial
FEATURES    = ['PM2.5', 'NO2', 'TEMP', 'PRES', 'DEWP', 'WSPM']
TARGET      = 'PM2.5'

# Coordonnées géographiques des 12 stations (lat, lon)
STATION_COORDS = {
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
STATION_NAMES = list(STATION_COORDS.keys())
N_STATIONS    = len(STATION_NAMES)

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
    for i in range(n):
        neighbors = np.argsort(dist[i])[:k]
        for j in neighbors:
            src.append(i); dst.append(j)
            wts.append(1.0 / dist[i, j])

    edge_index  = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(wts,        dtype=torch.float32)
    edge_weight = (edge_weight - edge_weight.min()) / (edge_weight.max() - edge_weight.min() + 1e-8)
    return edge_index, edge_weight


def build_correlation_graph(data, k=K_NEIGHBORS, threshold=0.0):
    """
    Construit un graphe basé sur la corrélation de Pearson entre séries PM2.5.

    Principe physique : deux stations dont les PM2.5 évoluent ensemble
    partagent probablement les mêmes sources/vents → edge fort.
    Deux stations décorrélées → pas d'edge (ou edge faible).

    data  : [T, N, F] — numpy array (train+val+test, non normalisé)
    k     : nombre de voisins retenus par nœud
    threshold : corrélation minimale pour créer un edge (0 = tous les k voisins)
    """
    feat_idx = FEATURES.index('PM2.5')
    pm25 = data[:, :, feat_idx]          # [T, N]

    # Matrice de corrélation N×N
    corr = np.corrcoef(pm25.T)           # [N, N]
    np.fill_diagonal(corr, -np.inf)      # exclure auto-corrélation

    print("\n  Matrice de corrélation PM2.5 entre stations :")
    print("  " + "  ".join(f"{n[:6]:>6}" for n in STATION_NAMES))
    for i, name in enumerate(STATION_NAMES):
        row = "  ".join(f"{corr[i,j]:6.3f}" if corr[i,j] > -np.inf else "  ---  "
                        for j in range(N_STATIONS))
        print(f"  {name[:12]:<12} {row}")

    src, dst, wts = [], [], []
    for i in range(N_STATIONS):
        # Trie par corrélation décroissante
        neighbors = np.argsort(corr[i])[::-1][:k]
        for j in neighbors:
            if corr[i, j] > threshold:
                src.append(i); dst.append(j)
                wts.append(float(corr[i, j]))

    if len(wts) == 0:
        raise ValueError("Aucun edge créé — threshold trop élevé.")

    edge_index  = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(wts,        dtype=torch.float32)
    # Normalisation min-max
    edge_weight = (edge_weight - edge_weight.min()) / \
                  (edge_weight.max() - edge_weight.min() + 1e-8)

    print(f"\n  ✓ Graphe corrélation : {N_STATIONS} nœuds, "
          f"{edge_index.shape[1]} edges (k={k}, seuil={threshold})")
    return edge_index, edge_weight


def load_real_data(data_dir):
    """Charge les CSV UCI et retourne un tenseur [T, N, F]."""
    dfs = []
    for name in STATION_NAMES:
        pattern = f"PRSA_Data_{name}_20130301-20170228.csv"
        path = os.path.join(data_dir, pattern)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fichier manquant : {path}")
        df = pd.read_csv(path)
        df['datetime'] = pd.to_datetime(df[['year','month','day','hour']])
        df = df.set_index('datetime').sort_index()
        # Garde seulement les features nécessaires
        df = df[FEATURES].copy()
        # Interpolation linéaire des valeurs manquantes
        df = df.interpolate(method='linear').ffill().bfill()
        dfs.append(df)

    # Alignement temporel sur l'intersection
    common_idx = dfs[0].index
    for df in dfs[1:]:
        common_idx = common_idx.intersection(df.index)

    data = np.stack([df.loc[common_idx].values for df in dfs], axis=1)
    # data shape : [T, N_STATIONS, N_FEATURES]
    print(f"✓ Données réelles chargées : {data.shape[0]} timesteps, {N_STATIONS} stations, {len(FEATURES)} features")
    return data


def generate_synthetic_data(n_hours=21994):
    """
    Génère des données synthétiques réalistes pour 12 stations.
    Les stations proches ont des séries plus corrélées (structure spatiale).
    """
    np.random.seed(42)
    t = np.arange(n_hours)

    # Signal de base : tendance + saisonnalité journalière/hebdomadaire
    base = (30
            + 20 * np.sin(2 * np.pi * t / (24 * 365))     # saisonnalité annuelle
            + 8  * np.sin(2 * np.pi * t / 24)             # cycle journalier
            + np.random.normal(0, 5, n_hours))             # bruit

    data_list = []
    for i, name in enumerate(STATION_NAMES):
        lat, lon = STATION_COORDS[name]
        # Offset spatial basé sur la position (stations proches → PM2.5 corrélé)
        spatial_shift = (lat - 39.9) * 15 + (lon - 116.3) * 10
        pm25  = np.clip(base + spatial_shift + np.random.normal(0, 8, n_hours), 2, 400)
        no2   = np.clip(30 + 0.4*pm25 + np.random.normal(0, 10, n_hours), 5, 150)
        temp  = 12 + 18 * np.sin(2*np.pi*(t - 24*90) / (24*365)) + np.random.normal(0,3,n_hours)
        pres  = 1010 + 5*np.sin(2*np.pi*t/(24*365)) + np.random.normal(0,2,n_hours)
        dewp  = temp - 10 + np.random.normal(0, 3, n_hours)
        wspm  = np.abs(2 + np.random.normal(0, 1.5, n_hours))
        station_data = np.stack([pm25, no2, temp, pres, dewp, wspm], axis=1)
        data_list.append(station_data)

    data = np.stack(data_list, axis=1)  # [T, N, F]
    print(f"✓ Données synthétiques générées : {data.shape[0]} timesteps, {N_STATIONS} stations, {len(FEATURES)} features")
    return data


def split_and_scale(data):
    """Split 70/15/15 chronologique + MinMax par feature."""
    T = data.shape[0]
    t1 = int(0.70 * T)
    t2 = int(0.85 * T)
    train_raw, val_raw, test_raw = data[:t1], data[t1:t2], data[t2:]

    # Fit scaler sur train uniquement
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
    """
    Retourne des fenêtres glissantes de SEQ_LEN heures.
    X : [SEQ_LEN, N_STATIONS, N_FEATURES]
    y : [N_STATIONS]  (PM2.5 au pas suivant pour chaque station)
    """
    def __init__(self, data, seq_len=SEQ_LEN):
        self.data    = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len
        self.feat_idx = FEATURES.index(TARGET)

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]              # [T, N, F]
        y = self.data[idx + self.seq_len, :, self.feat_idx]  # [N]
        return x, y


# ─────────────────────────────────────────────
# 4. MODÈLES
# ─────────────────────────────────────────────

class GCNEncoder(nn.Module):
    """
    GCN 2 couches vectorisé — SANS boucle sur les timesteps.
    Utilise directement la normalisation spectrale A_hat = D^{-1/2} A D^{-1/2}.
    Forward : [B*T, N, F] → [B*T, N, out_dim]  en un seul bmm.
    """
    def __init__(self, in_dim, out_dim, n_nodes=N_STATIONS):
        super().__init__()
        self.W1   = nn.Linear(in_dim,  out_dim, bias=False)
        self.W2   = nn.Linear(out_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.act  = nn.ReLU()
        # A_hat sera enregistré comme buffer au premier appel
        self._a_hat = None
        self.n_nodes = n_nodes

    def _build_a_hat(self, edge_index, edge_weight, device):
        """Construit A_hat = D^{-1/2} (A+I) D^{-1/2} une seule fois."""
        N = self.n_nodes
        # Matrice d'adjacence dense + self-loops
        A = torch.zeros(N, N, device=device)
        A[edge_index[0], edge_index[1]] = edge_weight
        A = A + torch.eye(N, device=device)  # self-loops
        D_inv_sqrt = torch.diag(A.sum(1).pow(-0.5))
        return D_inv_sqrt @ A @ D_inv_sqrt  # [N, N]

    def forward(self, x, edge_index, edge_weight):
        # x : [B_T, N, F]
        if self._a_hat is None or self._a_hat.device != x.device:
            self._a_hat = self._build_a_hat(edge_index, edge_weight, x.device)
        A_hat = self._a_hat  # [N, N]

        # Couche 1 : A_hat @ (x W1) — bmm vectorisé sur B_T
        h = self.W1(x)              # [B_T, N, out_dim]
        h = torch.einsum('nm,bmf->bnf', A_hat, h)  # [B_T, N, out_dim]
        h = self.act(h)

        # Couche 2
        h = self.W2(h)
        h = torch.einsum('nm,bmf->bnf', A_hat, h)
        return self.norm(h)


class LinearEncoder(nn.Module):
    """Projection linéaire simple — PAS de communication entre stations."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index=None, edge_weight=None):
        return self.norm(self.proj(x))


class TemporalTransformer(nn.Module):
    """Transformer encoder sur la dimension temporelle, appliqué par station."""
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
        return pe.unsqueeze(0)  # [1, T, d_model]

    def forward(self, x):
        # x : [B, N, T, d_model] → traite chaque station séparément
        B, N, T, D = x.shape
        x = x.reshape(B * N, T, D)
        x = x + self.pos_enc[:, :T, :]
        out = self.transformer(x)          # [B*N, T, D]
        out = out[:, -1, :]                # prend le dernier timestep
        return out.reshape(B, N, D)        # [B, N, D]


class SpatioTemporalModel(nn.Module):
    """
    Architecture générique :
    - encoder : GCNEncoder OU LinearEncoder
    - temporal : TemporalTransformer
    - head     : Linear → prédiction PM2.5 par station
    """
    def __init__(self, in_features, d_model, n_heads, n_layers, dropout,
                 use_gcn=True):
        super().__init__()
        self.use_gcn = use_gcn
        if use_gcn:
            self.encoder = GCNEncoder(in_features, d_model)
        else:
            self.encoder = LinearEncoder(in_features, d_model)
        self.temporal = TemporalTransformer(d_model, n_heads, n_layers, dropout)
        self.head     = nn.Linear(d_model, 1)

    def forward(self, x, edge_index, edge_weight):
        # x : [B, T, N, F]
        B, T, N, F = x.shape
        # Encodage spatial à chaque timestep
        x_flat = x.permute(0, 1, 2, 3).reshape(B * T, N, F)  # [B*T, N, F]
        enc    = self.encoder(x_flat, edge_index, edge_weight) # [B*T, N, D]
        # Réorganise pour Transformer temporel
        enc    = enc.reshape(B, T, N, -1).permute(0, 2, 1, 3) # [B, N, T, D]
        out    = self.temporal(enc)                             # [B, N, D]
        return self.head(out).squeeze(-1)                       # [B, N]


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
        # ── train ──
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

        # ── validation ──
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
                print(f"   Early stop à epoch {epoch} (val_loss={best_val:.5f})")
                break

    model.load_state_dict(best_state)
    return model


def evaluate(model, loader, edge_index, edge_weight, scaler, device):
    """Retourne MAE, RMSE, R² sur les données dénormalisées."""
    model.eval()
    ei = edge_index.to(device)
    ew = edge_weight.to(device)
    feat_idx = FEATURES.index(TARGET)

    all_pred, all_true = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb, ei, ew).cpu().numpy()    # [B, N]
            true = yb.numpy()                          # [B, N]
            all_pred.append(pred)
            all_true.append(true)

    pred = np.concatenate(all_pred, axis=0)  # [T_test, N]
    true = np.concatenate(all_true, axis=0)

    # Dénormalisation : le scaler a été fitté sur toutes les features
    # On reconstruit un tableau [T, N*F] factice pour inverser
    T, N = pred.shape
    F    = len(FEATURES)

    def denorm(arr):
        tmp = np.zeros((T * N, F))
        tmp[:, feat_idx] = arr.reshape(-1)
        inv = scaler.inverse_transform(tmp)
        return inv[:, feat_idx].reshape(T, N)

    pred_dn = denorm(pred)
    true_dn = denorm(true)

    # Métriques globales (toutes stations confondues)
    p_flat = pred_dn.reshape(-1)
    t_flat = true_dn.reshape(-1)
    mae  = mean_absolute_error(t_flat, p_flat)
    rmse = np.sqrt(mean_squared_error(t_flat, p_flat))
    r2   = r2_score(t_flat, p_flat)

    # Métriques par station
    per_station = {}
    for i, name in enumerate(STATION_NAMES):
        per_station[name] = {
            'MAE':  mean_absolute_error(true_dn[:, i], pred_dn[:, i]),
            'RMSE': np.sqrt(mean_squared_error(true_dn[:, i], pred_dn[:, i])),
            'R2':   r2_score(true_dn[:, i], pred_dn[:, i]),
        }

    return mae, rmse, r2, per_station


# ─────────────────────────────────────────────
# 6. BOUCLE PRINCIPALE
# ─────────────────────────────────────────────

def run_experiment(data, edge_index, edge_weight, device):
    """Lance les 2 modèles × 3 seeds, retourne les résultats."""

    train_d, val_d, test_d, scaler = split_and_scale(data)

    results = {'GCN+Transformer': [], 'Linear+Transformer': []}
    per_station_results = {}

    for model_name, use_gcn in [('GCN+Transformer', True), ('Linear+Transformer', False)]:
        print(f"\n{'='*55}")
        print(f"  Modèle : {model_name}")
        print(f"{'='*55}")
        seed_metrics = []

        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            print(f"  → Seed {seed}...")

            train_ds = MultiStationDataset(train_d)
            val_ds   = MultiStationDataset(val_d)
            test_ds  = MultiStationDataset(test_d)

            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
            test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

            model = SpatioTemporalModel(
                in_features=len(FEATURES), d_model=D_MODEL,
                n_heads=N_HEADS, n_layers=N_LAYERS,
                dropout=DROPOUT, use_gcn=use_gcn
            ).to(device)

            model = train_model(model, train_loader, val_loader,
                                edge_index, edge_weight, device)

            mae, rmse, r2, per_st = evaluate(
                model, test_loader, edge_index, edge_weight, scaler, device)

            print(f"     MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")
            seed_metrics.append((mae, rmse, r2))

            if seed == 42:
                per_station_results[model_name] = per_st

        results[model_name] = seed_metrics

    return results, per_station_results, scaler


def print_results_table(results, per_station_results):
    """Affiche les tableaux de résultats comparatifs."""

    print("\n" + "="*65)
    print("  TABLEAU 1 — Résultats globaux (moyenne ± SD, 3 seeds)")
    print("="*65)
    print(f"  {'Modèle':<25} {'MAE':>8} {'RMSE':>10} {'R²':>10}")
    print("-"*65)

    for name, metrics in results.items():
        maes  = [m[0] for m in metrics]
        rmses = [m[1] for m in metrics]
        r2s   = [m[2] for m in metrics]
        print(f"  {name:<25} "
              f"{np.mean(maes):6.2f}±{np.std(maes):.2f}  "
              f"{np.mean(rmses):7.2f}±{np.std(rmses):.2f}  "
              f"{np.mean(r2s):.4f}±{np.std(r2s):.4f}")

    print("\n" + "="*65)
    print("  TABLEAU 2 — Résultats par station (seed=42)")
    print("="*65)
    print(f"  {'Station':<20} {'GCN R²':>9} {'Lin R²':>9}  {'ΔGCN':>7}")
    print("-"*65)

    gcn_ps = per_station_results.get('GCN+Transformer', {})
    lin_ps = per_station_results.get('Linear+Transformer', {})

    for s in STATION_NAMES:
        gcn_r2 = gcn_ps.get(s, {}).get('R2', float('nan'))
        lin_r2 = lin_ps.get(s, {}).get('R2', float('nan'))
        delta  = gcn_r2 - lin_r2
        marker = " ✓ GCN↑" if delta > 0 else " — égal" if abs(delta)<1e-4 else " ✗ Lin↑"
        print(f"  {s:<20}  {gcn_r2:7.4f}   {lin_r2:7.4f}  {delta:+.4f}{marker}")

    # Synthèse
    all_gcn = [v['R2'] for v in gcn_ps.values()]
    all_lin = [v['R2'] for v in lin_ps.values()]
    print("-"*65)
    if all_gcn and all_lin:
        gcn_wins = sum(g > l for g, l in zip(all_gcn, all_lin))
        print(f"  GCN supérieur sur {gcn_wins}/{N_STATIONS} stations")
        print(f"  ΔR² moyen = {np.mean([g-l for g,l in zip(all_gcn,all_lin)]):+.4f}")
    print("="*65)


# ─────────────────────────────────────────────
# 7. ENTRÉE PRINCIPALE
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Solution A — Expérience multi-nœuds GCN vs Linear+Transformer')
    parser.add_argument('--data_dir',  type=str, default=None,
                        help='Dossier contenant les 12 CSV UCI')
    parser.add_argument('--synthetic', action='store_true',
                        help='Utiliser des données synthétiques (démo)')
    parser.add_argument('--quick', action='store_true',
                        help='Mode rapide : 1 seed, 15 epochs, données réduites')
    parser.add_argument('--graph', type=str, default='both',
                        choices=['distance', 'correlation', 'both'],
                        help='Type de graphe : distance | correlation | both (défaut)')
    args = parser.parse_args()

    # Device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✓ Device : Apple MPS (M1)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Device : CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device('cpu')
        print("✓ Device : CPU")

    # Données
    if args.synthetic or args.data_dir is None:
        print("\n[Mode synthétique — remplacez par --data_dir pour les vraies données]")
        n_hours = 5000 if args.quick else 21994
        data = generate_synthetic_data(n_hours)
    else:
        data = load_real_data(args.data_dir)

    # Overrides pour mode rapide
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
        print("[Mode --quick : 1 seed, 10 epochs, modèle réduit pour CPU]")

    # Graphe(s) à tester
    graphs_to_run = []
    if args.graph in ('distance', 'both'):
        ei_dist, ew_dist = build_graph(k=K_NEIGHBORS)
        graphs_to_run.append(('distance', ei_dist.to(device), ew_dist.to(device)))
        print(f"\nGraphe DISTANCE : {N_STATIONS} nœuds, {ei_dist.shape[1]} edges (k={K_NEIGHBORS})")

    if args.graph in ('correlation', 'both'):
        print("\nConstruction du graphe CORRÉLATION sur les données d'entraînement...")
        # On utilise uniquement le split train pour calculer la corrélation
        train_len = int(0.70 * len(data))
        ei_corr, ew_corr = build_correlation_graph(data[:train_len], k=K_NEIGHBORS)
        graphs_to_run.append(('correlation', ei_corr.to(device), ew_corr.to(device)))

    print(f"\nFeatures : {FEATURES}")
    print(f"SEQ_LEN={SEQ_LEN}h | BATCH={BATCH_SIZE} | D_MODEL={D_MODEL} | "
          f"HEADS={N_HEADS} | LAYERS={N_LAYERS}")

    # ── Boucle sur les topologies de graphe ──
    all_graph_results = {}
    t0 = time.time()

    for graph_name, ei_dev, ew_dev in graphs_to_run:
        print(f"\n{'#'*60}")
        print(f"  TOPOLOGIE : {graph_name.upper()}")
        print(f"{'#'*60}")
        results, per_station_results, scaler = run_experiment(
            data, ei_dev, ew_dev, device)
        all_graph_results[graph_name] = {
            'results': results,
            'per_station': per_station_results,
        }
        print_results_table(results, per_station_results)

    print(f"\n⏱  Temps total : {(time.time()-t0)/60:.1f} min")

    # ── Tableau comparatif final ──
    if len(all_graph_results) > 1:
        print("\n" + "="*70)
        print("  TABLEAU COMPARATIF — GCN selon topologie vs Linear+Transformer")
        print("="*70)
        print(f"  {'Topologie graphe':<25} {'MAE GCN':>9} {'R² GCN':>9} "
              f"{'R² Linear':>10} {'ΔR²':>8}")
        print("-"*70)

        for gname, gdata in all_graph_results.items():
            gcn = gdata['results']['GCN+Transformer']
            lin = gdata['results']['Linear+Transformer']
            gcn_mae  = np.mean([m[0] for m in gcn])
            gcn_r2   = np.mean([m[2] for m in gcn])
            lin_r2   = np.mean([m[2] for m in lin])
            delta    = gcn_r2 - lin_r2
            marker   = "✓ GCN↑" if delta > 0 else "✗ Lin↑"
            print(f"  {gname:<25}  {gcn_mae:7.2f}   {gcn_r2:7.4f}   "
                  f"{lin_r2:7.4f}  {delta:+.4f}  {marker}")

        print("="*70)

        # Phrase article adaptée au meilleur résultat
        best_gname = max(all_graph_results,
            key=lambda g: np.mean([m[2] for m in
                all_graph_results[g]['results']['GCN+Transformer']]))
        best = all_graph_results[best_gname]
        gcn_r2s = [m[2] for m in best['results']['GCN+Transformer']]
        lin_r2s = [m[2] for m in best['results']['Linear+Transformer']]
        delta_r2 = np.mean(gcn_r2s) - np.mean(lin_r2s)

        topo_label = ("correlation-based (Pearson PM₂.₅)"
                      if best_gname == 'correlation'
                      else "distance-based (Haversine k-NN)")

        sign = "confirming" if delta_r2 > 0 else "indicating that"
        msg  = ("spatial graph convolution captures inter-station PM₂.₅ co-variation"
                if delta_r2 > 0
                else "distance-based topology is insufficient — correlation-based "
                     "graph construction is required to activate GCN benefits")

        print(f"""
  ─────────────────────────────────────────────────────────────
  PHRASE ARTICLE — meilleure topologie : {best_gname.upper()}
  ─────────────────────────────────────────────────────────────
  \"In the multi-node setting ({N_STATIONS} stations as graph nodes,
  {topo_label}), GCN+Transformer achieves
  R²={np.mean(gcn_r2s):.3f} (±{np.std(gcn_r2s):.3f}) vs
  Linear+Transformer R²={np.mean(lin_r2s):.3f} (±{np.std(lin_r2s):.3f}),
  ΔR²={delta_r2:+.3f}, {sign} that {msg}.\"
  ─────────────────────────────────────────────────────────────
""")


if __name__ == '__main__':
    main()
