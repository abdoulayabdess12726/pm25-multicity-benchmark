#!/usr/bin/env python3
"""
14_sota_baselines.py — SOTA GNN spatiotemporelles : STGCN + Graph WaveNet
==========================================================================
R1.11 / R2.1 / R2.2 : comparaison head-to-head avec des architectures GNN
spatiotemporelles reconnues, ajoutées à la Table 3.

PÉRIMÈTRE STRICT : 2 modèles (STGCN, Graph WaveNet) × 3 villes × seed 42
UNIQUEMENT = 6 entraînements. Pas de multi-seed (coût de calcul ; déclaré
dans l'article : "SOTA baselines reported at the primary seed given their
computational cost").

POINT DE PROTOCOLE CRITIQUE — Madrid = 7 stations, MENDEZ ALVARO INCLUSE.
Le manuscrit §3.3 la conserve dans le benchmark de prévision (exclue
seulement de l'analyse station-level §5.5). Vérifié empiriquement : la
Table 3 originale (results/madrid/multistation_results.json) utilise déjà
n_stations=7 partout (Linear-Transformer R2_mean=0.814 sur 7 stations =
reconstruit exactement le ΔR² Madrid/distance = -0.321 du papier). Les
scripts E6/E8 (k-sensitivity) excluaient MENDEZ ALVARO — c'est un écart au
protocole publié propre à CES scripts, PAS reproduit ici.
MENDEZ ALVARO a un PM2.5 constant sur le train (std=0.0) => corrélation
indéfinie => le graphe de corrélation l'isole naturellement (0 arête, ni
entrante ni sortante) — comportement déjà présent dans le GCN-Transformer
canonique de la Table 3, pas une anomalie introduite ici.

ARCHITECTURES — implémentations de référence citées, pas de l'from-scratch
approximatif :
  - STGCN (Yu, Yin & Zhu, IJCAI 2018, "Spatio-Temporal Graph Convolutional
    Networks: A Deep Learning Framework for Traffic Forecasting") : bloc
    ST-Conv "sandwich" TemporalGatedConv -> ChebGraphConv -> TemporalGatedConv,
    2 blocs empilés + couche de sortie temporelle + FC. La convolution
    spectrale de Chebyshev suit Defferrard, Bresson & Vandergheynst (NeurIPS
    2016). Le Laplacien de Chebyshev exige un graphe NON DIRIGÉ : l'adjacence
    corrélation (k-NN, potentiellement asymétrique) est symétrisée (A+Aᵀ)/2
    avant normalisation — pratique standard pour les GCN spectraux sur un
    graphe candidat non garanti symétrique.
  - Graph WaveNet (Wu, Pan, Long, Jiang & Zhang, IJCAI 2019, "Graph WaveNet
    for Deep Spatial-Temporal Graph Modeling") : TCN dilatée gated (style
    WaveNet) + convolution de diffusion (Li, Yu, Shahabi & Liu, ICLR 2018,
    "DCRNN" — supports avant/arrière) sur l'adjacence prédéfinie PLUS
    adjacence ADAPTATIVE apprise A_adp = softmax(ReLU(E1 E2ᵀ)), configuration
    standard du modèle. Skip connections + empilement dilaté (1,2,1,2).
  Seule adaptation vs. papiers originaux : sortie horizon=1 (notre tâche)
  au lieu du seq2seq multi-step natif — gating, dilatations, conv
  spectrale/diffusion et adjacence adaptative sont fidèles aux papiers.

Adjacence prédéfinie = topologie CORRÉLATION du papier principal
(b.build_correlation_graph, k=K_NEIGHBORS=5, IDENTIQUE à 06_train_multistation.py).

PROTOCOLE DONNÉES — identique à 06 (hyperparams via b.*) :
splits chronologiques 70/15/15, 5 features, SEQ_LEN=24, horizon 1h, MinMax
fit train uniquement, cibles test[24:], R² agrégé dénormalisé (pas
per-station), MAX_EPOCHS/PATIENCE = b.MAX_EPOCHS/b.PATIENCE, device=cpu.

RÈGLES D'ARRÊT :
  - Madrid avec n_nodes != 7 -> exception (garde-fou protocole).
  - Beijing R2 < 0.75 -> exception ("bug de pipeline, pas un résultat" —
    Beijing est homophile, STGCN/Graph WaveNet doivent y être compétitifs).
  - NaN / non-convergence -> exception.
  - R2 > Linear-Transformer en régime hétérophile (London/Madrid) : PAS une
    exception (résultat scientifique valide, à rapporter tel quel) — juste
    un flag explicite `beats_linear` dans la sortie.

Sortie : results/table3_sota_baselines.csv
  (city, model, seed, R2_aggregate, r2_linear_ref, beats_linear,
   n_edges, n_isolated_nodes)

Usage : python 14_sota_baselines.py --city beijing --model stgcn --seed 42 --cpu
"""
import argparse
import io
import resource
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "results" / "table3_sota_baselines.csv"
EXPECTED_N = {"beijing": 12, "london": 8, "madrid": 7}


def load_bench():
    import importlib.util
    spec = importlib.util.spec_from_file_location("bench", str(ROOT / "06_train_multistation.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench"] = mod
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────────────────
# STGCN (Yu, Yin & Zhu, IJCAI 2018) — Chebyshev spectral graph conv
# (Defferrard, Bresson & Vandergheynst, NeurIPS 2016)
# ─────────────────────────────────────────────────────────────────────────

class ChebGraphConv(nn.Module):
    """Convolution spectrale de Chebyshev, ordre K. Graphe supposé non dirigé
    (symétrisé en amont par l'appelant via set_graph)."""

    def __init__(self, in_channels, out_channels, K=3):
        super().__init__()
        self.K = K
        self.theta = nn.Parameter(torch.empty(K, in_channels, out_channels))
        nn.init.xavier_uniform_(self.theta)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self._L_tilde = None

    def set_graph(self, edge_index, edge_weight, n_nodes, device):
        A = torch.zeros(n_nodes, n_nodes, device=device)
        A[edge_index[0], edge_index[1]] = edge_weight
        A = (A + A.t()) / 2                              # symétrisation (Chebyshev = graphe non dirigé)
        deg = A.sum(1)
        deg_inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        L = torch.eye(n_nodes, device=device) - D_inv_sqrt @ A @ D_inv_sqrt
        lambda_max = torch.linalg.eigvalsh(L).max().clamp(min=1e-4)
        self._L_tilde = (2.0 / lambda_max) * L - torch.eye(n_nodes, device=device)

    def forward(self, x):
        # x: (B, N, Cin) -> (B, N, Cout)
        Tk_2, Tk_1, out = None, None, 0
        for k in range(self.K):
            if k == 0:
                Tk = x
            elif k == 1:
                Tk = torch.einsum('nm,bmc->bnc', self._L_tilde, x)
            else:
                Tk = 2 * torch.einsum('nm,bmc->bnc', self._L_tilde, Tk_1) - Tk_2
            out = out + torch.einsum('bnc,cd->bnd', Tk, self.theta[k])
            Tk_2, Tk_1 = Tk_1, Tk
        return out + self.bias


class TemporalGatedConv(nn.Module):
    """Convolution temporelle causale gated (GLU), bloc ST-Conv de Yu et al. 2018."""

    def __init__(self, in_channels, out_channels, kt=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=(kt, 1))

    def forward(self, x):
        # x: (B, C, T, N)
        p, q = self.conv(x).chunk(2, dim=1)
        return p * torch.sigmoid(q)


class STConvBlock(nn.Module):
    """Bloc ST-Conv sandwich : TemporalGatedConv -> ChebGraphConv -> TemporalGatedConv."""

    def __init__(self, c_in, c_mid, c_out, n_nodes, kt=3, ks=3, dropout=0.1):
        super().__init__()
        self.n_nodes = n_nodes
        self.tconv1 = TemporalGatedConv(c_in, c_mid, kt)
        self.gconv = ChebGraphConv(c_mid, c_mid, K=ks)
        self.relu = nn.ReLU()
        self.tconv2 = TemporalGatedConv(c_mid, c_out, kt)
        self.norm = nn.LayerNorm(c_out)
        self.drop = nn.Dropout(dropout)

    def set_graph(self, edge_index, edge_weight, device):
        self.gconv.set_graph(edge_index, edge_weight, self.n_nodes, device)

    def forward(self, x):
        h = self.tconv1(x)                                # (B, c_mid, T-2, N)
        B, C, T, N = h.shape
        h = h.permute(0, 2, 3, 1).reshape(B * T, N, C)
        h = self.relu(self.gconv(h))
        h = h.reshape(B, T, N, C).permute(0, 3, 1, 2)
        h = self.tconv2(h)                                # (B, c_out, T-2, N)
        h = h.permute(0, 2, 3, 1)
        h = self.drop(self.norm(h))
        return h.permute(0, 3, 1, 2)


class STGCNModel(nn.Module):
    """STGCN (Yu, Yin & Zhu, IJCAI 2018) — 2 blocs ST-Conv + couche de sortie.
    SEQ_LEN=24 : bloc1 (24->20), bloc2 (20->16), out_tconv(kernel=16) -> T=1."""

    def __init__(self, in_features, n_nodes, kt=3, ks=3, dropout=0.1):
        super().__init__()
        self.block1 = STConvBlock(in_features, 32, 64, n_nodes, kt, ks, dropout)
        self.block2 = STConvBlock(64, 32, 64, n_nodes, kt, ks, dropout)
        self.out_tconv = nn.Conv2d(64, 64, kernel_size=(16, 1))
        self.fc = nn.Linear(64, 1)
        self._graph_set = False

    def _ensure_graph(self, edge_index, edge_weight, device):
        if not self._graph_set:
            self.block1.set_graph(edge_index, edge_weight, device)
            self.block2.set_graph(edge_index, edge_weight, device)
            self._graph_set = True

    def forward(self, x, edge_index, edge_weight):
        # x: (B, T, N, F)
        self._ensure_graph(edge_index, edge_weight, x.device)
        h = x.permute(0, 3, 1, 2)          # (B, F, T, N)
        h = self.block1(h)
        h = self.block2(h)
        h = self.out_tconv(h)              # (B, 64, 1, N)
        h = h.squeeze(2).permute(0, 2, 1)  # (B, N, 64)
        return self.fc(h).squeeze(-1)      # (B, N)


# ─────────────────────────────────────────────────────────────────────────
# Graph WaveNet (Wu, Pan, Long, Jiang & Zhang, IJCAI 2019) — diffusion conv
# (Li, Yu, Shahabi & Liu, ICLR 2018 "DCRNN") + adjacence adaptative apprise
# ─────────────────────────────────────────────────────────────────────────

class DiffusionGraphConv(nn.Module):
    """Convolution de diffusion sur plusieurs supports (avant/arrière/adaptatif),
    ordre K (Li et al., 2018 ; réutilisée par Graph WaveNet)."""

    def __init__(self, c_in, c_out, n_supports, order=2, dropout=0.1):
        super().__init__()
        self.order = order
        total = c_in * (order * n_supports + 1)           # +1 : terme d'ordre 0 (x lui-même)
        self.mlp = nn.Conv2d(total, c_out, kernel_size=(1, 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, supports):
        # x: (B, C, T, N) ; supports : liste de matrices de transition (N, N)
        out = [x]
        for A in supports:
            x_k = torch.einsum('nm,bctm->bctn', A, x)
            out.append(x_k)
            for _ in range(2, self.order + 1):
                x_k = torch.einsum('nm,bctm->bctn', A, x_k)
                out.append(x_k)
        h = torch.cat(out, dim=1)
        return self.drop(self.mlp(h))


class GatedDilatedConv(nn.Module):
    """TCN dilatée gated, style WaveNet : tanh(filter) * sigmoid(gate)."""

    def __init__(self, channels, kernel_size=2, dilation=1):
        super().__init__()
        self.filter_conv = nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), dilation=(dilation, 1))
        self.gate_conv = nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), dilation=(dilation, 1))

    def forward(self, x):
        return torch.tanh(self.filter_conv(x)) * torch.sigmoid(self.gate_conv(x))


class GraphWaveNetModel(nn.Module):
    """Graph WaveNet (Wu et al., IJCAI 2019). Supports prédéfinis (avant/arrière,
    topologie corrélation, PAS symétrisés — la diffusion conv gère nativement
    les graphes dirigés) + adjacence adaptative A_adp = softmax(ReLU(E1 E2ᵀ)).
    Sortie horizon=1 (adaptation ; cœur spatio-temporel fidèle au papier)."""

    def __init__(self, in_features, n_nodes, residual_channels=32, skip_channels=64,
                 end_channels=64, emb_dim=10, dilations=(1, 2, 1, 2), kernel_size=2,
                 order=2, dropout=0.1):
        super().__init__()
        self.n_nodes = n_nodes
        self.start_conv = nn.Conv2d(in_features, residual_channels, kernel_size=(1, 1))
        self.E1 = nn.Parameter(torch.randn(n_nodes, emb_dim) * 0.1)
        self.E2 = nn.Parameter(torch.randn(n_nodes, emb_dim) * 0.1)
        self.gated_convs = nn.ModuleList()
        self.gconvs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        for d in dilations:
            self.gated_convs.append(GatedDilatedConv(residual_channels, kernel_size, d))
            self.gconvs.append(DiffusionGraphConv(residual_channels, residual_channels, n_supports=3,
                                                  order=order, dropout=dropout))
            self.skip_convs.append(nn.Conv2d(residual_channels, skip_channels, kernel_size=(1, 1)))
            self.residual_convs.append(nn.Conv2d(residual_channels, residual_channels, kernel_size=(1, 1)))
        self.end_conv1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1))
        self.end_conv2 = nn.Conv2d(end_channels, 1, kernel_size=(1, 1))
        self._Pf, self._Pb = None, None

    def _static_supports(self, edge_index, edge_weight, device):
        if self._Pf is None:
            N = self.n_nodes
            A = torch.zeros(N, N, device=device)
            A[edge_index[0], edge_index[1]] = edge_weight
            self._Pf = A / A.sum(1, keepdim=True).clamp(min=1e-8)
            AT = A.t()
            self._Pb = AT / AT.sum(1, keepdim=True).clamp(min=1e-8)
        return self._Pf, self._Pb

    def forward(self, x, edge_index, edge_weight):
        # x: (B, T, N, F)
        Pf, Pb = self._static_supports(edge_index, edge_weight, x.device)
        A_adp = torch.softmax(torch.relu(self.E1 @ self.E2.t()), dim=1)
        supports = [Pf, Pb, A_adp]

        h = self.start_conv(x.permute(0, 3, 1, 2))         # (B, C, T, N)
        skip_sum = None
        for gate, gconv, skc, resc in zip(self.gated_convs, self.gconvs,
                                          self.skip_convs, self.residual_convs):
            residual = h
            fg = gate(h)                                    # (B, C, T', N), T' < T (causal, sans padding)
            s = skc(fg)
            skip_sum = s if skip_sum is None else skip_sum[..., -s.shape[2]:, :] + s
            g = gconv(fg, supports)
            h = resc(g) + residual[..., -g.shape[2]:, :]

        out = torch.relu(skip_sum)
        out = torch.relu(self.end_conv1(out))
        out = self.end_conv2(out)                           # (B, 1, T'', N)
        return out[:, :, -1, :].squeeze(1)                  # (B, N) — dernier pas causal = horizon 1


# ─────────────────────────────────────────────────────────────────────────
# Pipeline données / entraînement — protocole identique à 06 (hyperparams b.*)
# ─────────────────────────────────────────────────────────────────────────

def get_city(b, city):
    with redirect_stdout(io.StringIO()):
        if city == "beijing":
            ret = b.load_beijing_data(str(ROOT / "data/beijing_real/PRSA_Data_20130301-20170228"))
        elif city == "london":
            ret = b.load_london_data()
        else:
            ret = b.load_madrid_data()
        data = ret[0] if isinstance(ret, (tuple, list)) else ret
        data = np.asarray(data, dtype=np.float32)
        train_d, val_d, test_d, scaler = b.split_and_scale(data)
    names = list(b.STATION_NAMES)
    n_nodes = len(names)
    if n_nodes != EXPECTED_N[city]:                        # STOP : garde-fou protocole (MENDEZ ALVARO incluse)
        raise RuntimeError(f"{city}: n_nodes={n_nodes} != attendu {EXPECTED_N[city]} "
                           f"(Madrid doit avoir 7 stations, MENDEZ ALVARO incluse)")
    pm = b.FEATURES.index("PM2.5")
    Y = data[int(0.85 * len(data)):][b.SEQ_LEN:, :, pm]     # cibles test dénormalisées, TOUTES stations
    ss_tot = float(((Y - Y.mean()) ** 2).sum())
    return dict(city=city, data=data, names=names, pm=pm, n_nodes=n_nodes,
                Y=Y, ss_tot=ss_tot, scaler=scaler,
                train_d=train_d, val_d=val_d, test_d=test_d)


def ref_linear_r2_seed42(b, c):
    """R² seed-42 du Linear-Transformer canonique (Table 3), TOUTES stations
    (Madrid = 7, MENDEZ ALVARO incluse) — référence pour beats_linear."""
    import json
    js = json.loads((ROOT / f"results/{c['city']}/multistation_results.json").read_text())
    if js["station_names"] != c["names"]:
        raise RuntimeError(f"{c['city']}: stations JSON != loader")
    psa = js["graphs"]["distance"]["per_station_all_seeds"]["Linear+Transformer"].get("42")
    if psa is None:
        raise RuntimeError(f"{c['city']}: Linear-Transformer seed 42 introuvable dans le JSON")
    n = c["Y"].shape[0]
    ss_res = sum(psa[s]["RMSE"] ** 2 * n for s in c["names"])
    return 1.0 - ss_res / c["ss_tot"]


def count_isolated(edge_index, n_nodes):
    deg = np.zeros(n_nodes)
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    np.add.at(deg, src, 1)
    np.add.at(deg, dst, 1)
    return int((deg == 0).sum())


def upsert_row(row):
    cols = ["city", "model", "seed", "R2_aggregate", "r2_linear_ref", "beats_linear",
            "n_edges", "n_isolated_nodes"]
    df = pd.read_csv(CSV) if CSV.exists() else pd.DataFrame(columns=cols)
    m = (df.city == row["city"]) & (df.model == row["model"]) & (df.seed == row["seed"])
    df = df[~m]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = df.sort_values(["city", "model", "seed"]).reset_index(drop=True)
    df = df[cols]
    df.to_csv(CSV, index=False)
    return len(df)


def build_model(model_name, in_features, n_nodes):
    if model_name == "stgcn":
        return STGCNModel(in_features=in_features, n_nodes=n_nodes)
    elif model_name == "graphwavenet":
        return GraphWaveNetModel(in_features=in_features, n_nodes=n_nodes)
    raise ValueError(f"modèle inconnu: {model_name}")


def run_single(city, model_name, seed=42, cpu=True):
    t0 = time.time()
    device = "cpu"
    if not cpu:
        raise RuntimeError("Protocole : device=cpu imposé, pas de MPS pour ces runs SOTA.")
    torch.manual_seed(seed); np.random.seed(seed)

    b = load_bench()
    c = get_city(b, city)
    with redirect_stdout(io.StringIO()):
        ei, ew = b.build_correlation_graph(c["data"][:int(0.70 * len(c["data"]))], k=b.K_NEIGHBORS)
    n_edges = int(ei.shape[1])
    n_isolated = count_isolated(ei, c["n_nodes"])
    lin_ref = ref_linear_r2_seed42(b, c)

    model = build_model(model_name, in_features=len(b.FEATURES), n_nodes=c["n_nodes"]).to(device)
    tl = b.DataLoader(b.MultiStationDataset(c["train_d"]), batch_size=b.BATCH_SIZE, shuffle=True)
    vl = b.DataLoader(b.MultiStationDataset(c["val_d"]), batch_size=b.BATCH_SIZE)
    te = b.DataLoader(b.MultiStationDataset(c["test_d"]), batch_size=b.BATCH_SIZE)
    eid, ewd = ei.to(device), ew.to(device)

    model = b.train_model(model, tl, vl, eid, ewd, device,
                          max_epochs=b.MAX_EPOCHS, patience=b.PATIENCE)
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in te:
            preds.append(model(xb.to(device), eid, ewd).cpu().numpy())
    P = np.concatenate(preds)
    lo, hi = c["scaler"].data_min_[c["pm"]], c["scaler"].data_max_[c["pm"]]
    P = P * (hi - lo) + lo
    if P.shape != c["Y"].shape:                             # STOP : alignement cibles
        raise RuntimeError(f"{city}: pred {P.shape} != cibles {c['Y'].shape}")

    r2 = r2_score(c["Y"].reshape(-1), P.reshape(-1))
    if not np.isfinite(r2):                                  # STOP : NaN / non-convergence
        raise RuntimeError(f"{city}/{model_name}: R² non fini ({r2})")
    if city == "beijing" and r2 < 0.75:                       # STOP : bug pipeline suspecté
        raise RuntimeError(f"{city}/{model_name}: R²={r2:.4f} anormalement bas pour une ville "
                           f"homophile — bug de pipeline suspecté, PAS un résultat")

    beats_linear = bool(r2 > lin_ref)
    row = dict(city=city, model=model_name, seed=seed, R2_aggregate=round(r2, 4),
               r2_linear_ref=round(lin_ref, 4), beats_linear=beats_linear,
               n_edges=n_edges, n_isolated_nodes=n_isolated)
    ntot = upsert_row(row)

    dt = time.time() - t0
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)
    flag = "  *** BAT LE LINEAR-TRANSFORMER — RÉSULTAT SCIENTIFIQUE À RAPPORTER TEL QUEL ***" if beats_linear else ""
    print(f"[{city}/{model_name}/seed={seed}] device={device} n_nodes={c['n_nodes']} "
          f"(MENDEZ ALVARO {'incluse' if city=='madrid' else 'n/a'}) edges={n_edges} "
          f"isolated={n_isolated} R2={r2:.4f} Linear_ref={lin_ref:.4f}{flag}", file=sys.stderr)
    print(f"DUREE_S={dt:.0f} DUREE_MIN={dt/60:.1f} RSS_GB={rss_gb:.2f} CSV_LIGNES={ntot}",
          file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", choices=["beijing", "london", "madrid"], required=True)
    ap.add_argument("--model", choices=["stgcn", "graphwavenet"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true", default=True)
    args = ap.parse_args()
    run_single(args.city, args.model, seed=args.seed, cpu=args.cpu)


if __name__ == "__main__":
    main()
