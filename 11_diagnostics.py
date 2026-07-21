#!/usr/bin/env python3
"""
11_diagnostics.py  —  Étape 7 (E4/E5) : contrôles diagnostiques
==========================================================================
(a) SHUFFLED-GRAPH (E4) : permutation d'arêtes préservant les degrés
    (double-edge swap sur le graphe réel), ré-entraîne le GCN-Transformer,
    compare ΔR² au graphe réel. Si ΔR²_shuffled ≈ ΔR²_real, la dégradation ne
    dépend pas de la structure spécifique ; si le graphe réel est pire, ce sont
    bien les arêtes hétérophiles réelles qui nuisent.
(b) NO-METEOROLOGY (E5) : PM2.5 seul (1 feature), GCN et Linear ré-entraînés,
    comparés au cas 5-features. Teste si le GCN s'appuie sur la météo.

3 villes, seed 42, 2 topologies. Protocole 06 identique (splits/scaling/early-
stopping). ΔR²_real (seed 42) réutilisé des JSON (pas de ré-entraînement).
Agrégat global sur TOUTES les stations (cohérent avec le benchmark canonique).

Fix mémoire MPS (empty_cache + gc entre entraînements) — obligatoire ici.

Sortie : results/diagnostics.csv
Usage : python 11_diagnostics.py --cities madrid   (puis london beijing)
"""
import argparse
import gc
import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
import pandas as pd

ROOT = Path(__file__).resolve().parent
SEED = 42
TOPOS = ["distance", "correlation"]
CSV = ROOT / "results" / "diagnostics.csv"


def _free_mps():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def load_bench():
    spec = importlib.util.spec_from_file_location(
        "bench", str(ROOT / "06_train_multistation.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench"] = mod
    spec.loader.exec_module(mod)
    return mod


def get_city(b, city):
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
    return dict(city=city, data=data, n_nodes=len(b.STATION_NAMES),
                pm=b.FEATURES.index("PM2.5"))


def real_graph(b, c, topo):
    with redirect_stdout(io.StringIO()):
        if topo == "distance":
            return b.build_graph(k=b.K_NEIGHBORS)
        train_len = int(0.70 * len(c["data"]))
        return b.build_correlation_graph(c["data"][:train_len], k=b.K_NEIGHBORS)


def shuffle_degree_preserving(edge_index, seed, n_factor=30):
    """Double-edge swap : préserve degré entrant ET sortant, randomise les cibles."""
    rng = np.random.default_rng(seed)
    src = edge_index[0].cpu().numpy().copy()
    dst = edge_index[1].cpu().numpy().copy()
    E = len(src)
    eset = set(zip(src.tolist(), dst.tolist()))
    for _ in range(n_factor * E):
        i, j = int(rng.integers(E)), int(rng.integers(E))
        a, b, cc, d = src[i], dst[i], src[j], dst[j]
        if i == j or a == d or cc == b:
            continue
        if (a, d) in eset or (cc, b) in eset:
            continue
        eset.discard((a, b)); eset.discard((cc, d))
        eset.add((a, d)); eset.add((cc, b))
        dst[i], dst[j] = d, b
    return torch.tensor(np.stack([src, dst]), dtype=torch.long)


def train_r2(b, data, ei, ew, encoder_type, seed, device):
    """Entraîne (gcn2 ou linear) au protocole 06 sur `data` (T,N,F) -> R² agrégé.
    Utilise bench.FEATURES courant (5 features, ou ['PM2.5'] pour no-meteo)."""
    torch.manual_seed(seed); np.random.seed(seed)
    with redirect_stdout(io.StringIO()):
        tr, va, te, sc = b.split_and_scale(data)
        tl = b.DataLoader(b.MultiStationDataset(tr), batch_size=b.BATCH_SIZE, shuffle=True)
        vl = b.DataLoader(b.MultiStationDataset(va), batch_size=b.BATCH_SIZE)
        tel = b.DataLoader(b.MultiStationDataset(te), batch_size=b.BATCH_SIZE)
        model = b.SpatioTemporalModel(
            in_features=len(b.FEATURES), d_model=b.D_MODEL, n_heads=b.N_HEADS,
            n_layers=b.N_LAYERS, dropout=b.DROPOUT, n_nodes=data.shape[1],
            encoder_type=encoder_type).to(device)
        model = b.train_model(model, tl, vl, ei, ew, device,
                              max_epochs=b.MAX_EPOCHS, patience=b.PATIENCE)
        _, _, r2, _ = b.evaluate(model, tel, ei, ew, sc, device)
    del model, tl, vl, tel
    _free_mps()
    return float(r2)


def real_from_json(city):
    g = json.loads((ROOT / f"results/{city}/multistation_results.json").read_text())["graphs"]
    out = {}
    for t in TOPOS:
        out[t] = dict(gcn=g[t]["GCN+Transformer"]["R2"][0],      # seed 42 = index 0
                      lin=g[t]["Linear+Transformer"]["R2"][0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", required=True,
                    choices=["beijing", "london", "madrid"])
    args = ap.parse_args()
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    b = load_bench()
    FEATURES_FULL = list(b.FEATURES)

    rows = []
    for city in args.cities:
        c = get_city(b, city)
        rj = real_from_json(city)
        # data PM2.5 seul (no-meteo)
        data_pm = c["data"][:, :, [c["pm"]]]
        print(f"\n[{city}] {c['n_nodes']} nœuds — diagnostics seed {SEED}", file=sys.stderr)

        # --- no-meteo Linear (indépendant de la topologie) ---
        b.FEATURES = ["PM2.5"]; b.TARGET = "PM2.5"
        ei0 = torch.zeros((2, 0), dtype=torch.long); ew0 = torch.zeros((0,))
        lin_nm = train_r2(b, data_pm, ei0, ew0, "linear", SEED, device)
        b.FEATURES = FEATURES_FULL; b.TARGET = "PM2.5"
        print(f"  no-meteo Linear R²={lin_nm:.4f}", file=sys.stderr)

        for topo in TOPOS:
            ei, ew = real_graph(b, c, topo)
            gcn_real, lin_real = rj[topo]["gcn"], rj[topo]["lin"]
            d_real = gcn_real - lin_real

            # (a) shuffled-graph : GCN 5-features sur graphe permuté (degrés préservés)
            ei_shuf = shuffle_degree_preserving(ei, SEED)
            gcn_shuf = train_r2(b, c["data"], ei_shuf.to(device), ew.to(device),
                                "gcn2", SEED, device)
            d_shuf = gcn_shuf - lin_real

            # (b) no-meteo GCN : 1 feature, graphe réel
            b.FEATURES = ["PM2.5"]; b.TARGET = "PM2.5"
            gcn_nm = train_r2(b, data_pm, ei.to(device), ew.to(device), "gcn2", SEED, device)
            b.FEATURES = FEATURES_FULL; b.TARGET = "PM2.5"
            d_nm = gcn_nm - lin_nm

            for exp, gr, lr, dr, nf in [
                ("real", gcn_real, lin_real, d_real, 5),
                ("shuffled_graph", gcn_shuf, lin_real, d_shuf, 5),
                ("no_meteorology", gcn_nm, lin_nm, d_nm, 1)]:
                rows.append(dict(city=city, topology=topo, experiment=exp, seed=SEED,
                                 n_features=nf, gcn_r2=round(gr, 4), lin_r2=round(lr, 4),
                                 delta_r2=round(dr, 4)))
            print(f"  {topo:12s}: ΔR² real={d_real:+.4f}  shuffled={d_shuf:+.4f}  "
                  f"no-meteo={d_nm:+.4f}", file=sys.stderr)

    new = pd.DataFrame(rows)
    if CSV.exists():
        old = pd.read_csv(CSV)
        old = old[~old.city.isin(args.cities)]
        full = pd.concat([old, new], ignore_index=True)
    else:
        full = new
    full = full.sort_values(["city", "topology", "experiment"]).reset_index(drop=True)
    full.to_csv(CSV, index=False)
    print(f"\nCSV : {CSV}  ({len(full)} lignes)", file=sys.stderr)


if __name__ == "__main__":
    main()
