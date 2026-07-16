#!/usr/bin/env python3
"""
graphs/export_adjacency.py
==========================================================================
Reconstruit et sauvegarde les 18 matrices d'adjacence utilisées dans le
papier (3 villes x 2 topologies x k in {3,5,8}) sous
    graphs/adjacency/{city}_{topology}_k{3|5|8}.npy

RÈGLE : réutilise EXACTEMENT les fonctions de construction de graphe de
06_train_multistation.py (build_graph pour la distance, build_correlation_graph
pour la corrélation) via importlib — aucune logique de graphe n'est redéfinie ici.

Rapport imprimé : k effectif par ville (plafonné à N-1 ?), self-loops oui/non,
symétrie de chaque matrice.
"""
import importlib.util
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "graphs" / "adjacency"
OUT.mkdir(parents=True, exist_ok=True)

KS = [3, 5, 8]
TOPOLOGIES = ["distance", "correlation"]


def load_bench():
    """Importe 06_train_multistation.py (nom commençant par un chiffre)."""
    spec = importlib.util.spec_from_file_location(
        "bench", str(ROOT / "06_train_multistation.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench"] = mod
    spec.loader.exec_module(mod)
    return mod


def to_dense(edge_index, edge_weight, n):
    """edge_index (2,E) + edge_weight (E,) -> matrice dense n x n (A[src,dst]=w)."""
    A = np.zeros((n, n), dtype=np.float64)
    ei = edge_index.cpu().numpy()
    ew = edge_weight.cpu().numpy()
    A[ei[0], ei[1]] = ew
    return A


def main():
    b = load_bench()
    loaders = {
        "beijing": lambda: b.load_beijing_data(
            str(ROOT / "data/beijing_real/PRSA_Data_20130301-20170228")),
        "london": b.load_london_data,
        "madrid": b.load_madrid_data,
    }

    rows = []
    for city, loader in loaders.items():
        buf = io.StringIO()
        with redirect_stdout(buf):                    # les loaders/graphes sont verbeux
            ret = loader()                            # met à jour les globals de bench
            data = ret[0] if isinstance(ret, (tuple, list)) else ret
            data = np.asarray(data, dtype=np.float32)
            n = b.N_STATIONS
            train_len = int(0.70 * len(data))         # identique à 06 (corrélation sur train)

            for k in KS:
                k_eff = min(k, n - 1)
                for topo in TOPOLOGIES:
                    if topo == "distance":
                        ei, ew = b.build_graph(k=k)
                    else:
                        ei, ew = b.build_correlation_graph(data[:train_len], k=k)
                    A = to_dense(ei, ew, n)
                    fname = f"{city}_{topo}_k{k}.npy"
                    np.save(OUT / fname, A)

                    n_edges = int(ei.shape[1])
                    self_loops = bool(np.any(np.diag(A) != 0.0))
                    symmetric = bool(np.allclose(A, A.T, atol=1e-6))
                    rows.append(dict(
                        city=city, n_stations=n, topology=topo, k_nominal=k,
                        k_eff=k_eff, capped=(k > n - 1), n_edges=n_edges,
                        avg_out_degree=round(n_edges / n, 3),
                        self_loops=self_loops, symmetric=symmetric, file=fname))

    # ── Rapport ──
    print("=" * 96)
    print("  EXPORT DES MATRICES D'ADJACENCE — 18 fichiers (.npy) dans graphs/adjacency/")
    print("=" * 96)
    hdr = ("city", "N", "topology", "k", "k_eff", "capped", "edges",
           "out_deg", "self_loops", "symmetric")
    print(f"  {hdr[0]:<8}{hdr[1]:<4}{hdr[2]:<12}{hdr[3]:<3}{hdr[4]:<6}"
          f"{hdr[5]:<8}{hdr[6]:<7}{hdr[7]:<9}{hdr[8]:<12}{hdr[9]}")
    print("-" * 96)
    for r in rows:
        print(f"  {r['city']:<8}{r['n_stations']:<4}{r['topology']:<12}"
              f"{r['k_nominal']:<3}{r['k_eff']:<6}{str(r['capped']):<8}"
              f"{r['n_edges']:<7}{r['avg_out_degree']:<9}"
              f"{str(r['self_loops']):<12}{str(r['symmetric'])}")
    print("-" * 96)
    print(f"  TOTAL : {len(rows)} matrices sauvegardées.")
    print(f"  Self-loops présents dans une matrice ? "
          f"{any(r['self_loops'] for r in rows)}")
    print(f"  Matrices symétriques ? "
          f"{sorted(set(r['symmetric'] for r in rows))}  (k-NN dirigé => attendu False)")


if __name__ == "__main__":
    main()
