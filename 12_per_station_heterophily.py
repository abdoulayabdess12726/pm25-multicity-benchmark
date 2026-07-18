#!/usr/bin/env python3
"""
12_per_station_heterophily.py  —  Expérience A (hétérophilie locale par station)
==========================================================================
Pour chaque station des 3 villes :
    h_i = 1 - (corrélation Pearson moyenne de la station avec ses k=5 voisins,
               période TRAIN, PM2.5)

Le VOISINAGE est EXACTEMENT celui de la topologie corrélation de
06_train_multistation.py : on réutilise build_correlation_graph(data[:train_len],
k=5) pour obtenir, par station, l'ensemble exact des voisins (arêtes du graphe).
h_i utilise les valeurs de corrélation BRUTES (avant normalisation min-max).

Croisement avec results/per_station_seed_topology.csv (seed 42), séparément par
topologie (distance, correlation), sur les 27 stations :
    - Pearson r + p, Spearman rho + p
    - IC bootstrap 95 % (10000 resamples) sur r
    - régression linéaire (pente, ordonnée, R2)

Sorties :
    results/per_station_heterophily.csv
    figures/per_station_heterophily_scatter.{png,svg}
"""
import importlib.util
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
K = 5
SEED_PRIMARY = 42
CITY_COLORS = {"beijing": "#4C72B0", "london": "#C44E52", "madrid": "#55A868"}
CITY_LABELS = {"beijing": "Beijing", "london": "London", "madrid": "Madrid"}


def load_bench():
    spec = importlib.util.spec_from_file_location(
        "bench", str(ROOT / "06_train_multistation.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench"] = mod
    spec.loader.exec_module(mod)
    return mod


def heterophily_for_city(b, city):
    """Renvoie {station: (h_i, n_neighbors)}.

    Le voisinage reproduit EXACTEMENT la ligne de build_correlation_graph (06) :
        corr = np.corrcoef(pm25_train.T) ; np.fill_diagonal(corr, -inf)
        neighbors = np.argsort(corr[i])[::-1][:k_eff]     # top-k par corrélation
    c.-à-d. les mêmes k voisins que la topologie corrélation. Le seuil corr>0 de
    06 ne filtre que la création d'ARÊTES, pas la définition du voisinage, donc
    chaque station a bien k_eff voisins (h_i toujours défini)."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        if city == "beijing":
            ret = b.load_beijing_data(
                str(ROOT / "data/beijing_real/PRSA_Data_20130301-20170228"))
        elif city == "london":
            ret = b.load_london_data()
        else:
            ret = b.load_madrid_data()
        data = ret[0] if isinstance(ret, (tuple, list)) else ret
        data = np.asarray(data, dtype=np.float32)

    names = list(b.STATION_NAMES)
    train_len = int(0.70 * len(data))
    feat_idx = b.FEATURES.index("PM2.5")
    pm25 = data[:train_len][:, :, feat_idx]

    corr = np.corrcoef(pm25.T)                         # identique à 06
    np.fill_diagonal(corr, -np.inf)                    # exclut self (comme 06)
    k_eff = min(K, len(names) - 1)

    # NOTE : une station à variance nulle sur le train (ex. Madrid/MENDEZ ALVARO,
    # PM2.5 constant) produit des corrélations NaN. numpy argsort place les NaN
    # EN TÊTE en ordre décroissant ; 06 les élimine ensuite via son seuil corr>0.
    # On reproduit ce comportement : voisins = top-k parmi les corrélations VALIDES
    # (finies). Une station sans aucun voisin valide a un h_i indéfini (NaN).
    result = {}
    for i, name in enumerate(names):
        row = corr[i]
        valid = np.isfinite(row)                       # exclut -inf (self) et NaN
        if valid.sum() == 0:
            result[name] = (float("nan"), 0)
            continue
        masked = np.where(valid, row, -np.inf)
        neighbors = np.argsort(masked)[::-1][:min(k_eff, int(valid.sum()))]
        vals = row[neighbors]                           # corrélations brutes des voisins
        result[name] = (1.0 - float(vals.mean()), int(len(neighbors)))
    return result


def bootstrap_ci_r(x, y, n_boot=10000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    n = len(x)
    rs = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        rs[k] = np.corrcoef(x[idx], y[idx])[0, 1]
    alpha = (1 - ci) / 2
    return np.nanpercentile(rs, [100 * alpha, 100 * (1 - alpha)])


def main():
    b = load_bench()

    # 1. h_i par station (voisinage corrélation de 06)
    het, ncount = {}, {}
    for city in ["beijing", "london", "madrid"]:
        print(f"[{city}] calcul h_i (voisinage corrélation k={K}, train)...",
              file=sys.stderr)
        for name, (h_i, nn) in heterophily_for_city(b, city).items():
            het[(city, name)] = h_i
            ncount[(city, name)] = nn

    # 2. ΔR² seed 42 par topologie
    df = pd.read_csv(ROOT / "results/per_station_seed_topology.csv")
    d42 = df[df.seed == SEED_PRIMARY]
    piv = d42.pivot_table(index=["city", "station"], columns="topology",
                          values="delta_r2").reset_index()

    rows = []
    for _, r in piv.iterrows():
        key = (r["city"], r["station"])
        rows.append(dict(city=r["city"], station=r["station"],
                         h_i=het[key], n_neighbors=ncount[key],
                         delta_r2_distance=r["distance"],
                         delta_r2_correlation=r["correlation"]))
    out = pd.DataFrame(rows, columns=["city", "station", "h_i", "n_neighbors",
                                      "delta_r2_distance", "delta_r2_correlation"])
    out = out.sort_values(["city", "station"]).reset_index(drop=True)
    out.to_csv(ROOT / "results/per_station_heterophily.csv", index=False)

    # 3. Statistiques par topologie (sur les stations à h_i défini)
    ana = out.dropna(subset=["h_i"]).reset_index(drop=True)
    dropped = out[out["h_i"].isna()]
    print("=" * 82)
    print(f"  EXPÉRIENCE A — hétérophilie locale h_i vs ΔR²  (n={len(ana)} stations, seed 42)")
    print("=" * 82)
    if len(dropped):
        print(f"  Exclues (h_i indéfini, PM2.5 constant sur le train -> corrélations NaN) : "
              f"{len(dropped)}")
        for _, r in dropped.iterrows():
            print(f"    {r.city}/{r.station}")
        print(f"  -> n effectif = {len(ana)} (au lieu de {len(out)}).")
    stats_by_topo = {}
    for topo, col in [("distance", "delta_r2_distance"),
                      ("correlation", "delta_r2_correlation")]:
        h = ana["h_i"].values
        y = ana[col].values
        r_p, p_p = stats.pearsonr(h, y)
        rho_s, p_s = stats.spearmanr(h, y)
        ci_lo, ci_hi = bootstrap_ci_r(h, y)
        lin = stats.linregress(h, y)
        stats_by_topo[topo] = dict(r=r_p, p_r=p_p, rho=rho_s, p_rho=p_s,
                                   ci_lo=ci_lo, ci_hi=ci_hi, slope=lin.slope,
                                   intercept=lin.intercept, r2=lin.rvalue**2)
        print(f"\n  --- Topologie : {topo}  (ΔR² GCN−Linear, seed 42) ---")
        print(f"    Pearson  r   = {r_p:+.3f}   p = {p_p:.4g}   IC95% bootstrap = [{ci_lo:+.3f}, {ci_hi:+.3f}]")
        print(f"    Spearman rho = {rho_s:+.3f}   p = {p_s:.4g}")
        print(f"    Régression   : ΔR² = {lin.slope:+.4f}·h_i {lin.intercept:+.4f}   (R² = {lin.rvalue**2:.3f})")
        sig = "SIGNIFICATIVE" if p_p < 0.05 else "NON significative"
        print(f"    -> à n={len(ana)} : relation Pearson {sig} au seuil 0.05")

    # info voisinage
    print(f"\n  Voisinage : top-{K} par corrélation (train), reproduisant la ligne "
          f"'neighbors' de build_correlation_graph (NaN exclus).")
    nn_defined = out[out.h_i.notna()].n_neighbors
    print(f"    Voisins par station à h_i défini : {sorted(nn_defined.unique())} "
          f"(= k_eff = {min(K, 6)}). Stations à 0 voisin valide : "
          f"{int((out.n_neighbors == 0).sum())} (exclue).")

    # 4. Scatter plot
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11, "axes.linewidth": 0.8,
        "xtick.direction": "in", "ytick.direction": "in",
    })
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, (topo, col) in zip(axes, [("distance", "delta_r2_distance"),
                                      ("correlation", "delta_r2_correlation")]):
        for city in ["beijing", "london", "madrid"]:
            sub = ana[ana.city == city]
            ax.scatter(sub["h_i"], sub[col], s=34, alpha=0.85,
                       color=CITY_COLORS[city], edgecolor="white",
                       linewidth=0.5, label=CITY_LABELS[city], zorder=3)
        st = stats_by_topo[topo]
        xs = np.linspace(ana["h_i"].min(), ana["h_i"].max(), 100)
        ax.plot(xs, st["slope"] * xs + st["intercept"], color="0.35",
                lw=1.2, ls="--", zorder=2)
        ax.axhline(0, color="0.7", lw=0.7, zorder=1)
        ax.set_xlabel(r"Local heterophily $h_i = 1-\overline{\rho}_{\mathrm{neigh}}$")
        ax.set_ylabel(r"$\Delta R^2$ (GCN $-$ Linear)")
        ax.set_title(f"{topo.capitalize()} topology", fontsize=11)
        ax.annotate(f"Pearson $r={st['r']:+.2f}$ ($p={st['p_r']:.3f}$)\n"
                    f"Spearman $\\rho={st['rho']:+.2f}$",
                    xy=(0.04, 0.06), xycoords="axes fraction", fontsize=9,
                    va="bottom", ha="left")
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(ROOT / f"figures/per_station_heterophily_scatter.{ext}",
                    dpi=600, bbox_inches="tight")
    print(f"\n  Figure : figures/per_station_heterophily_scatter.{{png,svg}} (600 DPI)")
    print(f"  CSV    : results/per_station_heterophily.csv ({len(out)} lignes)")

    # police effectivement utilisée
    used = plt.rcParams["font.serif"][0]
    from matplotlib import font_manager as fmod
    have_tnr = "Times New Roman" in {f.name for f in fmod.fontManager.ttflist}
    print(f"  Police : {'Times New Roman' if have_tnr else 'fallback ' + used}")


if __name__ == "__main__":
    main()
