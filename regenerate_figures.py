#!/usr/bin/env python3
"""
regenerate_figures.py — P10 : régénère les figures DE DONNÉES du manuscrit
en vectoriel (PDF + SVG), depuis raw_results.csv.

⚠️ Le manuscrit a 5 figures numérotées, mais seules 3 sont des graphiques de
données : Figure 3 (h(D) vs ΔR², 3→4 réseaux), Figure 4 (per-station h_i vs
ΔR²), Figure 5 (courbe de pruning). Figures 1 et 2 sont des illustrations
conceptuelles / un schéma de pipeline — rien à régénérer depuis
raw_results.csv, elles n'ont pas de données sous-jacentes. Ce script ne
produit donc que 3 figures, pas 5 — signalé explicitement, pas de figures
fantômes générées pour faire le compte.

Gabarit IJIES (source : (Ver. 2025.5.18) IJIES_Format (2).docx, A4,
marges 0.75in, 2 colonnes, espace inter-colonnes 0.3in) :
  - pleine largeur (2 colonnes)  : 6.77 in
  - une colonne                 : 3.235 in
Police ≥9pt à la taille FINALE (vectoriel : la taille de police déclarée
est la taille imprimée, aucune perte au rendu PNG).

h_i (hétérophilie locale par station) n'est pas dans raw_results.csv par
nature (propriété des séries PM2.5 brutes, pas une sortie de modèle) —
recalculé ici à l'identique de analysis/build_per_station_dataset.py /
12_per_station_heterophily.py (même méthode sûre, masquage NaN avant tri).
ΔR² vient exclusivement de raw_results.csv.

Sorties : figures/fig3_h_index_vs_delta_r2.{pdf,svg}
          figures/fig4_per_station_heterophily.{pdf,svg}
          figures/fig5_edge_pruning_curve.{pdf,svg}
"""
import importlib.util
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.stats import agg_mean_std  # noqa: E402

# --------------------------------------------------------------------------- #
# Style commun aux 3 figures
# --------------------------------------------------------------------------- #
FULL_WIDTH = 6.77     # in, pleine largeur 2 colonnes (gabarit IJIES)
COL_WIDTH = 3.235     # in, une colonne
FONT_SIZE = 9          # pt, taille finale (≥9pt exigé par le relecteur 1)
K_LOCAL = 5

CITIES = ["beijing", "london", "madrid", "czt"]
CITY_LABELS = {"beijing": "Beijing", "london": "London", "madrid": "Madrid", "czt": "Chang-Zhu-Tan"}
# Palette : colorblind-safe (Okabe-Ito), cohérente sur les 3 figures.
CITY_COLORS = {
    "beijing": "#0072B2",   # bleu
    "london": "#D55E00",    # orange/rouge
    "madrid": "#009E73",    # vert
    "czt": "#CC79A7",       # rose/violet
}
CITY_MARKERS = {"beijing": "o", "london": "s", "madrid": "^", "czt": "D"}
H_INDEX = {"beijing": 0.497, "london": 0.656, "madrid": 0.728, "czt": 0.413}


def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "axes.linewidth": 0.8,
        "xtick.direction": "in", "ytick.direction": "in",
        "svg.fonttype": "none",   # texte éditable dans le SVG, pas des tracés
        "pdf.fonttype": 42,       # police intégrée en TrueType, pas en bitmap
    })


def save_fig(fig, name):
    for ext in ("pdf", "svg"):
        fig.savefig(ROOT / "figures" / f"{name}.{ext}", bbox_inches="tight")
    print(f"  écrit : figures/{name}.pdf + .svg")


# --------------------------------------------------------------------------- #
# h_i par station — identique à analysis/build_per_station_dataset.py
# --------------------------------------------------------------------------- #
def load_bench():
    spec = importlib.util.spec_from_file_location("bench", str(ROOT / "06_train_multistation.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench"] = mod
    with redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


def heterophily_for_city(b, city):
    with redirect_stdout(io.StringIO()):
        if city == "beijing":
            ret = b.load_beijing_data(str(ROOT / "data/beijing_real/PRSA_Data_20130301-20170228"))
        elif city == "london":
            ret = b.load_london_data()
        elif city == "madrid":
            ret = b.load_madrid_data()
        else:
            ret = b.load_czt_data()
        data = ret[0] if isinstance(ret, (tuple, list)) else ret
        data = np.asarray(data, dtype=np.float32)
    names = list(b.STATION_NAMES)
    train_len = int(0.70 * len(data))
    feat_idx = b.FEATURES.index("PM2.5")
    pm25 = data[:train_len][:, :, feat_idx]
    corr = np.corrcoef(pm25.T)
    np.fill_diagonal(corr, -np.inf)
    k_eff = min(K_LOCAL, len(names) - 1)
    result = {}
    for i, name in enumerate(names):
        row = corr[i]
        valid = np.isfinite(row)
        if valid.sum() == 0:
            result[name] = float("nan")
            continue
        masked = np.where(valid, row, -np.inf)
        neighbors = np.argsort(masked)[::-1][:min(k_eff, int(valid.sum()))]
        result[name] = 1.0 - float(row[neighbors].mean())
    return result


def load_raw_results():
    df = pd.read_csv(ROOT / "results" / "raw_results.csv", dtype=str)
    df["r2"] = pd.to_numeric(df["r2"], errors="coerce")
    df["k"] = pd.to_numeric(df["k"], errors="coerce")
    df["variant"] = df["variant"].fillna("")
    df["topology"] = df["topology"].fillna("")
    return df


def gcn_lin_delta_3seed(df, city, topo):
    agg = df[df.station == "__aggregate__"]
    gcn = agg[(agg.city == city) & (agg.model == "GCN-Transformer") & (agg.topology == topo)
             & (agg.k == 5) & (agg.variant == "") & (agg.seed.isin(["42", "123", "777"]))]
    lin = agg[(agg.city == city) & (agg.model == "Linear-Transformer") & (agg.variant == "")
             & (agg.seed.isin(["42", "123", "777"]))].drop_duplicates(subset=["seed"])
    if len(gcn) != 3 or len(lin) != 3:
        return None
    gm, gs = agg_mean_std(gcn.r2.tolist())
    lm, ls = agg_mean_std(lin.r2.tolist())
    return gm - lm, np.sqrt(gs**2 + ls**2)


# --------------------------------------------------------------------------- #
# Figure 3 — h(D) vs ΔR², 4 réseaux, 2 panneaux topologie
# --------------------------------------------------------------------------- #
def fig3(df):
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, FULL_WIDTH * 0.42))
    for ax, topo in zip(axes, ["distance", "correlation"]):
        hs, ds, es = [], [], []
        for city in CITIES:
            r = gcn_lin_delta_3seed(df, city, topo)
            if r is None:
                continue
            delta, err = r
            h = H_INDEX[city]
            hs.append(h); ds.append(delta); es.append(err)
            ax.errorbar(h, delta, yerr=err, fmt=CITY_MARKERS[city], color=CITY_COLORS[city],
                       markersize=6, capsize=3, markeredgecolor="white", markeredgewidth=0.6,
                       label=CITY_LABELS[city], zorder=3)
        if len(hs) >= 3:
            rho, p = stats.spearmanr(hs, ds)
            r_p, p_p = stats.pearsonr(hs, ds)
            ax.annotate(f"Pearson $r={r_p:+.2f}$ ($p={p_p:.3f}$)\nSpearman $\\rho={rho:+.2f}$",
                       xy=(0.96, 0.95), xycoords="axes fraction", fontsize=FONT_SIZE,
                       va="top", ha="right")
        ax.axhline(0, color="0.75", lw=0.7, zorder=1)
        ax.set_xlabel(r"Spatial heterogeneity index $h(D)$")
        ax.set_title(f"{topo.capitalize()} topology", fontsize=FONT_SIZE)
    axes[0].set_ylabel(r"$\Delta R^2$ (GCN $-$ Linear, 3-seed mean)")
    axes[0].legend(frameon=False, loc="lower left", handletextpad=0.4)
    fig.tight_layout()
    save_fig(fig, "fig3_h_index_vs_delta_r2")


# --------------------------------------------------------------------------- #
# Figure 4 — per-station h_i vs ΔR², axes indépendants, bande de confiance,
# annotation de la station atypique
# --------------------------------------------------------------------------- #
def fig4(het, ds):
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, FULL_WIDTH * 0.5))
    stats_by_topo = {}
    legend_handles = None
    for ax, (topo, col) in zip(axes, [("distance", "delta_r2_distance"),
                                      ("correlation", "delta_r2_correlation")]):
        d = ds.dropna(subset=["h_i", col])
        for city in CITIES:
            sub = d[d.city == city]
            ax.scatter(sub["h_i"], sub[col], s=32, alpha=0.9, color=CITY_COLORS[city],
                      marker=CITY_MARKERS[city], edgecolor="white", linewidth=0.5,
                      label=CITY_LABELS[city], zorder=3)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

        # ajustement + bande de confiance 95% (statsmodels OLS, IC de la moyenne)
        X = sm.add_constant(d["h_i"].values)
        model = sm.OLS(d[col].values, X).fit()
        xs = np.linspace(d["h_i"].min(), d["h_i"].max(), 100)
        pred = model.get_prediction(sm.add_constant(xs)).summary_frame(alpha=0.05)
        ax.fill_between(xs, pred["mean_ci_lower"], pred["mean_ci_upper"],
                        color="0.6", alpha=0.25, zorder=1, linewidth=0)
        ax.plot(xs, pred["mean"], color="0.25", lw=1.2, ls="--", zorder=2)

        rho, p_rho = stats.spearmanr(d["h_i"], d[col])
        r_p, p_p = stats.pearsonr(d["h_i"], d[col])
        stats_by_topo[topo] = dict(rho=rho, p_rho=p_rho, r=r_p, p_r=p_p, n=len(d))

        ax.axhline(0, color="0.75", lw=0.7, zorder=1)
        ax.set_xlabel(r"Local heterophily $h_i = 1-\overline{\rho}_{\mathrm{neigh}}$")
        ax.set_title(f"{topo.capitalize()} topology (n={len(d)})", fontsize=FONT_SIZE)
        ax.annotate(f"Pearson $r={r_p:+.2f}$ ($p={p_p:.3f}$)\nSpearman $\\rho={rho:+.2f}$ "
                   f"($p={p_rho:.3f}$)", xy=(0.96, 0.95), xycoords="axes fraction",
                   fontsize=FONT_SIZE, va="top", ha="right")
        # axes indépendants : PAS de sharey, chaque panneau prend ses propres limites
        # (fait par défaut avec plt.subplots sans sharey=True — vérifié au rendu,
        # les deux panneaux ont bien des étendues Y différentes)

        # annotation explicite de la station atypique (London/CT3, ΔR²≈-2.2 en
        # correlation, également extrême en distance) — positionnée dans la
        # marge basse, loin du nuage de points et des autres annotations.
        outlier = d[(d.city == "london")].sort_values(col).iloc[0] if (d.city == "london").any() else None
        if outlier is not None and outlier[col] < -1.5:
            ax.annotate(f"London / {outlier.station}\n(ΔR²={outlier[col]:.2f})",
                       xy=(outlier.h_i, outlier[col]), xytext=(0.30, 0.06),
                       textcoords="axes fraction", fontsize=FONT_SIZE,
                       ha="center", va="bottom",
                       arrowprops=dict(arrowstyle="->", color="0.2", lw=0.8,
                                      shrinkA=2, shrinkB=4),
                       zorder=4)

    axes[0].set_ylabel(r"$\Delta R^2$ (GCN $-$ Linear, seed 42)")
    # légende unique, partagée, au-dessus des deux panneaux — évite toute
    # collision avec les annotations (corrigé après un premier rendu où la
    # légende recouvrait le texte Pearson/Spearman, cf. commit).
    fig.legend(legend_handles, legend_labels, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.08), ncol=len(legend_labels), handletextpad=0.4,
              columnspacing=1.2)
    fig.tight_layout()
    save_fig(fig, "fig4_per_station_heterophily")
    return stats_by_topo


# --------------------------------------------------------------------------- #
# Figure 5 — courbe de pruning guidé, 3 villes (Beijing/London/Madrid — CZT
# n'a pas d'expérience de pruning, non inventée ici)
# --------------------------------------------------------------------------- #
def fig5(df):
    agg = df[df.station == "__aggregate__"]
    levels = [1.0, 0.75, 0.5, 0.25, 0.0]
    fig, ax = plt.subplots(figsize=(COL_WIDTH * 1.35, COL_WIDTH * 1.05))
    for city in ["beijing", "london", "madrid"]:
        xs, ys, es = [], [], []
        for lvl in levels:
            sub = agg[(agg.city == city) & (agg.model == "GCN-Transformer")
                     & (agg.variant == "") & (agg.keep_frac.astype(float) == lvl)
                     & (agg.seed.isin(["42", "123", "777"]))]
            if len(sub) == 0:
                continue
            m, s = agg_mean_std(pd.to_numeric(sub.r2).tolist())
            xs.append(lvl); ys.append(m); es.append(s)
        if xs:
            ax.errorbar(xs, ys, yerr=es, color=CITY_COLORS[city], marker=CITY_MARKERS[city],
                       markersize=5, capsize=3, lw=1.3, markeredgecolor="white",
                       markeredgewidth=0.5, label=CITY_LABELS[city], zorder=3)
        lin = agg[(agg.city == city) & (agg.model == "Linear-Transformer") & (agg.variant == "")
                 & (agg.seed.isin(["42", "123", "777"]))].drop_duplicates(subset=["seed"])
        if len(lin):
            lm, _ = agg_mean_std(pd.to_numeric(lin.r2).tolist())
            ax.axhline(lm, color=CITY_COLORS[city], lw=0.8, ls=":", alpha=0.7, zorder=2)
    ax.invert_xaxis()
    ax.set_xlabel("Fraction of edges retained")
    ax.set_ylabel(r"Aggregate $R^2$ (GCN-Transformer)")
    ax.legend(frameon=False, loc="lower right", handletextpad=0.4)
    fig.tight_layout()
    save_fig(fig, "fig5_edge_pruning_curve")


def main():
    set_style()
    df = load_raw_results()

    print("Figure 3 (h(D) vs ΔR², 4 réseaux)...")
    fig3(df)

    print("Figure 4 (per-station h_i vs ΔR², axes indépendants, bande CI, annotation)...")
    b = load_bench()
    het = {}
    for city in CITIES:
        print(f"  [{city}] h_i...", file=sys.stderr)
        het.update({(city, k): v for k, v in heterophily_for_city(b, city).items()})
    base = pd.read_csv(ROOT / "analysis" / "per_station_dataset.csv")
    stats4 = fig4(het, base)
    print("  stats Figure 4 :", stats4)

    print("Figure 5 (courbe de pruning guidé, 3 villes)...")
    fig5(df)

    print("\n⚠️ Figures 1 et 2 non régénérées : illustrations conceptuelles / "
          "schéma de pipeline, aucune donnée sous-jacente dans raw_results.csv "
          "— hors périmètre de ce script (cf. docstring).")


if __name__ == "__main__":
    main()
