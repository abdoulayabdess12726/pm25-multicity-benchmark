#!/usr/bin/env python3
"""
16_pruning_controls_figure.py — E12 (P5) : figure du contrôle de pruning
==========================================================================
3 courbes par ville (guidé / aléatoire à densité appariée / inverse) +
niveau Linear-Transformer en pointillés. Lit exclusivement raw_results.csv
(source unique, REVISION_BRIEF.md). Style sobre, Times New Roman, cohérent
avec 12_per_station_heterophily.py.

Usage : python 16_pruning_controls_figure.py
Sortie : figures/pruning_controls.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.stats import agg_mean_std  # noqa: E402

CITIES = ["beijing", "london", "madrid"]
LEVELS_SHARED = [1.0, 0.0]          # communs aux 3 stratégies (guidé=aléatoire=inverse par construction)
LEVELS_CONTROLS = [0.75, 0.25]      # niveaux réentraînés pour aléatoire/inverse (E12, budget réduit)


def load():
    df = pd.read_csv(ROOT / "results" / "raw_results.csv", dtype=str)
    df["r2"] = pd.to_numeric(df["r2"], errors="coerce")
    df["keep_frac"] = pd.to_numeric(df["keep_frac"], errors="coerce")
    df["variant"] = df["variant"].fillna("")
    return df[df.station == "__aggregate__"].copy()


def curve(agg, city, variant):
    levels = sorted(LEVELS_SHARED + LEVELS_CONTROLS)
    xs, ys, es = [], [], []
    for lvl in levels:
        sub = agg[(agg.city == city) & (agg.model == "GCN-Transformer")
                 & (agg.keep_frac == lvl) & (agg.variant == variant)]
        if len(sub) == 0:
            continue
        m, s = agg_mean_std(sub.r2.tolist())
        xs.append(lvl); ys.append(m); es.append(s)
    return xs, ys, es


def linear_ref(agg, city):
    sub = agg[(agg.city == city) & (agg.model == "Linear-Transformer")
             & (agg.variant == "") & (agg.seed.isin(["42", "123", "777"]))].drop_duplicates("seed")
    m, _ = agg_mean_std(sub.r2.tolist())
    return m


def main():
    agg = load()
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11, "axes.linewidth": 0.8,
    })
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=False)
    styles = {
        "": dict(label="Guidé (hétérophilie décroissante)", color="#1b4965", marker="o"),
        "random_matched": dict(label="Aléatoire (densité appariée, 5 tirages)", color="#7a7a7a", marker="s"),
        "inverse": dict(label="Inverse (homophilie décroissante)", color="#bc4749", marker="^"),
    }
    for ax, city in zip(axes, CITIES):
        for variant, style in styles.items():
            xs, ys, es = curve(agg, city, variant)
            if not xs:
                continue
            ax.errorbar(xs, ys, yerr=es, capsize=3, linewidth=1.4, markersize=5, **style)
        lin = linear_ref(agg, city)
        ax.axhline(lin, linestyle="--", color="black", linewidth=1.0, alpha=0.7,
                  label="Linear-Transformer")
        ax.set_title(city.capitalize(), fontsize=11)
        ax.set_xlabel("Fraction d'arêtes conservée")
        ax.invert_xaxis()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel(r"$R^2$ (agrégé, test)")
    axes[-1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Contrôle de pruning : guidé vs aléatoire à densité appariée vs inverse (E12, P5)",
                fontsize=11, y=1.02)
    fig.tight_layout()
    out = ROOT / "figures" / "pruning_controls.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Figure écrite : {out}")


if __name__ == "__main__":
    main()
