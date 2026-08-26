#!/usr/bin/env python3
"""
regenerate_figures.py — P10 : régénère les 5 figures du manuscrit en
vectoriel (PDF + SVG).

Figures 3, 4, 5 sont des graphiques de données, générés depuis
raw_results.csv. Figures 1 (illustration conceptuelle homogène/hétérogène)
et 2 (schéma du pipeline) N'ONT PAS de données sous-jacentes — ce sont des
illustrations, recréées ici en code à partir du même contenu conceptuel que
l'original (retrouvé seulement en PNG bitmap dans le .docx du manuscrit,
aucun script source dans ce dépôt), pas depuis raw_results.csv. Signalé
explicitement pour ne pas laisser croire qu'elles sortent des résultats.

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

Sorties : figures/fig1_conceptual_illustration.{pdf,svg}   (pas de données)
          figures/fig2_pipeline.{pdf,svg}                   (pas de données)
          figures/fig3_h_index_vs_delta_r2.{pdf,svg}
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
from scripts.regenerate_tables import load as load_resolved_raw_results  # noqa: E402

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


def _load_h_index_from_table2():
    """h(D) tracé en Figure 3 DOIT être celui de Table 2, pas une copie —
    lit directement manuscript/tables/table2_h_index.md (déjà régénéré par
    scripts/regenerate_tables.py depuis results/heterogeneity_index_v2.csv,
    l'EXCEPTION documentée hors raw_results.csv, cf. P3) plutôt qu'un dict
    recopié à la main dans ce fichier.

    Un H_INDEX = {...} en dur ici a exactement causé le risque que le
    relecteur 1 traque : le 2026-08-24, h(D) CZT est passé de 0.413 à 0.469
    (PREREGISTRATION_CZT.md §9 — 0.413 venait d'un calcul train-only
    incohérent avec la convention jeu-complet des 3 autres réseaux) ; la
    copie en dur avait bien été mise à jour à la main ce jour-là, mais nous
    étions à un accident près d'une figure et d'une table en désaccord sans
    aucun garde-fou. Lire Table 2 directement élimine la classe de bug —
    un futur recalcul de h(D) se propage automatiquement à la figure."""
    path = ROOT / "manuscript" / "tables" / "table2_h_index.md"
    if not path.exists():
        raise RuntimeError(
            "manuscript/tables/table2_h_index.md introuvable — lancer "
            "scripts/regenerate_tables.py avant regenerate_figures.py : "
            "Figure 3 lit h(D) depuis Table 2, jamais une copie indépendante."
        )
    label_to_code = {v: k for k, v in CITY_LABELS.items()}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        code = label_to_code.get(cells[0])
        if code is None:
            continue
        out[code] = float(cells[-1])
    missing = set(CITIES) - set(out)
    if missing:
        raise RuntimeError(f"Table 2 (h(D)) n'a pas de ligne pour : {sorted(missing)}")
    return out


H_INDEX = _load_h_index_from_table2()


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
    """Délègue à scripts.regenerate_tables.load() — SOURCE UNIQUE de la vue
    résolue de raw_results.csv (supersession SUSPECT_6STATION, doublon
    d'agrégat Madrid/correlation post-correctif NaN-sort, etc.).

    Une lecture indépendante ici (pd.read_csv brut, sans passer par ces
    résolutions) a exactement fait disparaître Madrid du panneau corrélation
    de Figure 3 (trouvé 2026-08-2x, cf. CHANGELOG_TABLES.md) : ce fichier
    lisait raw_results.csv sans le nettoyage déjà appliqué côté tables,
    tombait sur 6 lignes d'agrégat concurrentes au lieu de 3 pour
    Madrid/correlation/k=5, et gcn_lin_delta_3seed() rejetait la condition
    entière (len(gcn) != 3). Ne JAMAIS réimplémenter ce filtrage ici —
    passer par regenerate_tables.load(), le seul endroit où il est
    maintenu."""
    df = load_resolved_raw_results()
    df["r2"] = pd.to_numeric(df["r2"], errors="coerce")
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
# Figure 1 — illustration conceptuelle (homogène vs hétérogène)
# --------------------------------------------------------------------------- #
# Existait seulement en PNG bitmap intégré dans le .docx du manuscrit (pas de
# script source retrouvé dans ce dépôt — recherché, aucun match). Le style
# déjà en place (serif, axes en boîte) montre qu'elle avait déjà été produite
# par un outil de tracé (probablement matplotlib) plutôt que dessinée à la
# main — donc refaisable en code. Contenu conceptuel identique à l'original
# (2 panneaux, courbes synthétiques illustratives, PAS des données réelles —
# à ne jamais confondre avec un graphique de résultats), mis à jour avec CZT
# comme second exemple homogène (r̄=0.904, encore plus homogène que Beijing).
def fig1():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 4 * np.pi, 300)
    base = np.sin(t) + 0.3 * np.sin(2.3 * t)

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, FULL_WIDTH * 0.42))
    # Marges explicites (pas tight_layout, qui recalcule les positions et
    # entre en conflit avec les blocs de texte placés manuellement en
    # dessous — corrigé après un premier rendu où titre/légendes/texte se
    # chevauchaient en cascade).
    fig.subplots_adjust(left=0.06, right=0.98, top=0.72, bottom=0.30, wspace=0.12)
    fig.suptitle(r"Spatial (graph) encoding of PM$_{2.5}$ across IoT stations",
                fontsize=FONT_SIZE + 1, y=0.98)

    # Panneau homogène : 4 séries quasi identiques (bruit minime autour de
    # la même série de base) — Beijing/CZT, r̄≈0.88-0.90.
    ax = axes[0]
    colors = [CITY_COLORS[c] for c in ["beijing", "czt", "madrid", "london"]]
    for c in colors:
        y = base + rng.normal(0, 0.03, size=t.shape)
        ax.plot(t, y, color=c, lw=1.0, alpha=0.85)
    ax.set_title("Homogeneous network\n(e.g. Beijing / CZT, " r"$\bar r \approx 0.88\text{-}0.90$)",
                fontsize=FONT_SIZE, pad=8)
    ax.set_xlabel("time", fontsize=FONT_SIZE, labelpad=4)

    # Panneau hétérogène : 4 séries dissimilaires (marches aléatoires
    # indépendantes + faible composante commune) — London/Madrid, r̄≈0.4-0.5.
    ax = axes[1]
    colors = [CITY_COLORS[c] for c in ["beijing", "london", "madrid", "czt"]]
    for c in colors:
        walk = np.cumsum(rng.normal(0, 0.12, size=t.shape))
        y = 0.3 * base + walk
        y = (y - y.mean()) / y.std()
        ax.plot(t, y, color=c, lw=1.0, alpha=0.85)
    ax.set_title("Heterogeneous network\n(e.g. London / Madrid, " r"$\bar r \approx 0.4\text{-}0.5$)",
                fontsize=FONT_SIZE, pad=8)
    ax.set_xlabel("time", fontsize=FONT_SIZE, labelpad=4)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    # Blocs de texte du bas, en coordonnées FIGURE (pas axes) — centrés sur
    # chaque panneau via sa position réelle après subplots_adjust, pour ne
    # jamais dépendre d'un décalage devinable en coordonnées axes.
    for ax, txt in zip(axes, [
        "neighbours nearly identical\n$\\Rightarrow$ aggregation redundant\n$\\Rightarrow$ small penalty",
        "neighbours dissimilar\n$\\Rightarrow$ averaging mixes signals\n$\\Rightarrow$ large penalty",
    ]):
        pos = ax.get_position()
        xc = (pos.x0 + pos.x1) / 2
        fig.text(xc, 0.16, txt, ha="center", va="top", fontsize=FONT_SIZE)

    save_fig(fig, "fig1_conceptual_illustration")


# --------------------------------------------------------------------------- #
# Figure 2 — schéma du pipeline
# --------------------------------------------------------------------------- #
# Même situation que Figure 1 : bitmap seul retrouvé, style déjà cohérent
# avec matplotlib, aucun script source dans ce dépôt. Contenu mis à jour :
# la liste des stations dans la case « Input » incluait seulement Beijing/
# London/Madrid — Chang-Zhu-Tan (20 stations, E16) ajoutée pour rester
# exacte après la révision. Le reste (protocole, architecture) inchangé.
def _text_width_in(fig, s, fontsize, weight="normal"):
    """Largeur réelle (pouces) de la ligne la plus longue de `s`, mesurée
    par rendu — pas devinée. Nécessaire ici : un premier essai à largeur de
    boîte uniforme (6 boîtes égales) a débordé sur les boîtes voisines,
    seulement ~10 caractères tenant par ligne à 9pt sur 6.77in/6 boîtes."""
    renderer = fig.canvas.get_renderer()
    widths = []
    for line in s.split("\n"):
        t = fig.text(0, 0, line, fontsize=fontsize, fontweight=weight)
        bbox = t.get_window_extent(renderer=renderer)
        widths.append(bbox.width / fig.dpi)
        t.remove()
    return max(widths) if widths else 0.0


def _text_height_in(fig, s, fontsize, weight="normal", linespacing=1.0):
    """Hauteur réelle (pouces) du bloc multi-ligne `s`, mesurée par rendu —
    nécessaire car un box_h uniforme avec un corps centré verticalement a
    fait déborder le corps le plus long (6 lignes, boîte « Output &
    evaluation ») dans le titre au-dessus."""
    renderer = fig.canvas.get_renderer()
    t = fig.text(0, 0, s, fontsize=fontsize, fontweight=weight, linespacing=linespacing)
    bbox = t.get_window_extent(renderer=renderer)
    t.remove()
    return bbox.height / fig.dpi


def fig2():
    boxes = [
        ("Input", "Beijing: 12\nLondon: 8\nMadrid: 7\nCZT: 20\n5 features,\nhourly", "#E8ECF7"),
        ("Preprocessing", "Interpolation;\nMinMax\n(train fit);\n70/15/15\nsplit", "#E8ECF7"),
        ("Graph\nconstruction", "k-NN, k=5\ndistance /\ncorrelation\n(GCN/GAT\nonly)", "#F0EBD8"),
        ("Spatial\nencoder", "Linear / GCN\n/ GAT\n(per\ntimestep)", "#DCEEDD"),
        ("Temporal\nTransformer", "24h window;\n2 layers,\n4 heads,\nd=64", "#F7E9D7"),
        ("Output &\nevaluation", r"PM$_{2.5}$ at" "\n" r"$t{+}1$;" "\nRMSE, MAE,\n$R^2$;\nWilcoxon+\nHolm, $d$", "#F5E1E4"),
    ]
    n = len(boxes)
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, FULL_WIDTH * 0.42))
    # L'axe occupe (quasi) toute la figure : les largeurs de boîte calculées
    # en pouces ci-dessous sont utilisées directement comme coordonnées data,
    # ce qui ne correspond aux pouces réels à l'écran QUE si l'axe couvre
    # bien 100% de la largeur figure — sinon les boîtes seraient mises à
    # l'échelle sans que le texte mesuré ne le soit, et redéborderaient.
    ax.set_position([0.0, 0.02, 1.0, 0.96])
    fig.canvas.draw()  # nécessaire avant toute mesure de texte

    pad_in = 0.10  # marge interne gauche/droite par boîte, en pouces
    gap_in = 0.20
    widths_in = []
    for title, body, _ in boxes:
        w = max(_text_width_in(fig, title, FONT_SIZE, "bold"),
               _text_width_in(fig, body, FONT_SIZE))
        widths_in.append(w + 2 * pad_in)
    scale = (FULL_WIDTH - (n - 1) * gap_in) / sum(widths_in)
    widths_in = [w * scale for w in widths_in]  # remplit exactement FULL_WIDTH

    # box_h et l'ancrage vertical titre/corps sont mesurés, pas devinés : le
    # corps le plus long (6 lignes, « Output & evaluation ») centré dans une
    # boîte de hauteur uniforme débordait dans le titre au-dessus. Ici le
    # titre est ancré en haut, le corps ancré juste en dessous (va="top"
    # pour les deux) — un corps plus long grandit vers le bas de la boîte,
    # jamais vers le titre.
    top_pad = 0.14
    title_body_gap = 0.10
    bottom_pad = 0.14
    title_h = max(_text_height_in(fig, t, FONT_SIZE, "bold") for t, _, _ in boxes)
    body_h = max(_text_height_in(fig, b, FONT_SIZE, linespacing=1.5) for _, b, _ in boxes)
    box_h = top_pad + title_h + title_body_gap + body_h + bottom_pad
    title_y = box_h - top_pad
    body_y = title_y - title_h - title_body_gap
    gap = gap_in  # unités data == pouces (axes positionnée pour occuper toute la figure, cf. plus bas)
    xs = []
    x_cursor = 0.0
    for w in widths_in:
        xs.append(x_cursor)
        x_cursor += w + gap

    from matplotlib.patches import FancyBboxPatch
    total_w = x_cursor - gap
    for i, ((title, body, color), x, w) in enumerate(zip(boxes, xs, widths_in)):
        highlight = ("encoder" in title)
        box = FancyBboxPatch((x, 0), w, box_h,
                             boxstyle="round,pad=0.02,rounding_size=0.05",
                             linewidth=1.4 if highlight else 0.9,
                             edgecolor=CITY_COLORS["madrid"] if highlight else "0.15",
                             facecolor=color, zorder=2)
        ax.add_patch(box)
        ax.text(x + w / 2, title_y, title, ha="center", va="top",
               fontsize=FONT_SIZE, fontweight="bold", zorder=3)
        ax.text(x + w / 2, body_y, body, ha="center", va="top",
               fontsize=FONT_SIZE, zorder=3, linespacing=1.5)
        if i < n - 1:
            xm = x + w + gap / 2
            ax.annotate("", xy=(xm + gap / 2 - 0.015, box_h / 2), xytext=(xm - gap / 2 + 0.015, box_h / 2),
                       arrowprops=dict(arrowstyle="-|>", color="0.15", lw=1.1), zorder=1)
        if highlight:
            ax.annotate("component under test", xy=(x + w / 2, 0), xytext=(x + w / 2, -0.32),
                       ha="center", va="top", fontsize=FONT_SIZE, style="italic",
                       color=CITY_COLORS["madrid"],
                       arrowprops=dict(arrowstyle="-", color=CITY_COLORS["madrid"], lw=1.0))

    ax.set_xlim(-0.05, total_w + 0.05)
    ax.set_ylim(-0.55, box_h + 0.15)
    ax.set_aspect("auto")
    ax.axis("off")
    save_fig(fig, "fig2_pipeline")


# --------------------------------------------------------------------------- #
# Figure 3 — h(D) vs ΔR², 4 réseaux, 2 panneaux topologie
# --------------------------------------------------------------------------- #
def _load_table4_network_counts():
    """Nombre de réseaux avec une ligne ΔR² non-MISSING dans Table 4, par
    topologie — sert à vérifier que Figure 3 ne perd silencieusement aucun
    point (cf. incident Madrid absent du panneau corrélation, 2026-08-2x :
    l'assertion h(D)==Table 2 seule ne l'aurait pas attrapée, puisque le
    problème n'était pas la valeur de h(D) mais l'absence totale du point)."""
    path = ROOT / "manuscript" / "tables" / "table4_benchmark.md"
    if not path.exists():
        raise RuntimeError("manuscript/tables/table4_benchmark.md introuvable — "
                           "lancer scripts/regenerate_tables.py avant regenerate_figures.py.")
    counts = {"distance": 0, "correlation": 0}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells[0] == "City" or len(cells) < 5:
            continue
        topo, delta = cells[1], cells[4]
        if topo in counts and delta != "MISSING":
            counts[topo] += 1
    return counts


def _load_table7_coefficients():
    """Spearman ρ/p par topologie depuis Table 7 (rendue) — Table 7 ne
    rapporte pas Pearson, donc seul ρ/p est comparable ; Pearson r/p reste
    annoté sur la figure mais sans contrepartie tabulaire à vérifier."""
    path = ROOT / "manuscript" / "tables" / "table7_cross_city_correlation.md"
    if not path.exists():
        raise RuntimeError("manuscript/tables/table7_cross_city_correlation.md introuvable — "
                           "lancer scripts/regenerate_tables.py avant regenerate_figures.py.")
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells[0] == "Topology" or len(cells) < 3:
            continue
        try:
            out[cells[0]] = (float(cells[1]), float(cells[2]))
        except ValueError:
            continue
    return out


def fig3(df):
    # Assertion de cohérence figure/table #1, indépendante du chargement de
    # H_INDEX ci-dessus (re-lit Table 2 depuis zéro plutôt que de réutiliser
    # le dict déjà en mémoire) : protège contre une divergence future même
    # si un refactor introduit un second chemin de calcul pour h(D) dans ce
    # fichier. Exactement le type d'incohérence table/figure que le
    # relecteur 1 traque (cf. incident H_INDEX en dur, 2026-08-24).
    h_index_table2 = _load_h_index_from_table2()
    for city in CITIES:
        assert H_INDEX[city] == h_index_table2[city], (
            f"Figure 3 : h(D) pour {city} ({H_INDEX[city]}) != Table 2 "
            f"({h_index_table2[city]}) — régénérer Table 2 avant Figure 3."
        )
    print(f"  h(D) vérifié == Table 2 pour les 4 réseaux : {H_INDEX}")

    # Assertions #2/#3 : nombre de points par panneau == Table 4, et
    # coefficients affichés == Table 7. Ajoutées après l'incident Madrid
    # (absent du panneau corrélation : gcn_lin_delta_3seed rejetait la
    # condition à cause d'un doublon d'agrégat non résolu ici — l'assertion
    # h(D)==Table 2 n'aurait rien détecté, puisque Madrid n'était simplement
    # jamais ajouté à la liste des points tracés).
    table4_counts = _load_table4_network_counts()
    table7_coeffs = _load_table7_coefficients()

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

        assert len(hs) == table4_counts[topo], (
            f"Figure 3 [{topo}] : {len(hs)} point(s) tracé(s), Table 4 en a "
            f"{table4_counts[topo]} pour cette topologie — un réseau a été "
            f"silencieusement écarté (cf. CHANGELOG_TABLES.md, incident Madrid)."
        )

        if len(hs) >= 3:
            rho, p = stats.spearmanr(hs, ds)
            r_p, p_p = stats.pearsonr(hs, ds)
            if topo in table7_coeffs:
                rho_t7, p_t7 = table7_coeffs[topo]
                assert abs(rho - rho_t7) < 5e-4 and abs(p - p_t7) < 5e-4, (
                    f"Figure 3 [{topo}] : Spearman ρ={rho:.4f}/p={p:.4f} tracés, "
                    f"Table 7 donne ρ={rho_t7:.4f}/p={p_t7:.4f} — divergence."
                )
            ax.annotate(f"Pearson $r={r_p:+.2f}$ ($p={p_p:.3f}$)\nSpearman $\\rho={rho:+.2f}$",
                       xy=(0.96, 0.95), xycoords="axes fraction", fontsize=FONT_SIZE,
                       va="top", ha="right")
        ax.axhline(0, color="0.75", lw=0.7, zorder=1)
        ax.set_xlabel(r"Spatial heterogeneity index $h(D)$")
        ax.set_title(f"{topo.capitalize()} topology", fontsize=FONT_SIZE)
    print(f"  points/panneau vérifiés == Table 4 : {table4_counts} ; "
         f"Spearman ρ/p vérifiés == Table 7 : {table7_coeffs}")
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
    # CZT explicitement déclarée absente, dans la figure elle-même — pas
    # seulement dans la légende texte du manuscrit (consigne 2026-08-25 :
    # le lecteur ne doit pas avoir à le remarquer par lui-même).
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color="none"))
    labels.append("(Chang-Zhu-Tan: no pruning\nexperiment run on this network)")
    ax.legend(handles, labels, frameon=False, loc="lower right", handletextpad=0.4,
             fontsize=FONT_SIZE)
    fig.tight_layout()
    save_fig(fig, "fig5_edge_pruning_curve")


def main():
    set_style()
    df = load_raw_results()

    print("Figure 1 (illustration conceptuelle)...")
    fig1()

    print("Figure 2 (schéma pipeline)...")
    fig2()

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

    print("\n⚠️ Figures 1 et 2 : illustrations recréées en code (même contenu "
          "conceptuel que l'original), PAS depuis raw_results.csv — aucune "
          "donnée sous-jacente, cf. docstring.")


if __name__ == "__main__":
    main()
