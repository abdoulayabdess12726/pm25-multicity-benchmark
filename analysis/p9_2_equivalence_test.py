#!/usr/bin/env python3
"""
p9_2_equivalence_test.py — P9.2 (R2.4) : test d'équivalence pratique,
chaque modèle graphe vs Linear-Transformer, différence appariée par station.

Aucun entraînement — calcul pur depuis raw_results.csv, seed primaire 42
(convention "Comparaisons par station : seed primaire (42)", REVISION_BRIEF.md
§Conventions statistiques).

TOST (two one-sided tests) informel via IC bootstrap : équivalent si l'IC95%
de la différence appariée moyenne est entièrement contenu dans
[-EQUIVALENCE_MARGIN, +EQUIVALENCE_MARGIN].

Modèles couverts : GCN-Transformer (4 réseaux, 2 topologies) ; STGCN et
Graph WaveNet (Beijing/London/Madrid, topologie correlation uniquement —
jamais entraînés sur CZT, cf. E16 : protocole limité à GCN-Transformer +
Linear-Transformer pour ce réseau).

Sortie : analysis/p9_2_equivalence_results.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Marge d'équivalence pratique en ΔR², validée par l'utilisateur (2026-08-24)
# AVANT de lancer cette analyse — engagement pris antérieurement, pas un choix
# post hoc : PREREGISTRATION_CZT.md §3 utilise déjà ce même seuil (ΔR² ≥ −0.02)
# comme frontière "neutre ou meilleur" pour juger la prédiction P1 du 4e réseau,
# rédigée et committée avant tout entraînement CZT. Réutiliser exactement 0.02
# ici garde le projet cohérent avec cet engagement plutôt que d'introduire un
# nouveau seuil choisi après avoir vu les résultats de ce test précis.
EQUIVALENCE_MARGIN = 0.02

SEED_PRIMARY = 42
N_BOOT = 10000
CI_LEVEL = 0.95
BOOT_SEED = 42  # reproductibilité, cohérent avec bootstrap_ci_r (12_per_station_heterophily.py)


def bootstrap_ci_mean(diffs, n_boot=N_BOOT, ci=CI_LEVEL, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs)
    n = len(diffs)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        means[b] = diffs[idx].mean()
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(means, [100 * alpha, 100 * (1 - alpha)])
    return float(lo), float(hi)


def equivalence_verdict(ci_lo, ci_hi, margin=EQUIVALENCE_MARGIN):
    if ci_lo >= -margin and ci_hi <= margin:
        return "ÉQUIVALENT (IC entièrement dans ±%.2f)" % margin
    if ci_hi < -margin:
        return "NON ÉQUIVALENT — dégrade au-delà de la marge"
    if ci_lo > margin:
        return "NON ÉQUIVALENT — améliore au-delà de la marge"
    return "NON CONCLUANT (IC chevauche la frontière de la marge)"


def paired_diff(df, city, model, topology, seed=SEED_PRIMARY):
    sub = df[(df.city == city) & (df.seed == str(seed)) & (df.station != "__aggregate__")]
    sub = sub[sub.variant.fillna("") == ""]
    g = sub[(sub.model == model) & (sub.topology == topology)].set_index("station")["r2"]
    lin = sub[sub.model == "Linear-Transformer"].drop_duplicates(subset=["station"]).set_index("station")["r2"]
    common = g.index.intersection(lin.index)
    if len(common) == 0:
        return None
    diffs = (g.loc[common] - lin.loc[common]).values
    return diffs, list(common)


def paired_diff_multiseed(df, city, model, topology, seeds):
    """Diff appariée par station, MOYENNÉE sur `seeds` (jamais mélangée avec
    le seed primaire dans la même quantité — utilisé uniquement quand le
    seed primaire manque, cf. missing_reason). Pour chaque seed disponible,
    calcule le diff par station, puis moyenne par station sur les seeds
    communs — la paire (modèle, Linear) reste toujours au même seed à
    chaque étape, jamais croisée entre seeds."""
    per_seed = {}
    for s in seeds:
        res = paired_diff(df, city, model, topology, seed=s)
        if res is not None:
            diffs, stations = res
            per_seed[s] = dict(zip(stations, diffs))
    if not per_seed:
        return None
    common_stations = set.intersection(*[set(d.keys()) for d in per_seed.values()])
    if not common_stations:
        return None
    common_stations = sorted(common_stations)
    mean_diffs = np.array([np.mean([per_seed[s][st] for s in per_seed]) for st in common_stations])
    return mean_diffs, common_stations, sorted(per_seed.keys())


def missing_reason(df, city, model, topology):
    """Distingue : jamais entraîné, vs entraîné mais per-station non capturé
    pour ce run historique (seed 42, migration E7/P2 — capture per-station
    ajoutée seulement en P3). Les deux se rapportent différemment."""
    any_seed = df[(df.city == city) & (df.model == model) & (df.topology == topology)]
    if len(any_seed) == 0:
        return "jamais entraîné pour ce réseau/modèle/topologie"
    per_station_seeds = sorted(any_seed[any_seed.station != "__aggregate__"].seed.unique())
    if per_station_seeds:
        return (f"seed {SEED_PRIMARY} (aggregate-only, migration E7/P2 historique) — "
               f"per-station disponible aux seeds {per_station_seeds}, pas au primaire")
    return "entraîné (aggregate) mais per-station jamais capturé, aucun seed"


def main():
    df = pd.read_csv(ROOT / "results" / "raw_results.csv", dtype=str)
    df["r2"] = pd.to_numeric(df["r2"], errors="coerce")
    df["k"] = pd.to_numeric(df["k"], errors="coerce")

    # GCN-Transformer : k=5 uniquement (comparaison canonique, cohérente avec Table 2/4)
    df_gcn = df[(df.model != "GCN-Transformer") | (df.k == 5)]

    comparisons = []
    for city in ["beijing", "london", "madrid", "czt"]:
        for topo in ["distance", "correlation"]:
            comparisons.append((city, "GCN-Transformer", topo))
    for city in ["beijing", "london", "madrid"]:
        for model in ["stgcn", "graphwavenet"]:
            comparisons.append((city, model, "correlation"))

    # 4 comparaisons sans per-station au seed primaire (Beijing/London x
    # STGCN/GraphWaveNet, migration E7/P2) : repli sur un protocole 2-seeds
    # (123, 777) explicitement étiqueté — décidé le 2026-08-25, jamais mélangé
    # avec le seed primaire dans la même quantité dérivée.
    MULTISEED_FALLBACK = {
        ("beijing", "stgcn", "correlation"), ("beijing", "graphwavenet", "correlation"),
        ("london", "stgcn", "correlation"), ("london", "graphwavenet", "correlation"),
    }
    FALLBACK_SEEDS = [123, 777]

    lines = [f"# P9.2 (R2.4) — Test d'équivalence pratique, marge ±{EQUIVALENCE_MARGIN:.2f} ΔR²\n"]
    lines.append(f"Marge validée le 2026-08-24, fixée AVANT ce calcul — cf. `PREREGISTRATION_CZT.md` "
                 f"§3 (seuil déjà engagé, ΔR² ≥ −0.02). Différence appariée par station, IC bootstrap "
                 f"{N_BOOT} tirages, {CI_LEVEL:.0%}. **Protocole** : seed primaire {SEED_PRIMARY} par "
                 f"défaut ; repli 2-seeds ({'/'.join(map(str, FALLBACK_SEEDS))}, moyenne par station) "
                 f"pour 4 comparaisons Beijing/London STGCN/GraphWaveNet où le seed primaire n'a pas de "
                 f"per-station (migration E7/P2 historique, aggregate-only) — explicitement étiqueté "
                 f"dans la colonne Protocole, jamais mélangé avec le seed primaire dans une même "
                 f"quantité.\n")
    lines.append("| Réseau | Modèle | Topologie | Protocole | n stations | Diff. moyenne | IC95% bootstrap | Verdict |")
    lines.append("|---|---|---|---|---|---|---|---|")

    results = []
    for city, model, topo in comparisons:
        use_fallback = (city, model, topo) in MULTISEED_FALLBACK
        if use_fallback:
            res = paired_diff_multiseed(df_gcn, city, model, topo, FALLBACK_SEEDS)
            if res is None:
                lines.append(f"| {city} | {model} | {topo} | — | — | — | — | "
                             f"MISSING DATA (repli 2-seeds tenté, échec) |")
                print(f"{city}/{model}/{topo}: MISSING DATA (repli 2-seeds échoué)")
                continue
            diffs, stations, seeds_used = res
            protocol = f"2-seeds ({'/'.join(map(str, seeds_used))}, moy./station)"
        else:
            res = paired_diff(df_gcn, city, model, topo)
            if res is None:
                reason = missing_reason(df_gcn, city, model, topo)
                lines.append(f"| {city} | {model} | {topo} | — | — | — | — | MISSING DATA ({reason}) |")
                print(f"{city}/{model}/{topo}: MISSING DATA -- {reason}")
                continue
            diffs, stations = res
            protocol = f"seed {SEED_PRIMARY} (primaire)"

        mean_diff = float(diffs.mean())
        ci_lo, ci_hi = bootstrap_ci_mean(diffs)
        verdict = equivalence_verdict(ci_lo, ci_hi)
        results.append(dict(city=city, model=model, topology=topo, protocol=protocol, n=len(diffs),
                            mean_diff=mean_diff, ci_lo=ci_lo, ci_hi=ci_hi, verdict=verdict))
        lines.append(f"| {city} | {model} | {topo} | {protocol} | {len(diffs)} | {mean_diff:+.4f} | "
                     f"[{ci_lo:+.4f}, {ci_hi:+.4f}] | {verdict} |")
        print(f"{city:8s} {model:16s} {topo:12s} [{protocol}] n={len(diffs):2d} "
              f"diff={mean_diff:+.4f} IC=[{ci_lo:+.4f},{ci_hi:+.4f}]  {verdict}")

    res_df = pd.DataFrame(results)
    res_df.to_csv(ROOT / "analysis" / "p9_2_equivalence_results.csv", index=False)

    n_equiv = (res_df.verdict.str.startswith("ÉQUIVALENT")).sum()
    n_degrade = (res_df.verdict.str.contains("dégrade")).sum()
    n_improve = (res_df.verdict.str.contains("améliore")).sum()
    n_inconcl = (res_df.verdict.str.startswith("NON CONCLUANT")).sum()
    lines.append(f"\n**Récapitulatif** : {len(res_df)} comparaisons testées — "
                 f"{n_equiv} équivalentes, {n_degrade} dégradent au-delà de la marge, "
                 f"{n_improve} améliorent au-delà de la marge, {n_inconcl} non conclusives.\n")

    out = ROOT / "analysis" / "p9_2_equivalence_results.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nRapport : {out}")


if __name__ == "__main__":
    main()
