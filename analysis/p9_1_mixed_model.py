#!/usr/bin/env python3
"""
p9_1_mixed_model.py — P9.1 (R2.1) : modèle mixte ΔR² ~ h_i, intercept
aléatoire par réseau (4 réseaux : Beijing, London, Madrid, CZT).

Aucun entraînement — calcul pur sur analysis/per_station_dataset.csv
(construit depuis raw_results.csv par build_per_station_dataset.py).

Un modèle par topologie (distance, correlation), cohérent avec le
traitement séparé déjà en vigueur dans 12_per_station_heterophily.py /
Table 6/7. Stations à h_i indéfini (MENDEZ ALVARO) exclues du modèle —
comme dans l'analyse originale — et signalées.

Sortie : analysis/p9_1_mixed_model_results.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MIN_N_FOR_SPEARMAN = 8  # sous ce seuil, p-value peu fiable — signalé, pas caché


def fit_mixed_model(df, topo_col):
    d = df.dropna(subset=["h_i", topo_col]).copy()
    d = d.rename(columns={topo_col: "delta_r2"})
    md = smf.mixedlm("delta_r2 ~ h_i", d, groups=d["city"])
    fit = md.fit(reml=True)

    beta = fit.params["h_i"]
    se = fit.bse["h_i"]
    ci_lo, ci_hi = fit.conf_int().loc["h_i"]
    p = fit.pvalues["h_i"]

    var_re = float(fit.cov_re.iloc[0, 0])
    var_resid = float(fit.scale)
    icc = var_re / (var_re + var_resid)

    return dict(n=len(d), n_cities=d.city.nunique(), beta=beta, se=se,
               ci_lo=ci_lo, ci_hi=ci_hi, p=p, icc=icc, var_re=var_re,
               var_resid=var_resid, fit=fit, data=d)


def within_city_spearman(df, topo_col):
    d = df.dropna(subset=["h_i", topo_col])
    rows = []
    for city, sub in d.groupby("city"):
        n = len(sub)
        if n < 3:
            rows.append(dict(city=city, n=n, rho=np.nan, p=np.nan,
                             note="n<3, Spearman non calculable"))
            continue
        rho, p = stats.spearmanr(sub["h_i"], sub[topo_col])
        note = "" if n >= MIN_N_FOR_SPEARMAN else f"n={n} < {MIN_N_FOR_SPEARMAN} — inférence peu fiable, à ne pas sur-interpréter"
        rows.append(dict(city=city, n=n, rho=rho, p=p, note=note))
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(ROOT / "analysis" / "per_station_dataset.csv")

    lines = ["# P9.1 (R2.1) — Modèle mixte ΔR² ~ hétérophilie locale, 4 réseaux\n"]
    lines.append(f"Base : `analysis/per_station_dataset.csv` ({len(df)} stations, "
                 f"4 réseaux : {sorted(df.city.unique())}). "
                 f"Stations à h_i indéfini exclues : "
                 f"{df[df.h_i.isna()][['city','station']].values.tolist()}.\n")

    for topo, col in [("distance", "delta_r2_distance"), ("correlation", "delta_r2_correlation")]:
        print(f"\n=== Topologie : {topo} ===")
        res = fit_mixed_model(df, col)
        print(res["fit"].summary())

        lines.append(f"## Topologie {topo}\n")
        lines.append(f"n = {res['n']} stations, {res['n_cities']} réseaux (intercept aléatoire).\n")
        lines.append(f"**Effet fixe h_i** : β = {res['beta']:+.4f}, SE = {res['se']:.4f}, "
                     f"IC95% = [{res['ci_lo']:+.4f}, {res['ci_hi']:+.4f}], p = {res['p']:.4g}\n")
        lines.append(f"**ICC** (part de variance attribuable au réseau) = {res['icc']:.3f} "
                     f"(var_réseau={res['var_re']:.5f}, var_résiduelle={res['var_resid']:.5f})\n")

        spear = within_city_spearman(df, col)
        lines.append("\n**Spearman intra-réseau** :\n")
        lines.append("| Réseau | n | ρ | p | Note |")
        lines.append("|---|---|---|---|---|")
        for _, r in spear.iterrows():
            rho_s = f"{r.rho:+.3f}" if pd.notna(r.rho) else "—"
            p_s = f"{r.p:.4g}" if pd.notna(r.p) else "—"
            lines.append(f"| {r.city} | {r.n} | {rho_s} | {p_s} | {r.note} |")
        lines.append("")
        print(spear.to_string(index=False))

    out = ROOT / "analysis" / "p9_1_mixed_model_results.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nRapport : {out}")


if __name__ == "__main__":
    main()
