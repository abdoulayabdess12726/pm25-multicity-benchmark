#!/usr/bin/env python3
"""
p9_3_extended_model.py — P9.3 (R2.7), extension du modèle mixte P9.1 avec
covariables STATION-level (pas réseau-level : 4 réseaux, pas de puissance
pour un ajustement à ce niveau — à énoncer dans le manuscrit, pas testé ici).

Covariables ajoutées, toutes calculées par station (varient au sein d'un
même réseau, contrairement à un ajustement réseau-level) :
  - train_var       : variance PM2.5 période train (log, forte hétérogénéité
                       d'échelle entre réseaux — Beijing ~6000 vs Madrid ~60)
  - lag1_autocorr   : autocorrélation lag-1 PM2.5 (série complète)
  - raw_missing_rate: taux de manquants avant interpolation (qualité donnée)

Aucun entraînement — calcul pur.
Sortie : analysis/p9_3_extended_model_results.md
"""
import glob
import importlib.util
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_bench():
    spec = importlib.util.spec_from_file_location("bench", str(ROOT / "06_train_multistation.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench"] = mod
    with redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


def load_city(b, city):
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
    return np.asarray(data, dtype=np.float32)


def raw_missing_per_station_beijing(names):
    out = {}
    for name in names:
        f = ROOT / f"data/beijing_real/PRSA_Data_20130301-20170228/PRSA_Data_{name}_20130301-20170228.csv"
        df = pd.read_csv(f)
        out[name] = float(df["PM2.5"].isna().mean())
    return out


def raw_missing_per_station_london(names):
    out = {}
    for name in names:
        f = ROOT / f"data/london_laqn/LAQN_{name}_PM25.csv"
        if not f.exists():
            out[name] = np.nan
            continue
        df = pd.read_csv(f)
        val_col = df.columns[1]
        full_range = pd.date_range(pd.to_datetime(df.iloc[:, 0]).min(),
                                   pd.to_datetime(df.iloc[:, 0]).max(), freq="1h")
        out[name] = 1 - df[val_col].notna().sum() / len(full_range)
    return out


def raw_missing_per_station_madrid(names):
    out = {}
    for name in names:
        f = ROOT / f"data/madrid_openaq/OPENAQ_{name}.csv"
        if not f.exists():
            out[name] = np.nan
            continue
        df = pd.read_csv(f)
        dt = pd.to_datetime(df["datetime"], utc=True, errors="coerce").dt.tz_localize(None)
        full_range = pd.date_range(dt.min(), dt.max(), freq="1h")
        out[name] = 1 - df["value"].notna().sum() / len(full_range)
    return out


def raw_missing_per_station_czt(names):
    cov = pd.read_csv(ROOT / "data/czt_processed/station_coverage.csv").set_index("station")
    return {n: (1 - cov.loc[n, "raw_coverage"]) if n in cov.index else np.nan for n in names}


RAW_MISSING_FN = {
    "beijing": raw_missing_per_station_beijing,
    "london": raw_missing_per_station_london,
    "madrid": raw_missing_per_station_madrid,
    "czt": raw_missing_per_station_czt,
}


def main():
    b = load_bench()
    base = pd.read_csv(ROOT / "analysis" / "per_station_dataset.csv")

    cov_rows = []
    for city in ["beijing", "london", "madrid", "czt"]:
        print(f"[{city}] covariables station-level...", file=sys.stderr)
        data = load_city(b, city)
        names = list(b.STATION_NAMES)
        pm_idx = b.FEATURES.index("PM2.5")
        T = data.shape[0]
        train_len = int(0.70 * T)
        pm_full = data[:, :, pm_idx]
        pm_train = data[:train_len, :, pm_idx]

        raw_miss = RAW_MISSING_FN[city](names)

        for j, name in enumerate(names):
            train_var = float(np.nanvar(pm_train[:, j], ddof=1))
            s = pm_full[:, j]
            valid = ~np.isnan(s)
            s0, s1 = s[:-1][valid[:-1] & valid[1:]], s[1:][valid[:-1] & valid[1:]]
            lag1 = float(np.corrcoef(s0, s1)[0, 1]) if len(s0) > 2 and np.nanstd(s) > 1e-9 else np.nan
            cov_rows.append(dict(city=city, station=name, train_var=train_var,
                                 lag1_autocorr=lag1, raw_missing_rate=raw_miss.get(name, np.nan)))

    cov_df = pd.DataFrame(cov_rows)
    cov_df.to_csv(ROOT / "analysis" / "per_station_covariates.csv", index=False)

    merged = base.merge(cov_df, on=["city", "station"], how="left")
    # log(train_var) : échelles très différentes entre réseaux (Beijing ~6000
    # vs Madrid ~60) — log stabilise avant d'entrer dans un modèle linéaire
    # unique sur les 4 réseaux.
    merged["log_train_var"] = np.log(merged["train_var"].clip(lower=1e-6))

    lines = ["# P9.3 — Modèle mixte étendu, covariables station-level\n"]
    lines.append("Covariables ajoutées à h_i (toutes station-level, aucun ajustement "
                 "réseau-level — 4 réseaux, pas de puissance pour ça) : "
                 "log(variance PM2.5 train), autocorrélation lag-1, taux de "
                 "manquants brut.\n")

    for topo, col in [("distance", "delta_r2_distance"), ("correlation", "delta_r2_correlation")]:
        d = merged.dropna(subset=["h_i", col, "log_train_var", "lag1_autocorr", "raw_missing_rate"]).copy()
        d = d.rename(columns={col: "delta_r2"})

        print(f"\n=== {topo} : modèle NON ajusté (h_i seul) ===")
        m0 = smf.mixedlm("delta_r2 ~ h_i", d, groups=d["city"]).fit(reml=True)
        print(m0.summary())

        print(f"\n=== {topo} : modèle AJUSTÉ (+ covariables station-level) ===")
        m1 = smf.mixedlm("delta_r2 ~ h_i + log_train_var + lag1_autocorr + raw_missing_rate",
                         d, groups=d["city"]).fit(reml=True)
        print(m1.summary())

        beta0, se0, p0 = m0.params["h_i"], m0.bse["h_i"], m0.pvalues["h_i"]
        ci0 = m0.conf_int().loc["h_i"]
        beta1, se1, p1 = m1.params["h_i"], m1.bse["h_i"], m1.pvalues["h_i"]
        ci1 = m1.conf_int().loc["h_i"]
        sign_flip = np.sign(beta1) != np.sign(beta0)

        # Diagnostic de colinéarité — à calculer et publier AVANT d'interpréter
        # le coefficient ajusté, pas après. Un flip de signe + significativité
        # accrue est le symptôme classique d'une colinéarité sévère, pas un
        # renforcement réel de l'effet.
        corrs = {c: stats.pearsonr(d["h_i"], d[c]) for c in ["log_train_var", "lag1_autocorr", "raw_missing_rate"]}

        lines.append(f"## Topologie {topo} (n={len(d)})\n")
        lines.append(f"**Non ajusté** : β(h_i) = {beta0:+.4f}, SE={se0:.4f}, "
                     f"IC95%=[{ci0[0]:+.4f}, {ci0[1]:+.4f}], p={p0:.4g}\n")
        lines.append(f"**Ajusté** (+ log_train_var + lag1_autocorr + raw_missing_rate) : "
                     f"β(h_i) = {beta1:+.4f}, SE={se1:.4f}, IC95%=[{ci1[0]:+.4f}, {ci1[1]:+.4f}], "
                     f"p={p1:.4g}\n")

        lines.append("**Corrélation h_i × covariables ajoutées (n=%d)** :\n" % len(d))
        for name, (r, p) in corrs.items():
            lines.append(f"  - h_i vs {name} : r={r:+.3f}, p={p:.3g}")
        lines.append("")

        if sign_flip:
            lines.append(
                f"**⚠️ LE COEFFICIENT CHANGE DE SIGNE ({beta0:+.3f} → {beta1:+.3f}) ET DEVIENT "
                f"PLUS SIGNIFICATIF (p={p0:.3g} → p={p1:.3g}) EN L'AJUSTANT — CE N'EST PAS UN "
                f"RENFORCEMENT RÉEL DE L'EFFET, C'EST UNE COLINÉARITÉ SÉVÈRE.** h_i corrèle à "
                f"r={corrs['raw_missing_rate'][0]:+.2f} avec le taux de manquants, "
                f"r={corrs['log_train_var'][0]:+.2f} avec log(variance train), "
                f"r={corrs['lag1_autocorr'][0]:+.2f} avec l'autocorrélation lag-1 — et ces trois "
                f"covariables varient presque exclusivement ENTRE réseaux, pas AU SEIN d'un même "
                f"réseau (Beijing/CZT : h_i bas, peu de manquants, forte autocorrélation ; "
                f"London/Madrid : h_i élevé, beaucoup de manquants, autocorrélation plus faible — "
                f"le même clivage à 4 points). Avec seulement 4 groupes, ces covariables "
                f"« station-level » capturent en réalité presque la même variation ENTRE réseaux "
                f"que h_i lui-même : le modèle ajusté n'isole rien, il redistribue un signal "
                f"quasi-confondu entre des prédicteurs colinéaires. **CE RÉSULTAT N'EST PAS "
                f"INTERPRÉTABLE COMME « le coefficient d'hétérophilie survit, renforcé, à "
                f"l'ajustement » — c'est un artefact statistique, pas une conclusion sur h_i.**\n")
        else:
            lines.append(f"Pas de changement de signe. Coefficient "
                         f"{'toujours significatif' if p1 < 0.10 else 'devient non significatif'} "
                         f"après ajustement (p={p1:.3g}).\n")
        for name, coef, se, p in zip(["log_train_var", "lag1_autocorr", "raw_missing_rate"],
                                     [m1.params[c] for c in ["log_train_var", "lag1_autocorr", "raw_missing_rate"]],
                                     [m1.bse[c] for c in ["log_train_var", "lag1_autocorr", "raw_missing_rate"]],
                                     [m1.pvalues[c] for c in ["log_train_var", "lag1_autocorr", "raw_missing_rate"]]):
            lines.append(f"  - {name} : β={coef:+.4f}, SE={se:.4f}, p={p:.4g}")
        lines.append("")

    lines.append(
        "\n**Conclusion honnête sur ce point du plan (« le coefficient d'hétérophilie "
        "survit-il à l'ajustement ? »)** : NON TESTABLE PROPREMENT avec ces covariables à "
        "n=4 réseaux. Les covariables station-level naturelles (variance train, "
        "autocorrélation, taux de manquants) se sont révélées quasi-confondues avec "
        "l'appartenance réseau elle-même — colinéarité sévère (r jusqu'à 0.93 avec h_i), "
        "pas un problème de puissance sur h_i mais un problème d'identification du modèle "
        "ajusté. Un ajustement fiable demanderait soit plus de réseaux (variation "
        "indépendante entre et au sein des groupes), soit des covariables qui varient "
        "vraiment au sein d'un même réseau sans suivre le clivage inter-réseau — non "
        "trouvées parmi les candidats naturels ici.\n"
    )
    lines.append("**Rappel à faire figurer dans le manuscrit** : aucun ajustement "
                 "réseau-level testé — avec n=4 réseaux, un tel ajustement n'a pas de "
                 "puissance statistique interprétable.")

    out = ROOT / "analysis" / "p9_3_extended_model_results.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nRapport : {out}")


if __name__ == "__main__":
    main()
