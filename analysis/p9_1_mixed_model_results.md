# P9.1 (R2.1) — Modèle mixte ΔR² ~ hétérophilie locale, 4 réseaux

Base : `analysis/per_station_dataset.csv` (47 stations, 4 réseaux : ['beijing', 'czt', 'london', 'madrid']). Stations à h_i indéfini exclues : [['madrid', 'MENDEZ ALVARO']].

## Topologie distance

n = 46 stations, 4 réseaux (intercept aléatoire).

**Effet fixe h_i** : β = -1.0646, SE = 0.5494, IC95% = [-2.1415, +0.0123], p = 0.05267

**ICC** (part de variance attribuable au réseau) = 0.433 (var_réseau=0.05291, var_résiduelle=0.06936)


**Spearman intra-réseau** :

| Réseau | n | ρ | p | Note |
|---|---|---|---|---|
| beijing | 12 | -0.035 | 0.9141 |  |
| czt | 20 | +0.186 | 0.4312 |  |
| london | 8 | -0.333 | 0.4198 |  |
| madrid | 6 | +0.486 | 0.3287 | n=6 < 8 — inférence peu fiable, à ne pas sur-interpréter |

## Topologie correlation

n = 46 stations, 4 réseaux (intercept aléatoire).

**Effet fixe h_i** : β = -1.0732, SE = 0.6276, IC95% = [-2.3034, +0.1569], p = 0.08727

**ICC** (part de variance attribuable au réseau) = 0.422 (var_réseau=0.06032, var_résiduelle=0.08278)


**Spearman intra-réseau** :

| Réseau | n | ρ | p | Note |
|---|---|---|---|---|
| beijing | 12 | -0.727 | 0.007355 |  |
| czt | 20 | -0.155 | 0.5144 |  |
| london | 8 | -0.048 | 0.9108 |  |
| madrid | 6 | +0.829 | 0.04156 | n=6 < 8 — inférence peu fiable, à ne pas sur-interpréter |
