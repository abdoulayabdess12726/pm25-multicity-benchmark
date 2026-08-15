# Journal des changements de valeurs — cycle de révision 20265149

Une ligne par valeur qui change entre la version soumise et la version
révisée : table, ancienne valeur, nouvelle valeur, cause. Alimente la lettre
de réponse aux relecteurs (cf. P11 point 6). Complété au fil des étapes, pas
seulement à la fin.

## Écart le plus important de la resoumission

**Table 3, Madrid / XGBoost, R² : 0.8061 → 0.6758 (Δ = −0.1304).** Cause :
MENDEZ ALVARO (PM2.5 constant sur tout le train, régime test normal — voir
`results/mendez_alvaro_diagnostic.md`) fait s'effondrer XGBoost
spécifiquement (R²=−0.254 sur cette station seule : le modèle apprend une
constante sur des lags eux-mêmes constants pendant l'entraînement, aucune
capacité d'extrapolation face à la variance du test). C'est le plus grand
écart chiffré entre la version soumise et la version révisée de tout ce
cycle de révision à ce stade. Décision d'agrégation : aucune exception,
MENDEZ ALVARO entre dans l'agrégat XGBoost comme dans toutes les autres
lignes (cf. `REVISION_BRIEF.md`).

**Provenance Table 3 / GCN-Transformer (répond à la demande du relecteur 1)**
: la ligne GCN-Transformer de la Table 3 utilise la topologie **distance**
(cf. `results/external_baselines_tables.md`, note : « GCN-Transformer:
topologie distance »). Linear-Transformer est temporel pur, identique quelle
que soit la topologie.

## P1 — Correctif MENDEZ ALVARO (Madrid = 7 stations, source unique de stations)

Cause commune à toutes les lignes ci-dessous : MENDEZ ALVARO (Madrid) était
exclue du reporting/agrégat par 3 scripts indépendants
(`10_external_baselines.py`, `13_edge_pruning.py`, `e6_k_sensitivity.py` /
`e8_k_sensitivity_3seeds.py`), chacun via son propre dict `EXCLUDE` local —
écart au protocole publié (manuscrit §3.3 : la station est conservée dans le
benchmark de prévision, exclue seulement de l'analyse station-level §5.5).
Corrigé par une source unique (`configs/stations/*.yaml` +
`src.stations.load_stations`). Décision d'agrégation : MENDEZ ALVARO entre
partout sans traitement particulier, y compris pour les modèles où elle
échoue (voir `REVISION_BRIEF.md`).

| Table | Ligne | Ancienne valeur (6 stations) | Nouvelle valeur (7 stations) | Cause | Statut |
|---|---|---|---|---|---|
| Table 3 | Madrid / Linear-Transformer, R² | 0.817 | 0.8140 | Ré-agrégation depuis le JSON canonique (sans ré-entraînement) | ✅ confirmé |
| Table 3 | Madrid / GCN-Transformer, R² | 0.472 | 0.4927 | Idem — MENDEZ ALVARO isolée dans le graphe corrélation, relativement facile à prédire une fois incluse | ✅ confirmé |
| Table 3 | Madrid / Persistence, R² | 0.7986 | 0.7961 | Reconstruit (6 stations existantes + MENDEZ ALVARO calculée en diagnostic) | ✅ confirmé, calcul sans ré-entraînement des 6 autres stations |
| Table 3 | Madrid / ARIMA, R² | 0.8114 | 0.8073 | Idem | ✅ confirmé |
| Table 3 | Madrid / XGBoost, R² | 0.8061 | **0.6758** | MENDEZ ALVARO : R²=−0.254 sur cette station seule (XGBoost apprend une constante sur un train à variance nulle, aucune capacité d'extrapolation face au régime test normal) — tire l'agrégat Madrid fortement vers le bas | ✅ confirmé — **variation la plus importante de cette étape** |
| Table 3 | Madrid / LSTM (seed 42), R² | 0.7990 | 0.7971 | Idem Persistence/ARIMA — architecture skip-persistance, MENDEZ ALVARO bien gérée malgré le train pathologique | ✅ confirmé |
| Table 7 | Madrid, k=5 (2 topologies × 3 seeds, 6 lignes) | 6 stations | 7 stations | k=5 ne ré-entraîne jamais rien (pull direct du JSON canonique, qui a toujours eu les 7 stations) — réparé par simple re-lecture, comme la Table 3. Nouvelles valeurs : distance ΔR² = −0.3282/−0.3192/−0.3165 (seeds 42/123/777), correlation ΔR² = −0.3913/−0.3879/−0.3594 | ✅ confirmé, zéro calcul |
| Table 7 | Madrid, k∈{3,8} (2 topologies × 2 k × 3 seeds, 12 lignes) | 6 stations | 7 stations (en attente) | Aucun checkpoint ni prédiction par nœud jamais persisté pour ces cellules (vérifié : `grep torch.save` → 0 résultat dans e6/e8 ; `train_gcn_r2` ne retourne que le R² agrégé déjà tronqué) — ré-entraînement complet requis | ⏳ en attente (E9, P5) |
| §6.2.1 (pruning) | Madrid, toutes lignes | 6 stations | 7 stations (en attente) | Idem — ré-entraînement complet requis (E10, P5) | ⏳ en attente (E10, P5) |

**Non modifié** : Beijing et London (aucune exclusion n'y a jamais été
appliquée, tous les scripts chargeaient déjà la totalité des stations pour
ces deux villes).
