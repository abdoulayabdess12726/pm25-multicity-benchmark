# Changelog des valeurs manuscrit — soumission initiale → version révisée (Paper ID 20265149)

> Document de travail pour la lettre de réponse aux relecteurs. Regroupé par
> cause racine plutôt que par table, sur demande explicite. Chaque ligne est
> traçable à un commit et/ou une section de `CHANGELOG_TABLES.md`.

Numérotation courante (voir `README.md`, « Reproducing the paper's tables ») :
T1 h(D) · T2 benchmark · T3 baselines externes · T4 tests statistiques ·
T5 corrélation inter-villes · T6 ΔR² par station · T7 sensibilité k ·
T8 over-smoothing/GAT · T9 diagnostics.

## 1. MENDEZ ALVARO au chargement (Madrid 6 → 7 stations)

Trois scripts en aval (`10_external_baselines.py`, `13_edge_pruning.py`,
`e6_k_sensitivity.py`/`e8_k_sensitivity_3seeds.py`) portaient chacun un
`EXCLUDE = {"madrid": {"MENDEZ ALVARO"}}` local, appliqué au reporting/agrégat.
Corrigé par une source unique `configs/stations/*.yaml` + `src.stations.load_stations`.

| Table | Ancienne valeur publiée | Nouvelle valeur | Cause |
|---|---|---|---|
| Table 3 | Madrid / Persistence (t−1) : R²=0.7986 ± 0.0000, MAE=2.8753, RMSE=4.8603 (6 stations) | R²=0.7961 ± 0.0000, MAE=2.8726, RMSE=4.8368 (7 stations) | Agrégat calculé sur 6 stations ; re-run complet de `10_external_baselines.py` avec la liste de stations canonique |
| Table 3 | Madrid / ARIMA : R²=0.8114 ± 0.0000, MAE=2.8527, RMSE=4.7038 | R²=0.8073 ± 0.0000, MAE=2.8533, RMSE=4.7023 | Idem |
| Table 3 | Madrid / XGBoost : R²=0.8061 ± 0.0000, MAE=3.0147, RMSE=4.7688 | **R²=0.6758 ± 0.0000**, MAE=3.6147, RMSE=6.0993 | Idem — MENDEZ ALVARO donne R²=−0.254 pour XGBoost seule (train PM2.5 constant, régime test normal → aucune extrapolation) ; **plus grand écart chiffré de tout le cycle (Δ=−0.1303)** |
| Table 3 | Madrid / LSTM : R²=0.7990 ± 0.0000, MAE=2.8801, RMSE=4.8551 | R²=0.7965 ± 0.0000, MAE=2.8771, RMSE=4.8316 | Idem (moyenne 3 seeds) |
| Table 3 | Madrid / Linear-Transformer : R²=0.817 ± 0.001 | R²=0.8140 ± 0.0013 | `agg_from_json()` ré-agrégeait sur 6 stations au lieu de lire le champ agrégat 7-stations déjà présent dans le JSON canonique — réparé sans ré-entraînement |
| Table 3 | Madrid / GCN-Transformer (distance) : R²=0.472 ± 0.005 | R²=0.4927 ± 0.0064 | Idem — MENDEZ ALVARO relativement facile à prédire une fois incluse |
| Table 7 | Madrid / distance k=5 : ΔR²=−0.3450 ± 0.0057 | ΔR²=−0.3213 ± 0.0061 | Agrégat k-sensitivity tronqué à 6 stations ; k=5 ne ré-entraîne rien (relecture directe du JSON canonique 7-stations) |
| Table 7 | Madrid / distance k=3 : ΔR²=−0.3045 ± 0.0196 | ΔR²=−0.2828 ± 0.0187 | Ré-entraînement complet 7 stations (E9) |
| Table 7 | Madrid / distance k=8 : ΔR²=−0.3711 ± 0.0044 | ΔR²=−0.3478 ± 0.0076 | Idem (E9) |
| §6.2.1 (courbe d'élagage, pas de table numérotée) | Madrid, ancre plein graphe keep_frac=1.0 : R² moyen = 0.4716 (6 stations, 15 conditions) | R² moyen = 0.4877 (7 stations) | Ré-entraînement complet E10 ; l'ancre devient cohérente avec Table 2 (écart &lt;0.008, tolérance 3×std) |

**Non affecté, à affirmer tel quel** : Table 2 (distance), Table 4 (distance),
Beijing et London partout (aucune exclusion n'y a jamais existé).

**Précision de traçabilité** : `CHANGELOG_TABLES.md` §P1 cite comme
« anciennes valeurs » Table 7 Madrid −0.3019/−0.3685 (distance k=3/k=8) et
−0.3437/−0.4540 (correlation k=3/k=8). Ce ne sont pas les valeurs soumises :
ce sont des ΔR² 6-stations recalculés contre la référence Linear 7-stations
déjà corrigée. Les valeurs du tableau ci-dessus sont celles réellement
publiées.

_Sources : `CHANGELOG_TABLES.md` §« P1 — Correctif MENDEZ ALVARO » et
§« Écart le plus important de la resoumission » ; `AUDIT.md` L50-52 ;
`REVISION_BRIEF.md` §« Décision d'agrégation MENDEZ ALVARO ». Commits :
introduction `0a041de`, correctif source unique `b470578`, re-runs E9/E10
`3c14c9a`, re-run Table 3 baselines rapides `3ad9bff` (2026-08-24, P11.2 —
42 lignes SUSPECT_6STATION remplacées par 48 lignes 7-stations, flag
`[SUSPECT]` disparu de Table 3). Lignes brutes : `results/raw_results.csv`,
run_id `10_external_baselines_1787566274_141d9e9e` (n_stations=7) vs
`migrated_10_external_baselines_madrid` (n_stations=6)._

## 2. Convention ddof incohérente (Table 4 en ddof=0, Table 7 en ddof=1 → ddof=1 partout)

Défaut implicite divergent entre `numpy.std()` (ddof=0) et
`pandas.Series.std()` (ddof=1). Facteur exact √(3/2) ≈ 1.2247 sur n=3 seeds.
Aucune moyenne ne change, aucun ré-entraînement.

| Table | Ancienne valeur publiée | Nouvelle valeur | Cause |
|---|---|---|---|
| Table 4 | Beijing / distance : ΔR² std = 0.0001 | 0.0001 | ddof=0 → ddof=1 ; inchangé à 4 décimales |
| Table 4 | Beijing / correlation : ΔR² std = 0.0007 | 0.0008 | Idem |
| Table 4 | **London / distance : ΔR² std = 0.0211** | **0.0258** | Idem — écart le plus visible ; met Table 4 en accord exact avec Table 7 (k=5) |
| Table 4 | **London / correlation : ΔR² std = 0.0050** | **0.0061** | Idem |
| Table 4 | Madrid / distance : ΔR² std = 0.0050 | 0.0061 | Idem |
| Table 4 | Madrid / correlation : ΔR² std = 0.0143 | 0.0325 *(valeur finale — voir aussi cause racine 3)* | Idem |
| Table 2 | Beijing : GCN dist std=0.000457, Linear std=0.000532, GCN corr std=0.000698 | 0.000559 / 0.000652 / 0.000855 (±0.0006 / ±0.0007 / ±0.0009) | Recalcul ddof=1 des 3 JSON canoniques |
| Table 2 | London : GCN dist std=0.020177, Linear std=0.003257, GCN corr std=0.003030 | ±0.0247 / ±0.0040 / ±0.0037 | Idem |
| Table 2 | Madrid : GCN dist std=0.005223, Linear std=0.001044, GCN corr std=0.013726 | ±0.0064 / ±0.0013 / ±0.0325 *(corr, valeur finale)* | Idem |
| Table 3 | Beijing/London / GCN-Transformer, Linear-Transformer, LSTM : ± en ddof=0 | ± en ddof=1 (voir `manuscript/tables/table3_external_baselines.md` pour les valeurs exactes par ligne) | Recalcul ddof=1 |
| Table 7 | *(inchangée par cette cause)* | — | Table 7 était **déjà** en ddof=1 dans la version soumise — c'est Table 4 qui a été alignée sur elle, pas l'inverse |

**Incident collatéral, sans effet sur une valeur publiée** : la fusion CSV de
`08_sensitivity_k.py` a effacé une ligne historique de
`results/sensitivity_k_canonical.csv` (Beijing k=3 distance, recompute) ;
marquée `UNRECOVERABLE`, régénérée par E9. Artefact interne, pas une cellule
de Table 7 du manuscrit.

_Sources : `CHANGELOG_TABLES.md` §« P2 — Convention ddof=1 unique » ;
`REVISION_BRIEF.md` §« Deux causes racines », §« Conventions statistiques » ;
`AUDIT.md` L97-102. Commit `6e604bd`. Vérification :
`tests/test_ddof.py::test_london_table4_table7_reconciliation`._

## 3. Tri NaN de `build_correlation_graph` (Madrid / topologie correlation uniquement)

`06_train_multistation.py::build_correlation_graph()` : `np.argsort(corr[i])[::-1]`
plaçait les corrélations NaN **en tête** du tri décroissant ; le seuil
`corr>0` éliminait ensuite l'arête sans réattribuer le rang au candidat réel
suivant. Madrid uniquement (MENDEZ ALVARO, variance train nulle → 12 NaN
hors diagonale) ; Beijing/London ont 0 NaN, comptage d'arêtes exact à tous
les k.

| Table | Ancienne valeur publiée | Nouvelle valeur | Cause |
|---|---|---|---|
| Table 2 | Madrid / correlation : GCN R²=0.4345 ± 0.0168 ; ΔR²=−0.3795 ± 0.0175 (arrondi −0.380 dans le manuscrit soumis) | **GCN R²=0.3996 ± 0.0335 ; ΔR²=−0.4144 ± 0.0325** | Graphe entraîné avec **24 arêtes au lieu de 35** à k=5 : MENDEZ ALVARO totalement isolée, les 6 autres stations perdant chacune 1 voisin sur 5. Après correctif : 30 arêtes (plafond réel = 6 stations à corrélation définie) |
| Table 3 | Madrid / GCN-Transformer (correlation) : 0.4345 ± 0.0168 | 0.3996 ± 0.0335 | Même graphe canonique |
| Table 3 | Madrid / STGCN, Graph WaveNet (correlation) | Recalculé (écart faible — adjacence adaptative, peu sensible au graphe d'entrée) | `14_sota_baselines.py` réutilise `b.build_correlation_graph` |
| Table 4 | Madrid / correlation : ΔR²=−0.3795 ± 0.0175 (agrégat) ; Cohen's d, Wilcoxon calculés sur graphe pré-correctif | **ΔR²=−0.4144 ± 0.0325 ; Cohen's d=−2.11** (Wilcoxon p et 7/7 GCN&lt;Linear inchangés) | Dérivé de Table 2 pour l'agrégat. **Le Cohen's d/Wilcoxon par-station n'avait pas été recalculé** après le correctif du graphe (le rerun E17 n'écrit que l'agrégat) — trouvé et corrigé le 2026-08-24 en préparant ce récapitulatif (commit `1c30f84`) : rerun dédié `06_train_multistation.py --city madrid --graph correlation`, Cohen's d passe de −2.01 (calculé par erreur sur l'ancien graphe) à −2.11 |
| Table 5 | ligne correlation : ρ=−0.500, p=0.6667 | **ρ=−1.000, p=0.0000** | Recalcul automatique depuis Table 2 : Madrid devenant plus dégradée que London, l'ordre h(D) est monotone en topologie correlation |
| Table 6 | Madrid / correlation, 7 lignes par station — restées sous l'ancien graphe jusqu'au 2026-08-24 | **Régénérées sous le graphe corrigé** (ex. MENDEZ ALVARO GCN R²=0.6590, ΔR²=−0.1338) | Même correctif que Table 4 ci-dessus, même commit `1c30f84` |
| Table 7 | Madrid / correlation k=3 : ΔR²=−0.3145 ± 0.0059 (12 arêtes au lieu de 21) | **−0.3692 ± 0.0239** (18 arêtes) | Même bug de tri, aux 3 valeurs de k |
| Table 7 | Madrid / correlation k=5 : ΔR²=−0.3795 ± 0.0175 | **−0.4144 ± 0.0325** | Idem |
| Table 7 | Madrid / correlation k=8 : ΔR²=−0.4143 (30 arêtes) | −0.4144 ± 0.0325 (30 arêtes) | Graphe identique avant/après à k=8 (déjà plafonné) → valeur pratiquement inchangée ; **k=5 et k=8 donnent désormais le même graphe Madrid** |

**Non affecté, à affirmer explicitement** : topologie distance (toutes
villes) ; Beijing et London (toutes topologies, tous k) ; Table 1 (h(D),
`nanmean` sûr) ; §6.2.1/E10/E12/E14 (graphe de base = distance) ; Table 8/E13
(distance seule, fonction de tri différente, vérifiée saine sur les
données Madrid réelles) ; `12_per_station_heterophily.py` (masquage NaN
explicite).

**Conséquence de rédaction** : le renversement Madrid/London (Madrid
h(D)=0.728 &gt; London 0.656 mais moins dégradée) **ne subsiste plus** en
topologie correlation (Madrid −0.4144 &lt; London −0.4014) ; il **subsiste
inchangé** en topologie distance (Madrid −0.3213 vs London −0.3754).

_Sources : `CHANGELOG_TABLES.md` §« PRIORITÉ ABSOLUE — ampleur du bug de tri
NaN » et §« RÉSOLU — correctif appliqué et 15 runs relancés (2026-08-21) ».
Commits : constat `1b09a36`, correctif + 10 tests de régression `9992793`
(`tests/test_correlation_graph.py`), résultats des 15 runs `ca626dc`,
réparation Table 4/6 par-station + JSON Madrid `1c30f84` (2026-08-24).
Lignes brutes post-correctif : `results/raw_results.csv`,
git_commit=`99927939…` (agrégat) et run_id=`06_madrid_1787591639_ab7abd7b`
(par-station, 2026-08-24)._

## 4. Table 8 non persistée (over-smoothing / GAT)

`09_controls_oversmoothing.py` était le seul script du dépôt à n'écrire
**aucun** résultat persistant (sortie console uniquement). Analyse de
couverture P3 : **36/36 conditions manquantes** dans `raw_results.csv`. Le
script a été instrumenté puis exécuté (E13 : 3 villes × 4 variantes ×
3 seeds, topologie distance).

| Table | Ancienne valeur publiée | Nouvelle valeur | Cause |
|---|---|---|---|
| Table 8 | Les 12 cellules R² (3 villes × linear1L/gcn1L/gcn2L/gat2L) de la version soumise : **valeurs non retrouvables** — aucune ligne persistée, aucun log/notebook/historique shell ; aucun commit ne les a introduites | Beijing 0.8790±0.0020 / 0.8852±0.0082 / 0.8825±0.0031 / 0.8619±0.0027 · London 0.6938±0.0202 / 0.2523±0.0122 / 0.1366±0.0077 / −0.0705±0.0220 · Madrid 0.6460±0.0050 / 0.4968±0.0086 / 0.4588±0.0037 / 0.3090±0.0045 | Table générée pour la première fois depuis des runs persistés (E13, 36 runs, ddof=1) |
| §6.2 (prose adossée à Table 8) | Beijing : ΔR² gcn1L vs Linear = **−0.008**, Wilcoxon p=0.005 | **+0.006**, p=**0.9939** | **Signe opposé et non significatif** — le run reproductible contredit l'affirmation « underperforms the baseline at every city » |
| §6.2 | London : ΔR² = **−0.279**, p=0.004 | **−0.442**, p=0.0039 | Même sens et même significativité, magnitude 58 % plus négative |
| §6.2 | Madrid : ΔR² = **−0.283**, p=0.008 | **−0.149**, p=0.0078 | Même sens et même significativité, magnitude ≈ moitié moins négative |

**Réserve à consigner telle quelle dans la lettre** : la source des trois
valeurs soumises (−0.279 / −0.283 / −0.008) **n'a pas été identifiée** —
recherche textuelle + pickaxe git sur tout l'historique, recalcul exhaustif
de toutes les combinaisons variante × référence Linear testées, aucune trace
de l'exécution console d'origine. Conclusion actée, pas une investigation
ouverte.

_Sources : `CHANGELOG_TABLES.md` §« P3 — results/raw_results.csv unifié »
et §« E13 — Table 8, over-smoothing/GAT (P5, 2026-08-17/18) — CONTREDIT LE
MANUSCRIT POUR BEIJING ». Commits : instrumentation `6474dc3`, table marquée
MISSING DATA `bc2f489`, remplissage par E13 `0fa3f05`, clôture d'investigation
`ebfc2f5`. 360 lignes `config_path=09_controls_oversmoothing.py` dans
`results/raw_results.csv`._

---

## Réserves de traçabilité (à lire avant rédaction de la lettre)

1. **Les « anciennes valeurs » Table 7 du §P1 du CHANGELOG (cause 1) sont des
   recalculs hybrides**, pas les valeurs soumises (détail en fin de section 1).
2. **Différences résiduelles au 4ᵉ chiffre**, hors des 4 causes racines,
   issues de la régénération à pleine précision depuis `raw_results.csv` :
   Table 7 Beijing/distance k=3 (−0.0138 → −0.0139), Beijing/correlation k=3
   (std 0.0028 → 0.0027), London/distance k=3 (−0.3296 → −0.3295),
   London/distance k=8 (std 0.0145 → 0.0144), Beijing/correlation k=8
   (std 0.0011 → 0.0012). Pur arrondi, à ne pas présenter comme des
   corrections.
3. **Chiffre « 0.011 » (résumé, §5.2, §6.1)** : valeur affichée inchangée,
   mais provenance à corriger — 0.010593 (moyenne 3 seeds,
   Beijing/Graph WaveNet) au lieu de 0.011351 (seed 42 seul). Pas une des
   4 causes racines, mais un point de la lettre (`CHANGELOG_TABLES.md` §E11 ;
   commit `0fa3f05`).
4. **`results/table7_k_sensitivity.csv` et `results/{city}/multistation_results.json`
   ne sont pas la source des tables du manuscrit** (uniquement
   `raw_results.csv` → `scripts/regenerate_tables.py`) — le premier contient
   pour Madrid/correlation une valeur intermédiaire gelée (−0.3795) qui n'a
   plus cours ; ne pas le citer par erreur comme référence.
