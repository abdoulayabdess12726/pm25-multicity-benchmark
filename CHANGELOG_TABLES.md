# Journal des changements de valeurs — cycle de révision 20265149

Une ligne par valeur qui change entre la version soumise et la version
révisée : table, ancienne valeur, nouvelle valeur, cause. Alimente la lettre
de réponse aux relecteurs (cf. P11 point 6). Complété au fil des étapes, pas
seulement à la fin.

## P2 — Convention ddof=1 unique (Table 4 vs Table 7)

Cause : défaut IMPLICITE différent entre `numpy.std()` (ddof=0) et
`pandas.Series.std()` (ddof=1) selon que le code agrégeait une liste/array ou
un DataFrame — pas un choix explicite incohérent. Centralisé dans
`src/stats.py::agg_mean_std`, qui ne délègue à aucun défaut de librairie
(ddof=1 toujours passé explicitement). Tous les écarts-types inter-seeds
recalculés depuis les données brutes déjà persistées (listes par-seed dans
les JSON canoniques, lignes par-seed dans les CSV existants) — **aucun
ré-entraînement**. Hors périmètre, documenté sur place : dispersion
intra-série de `05_compute_heterogeneity_v2.py` (n grand, sans effet
numérique) et dispersion inter-station de `cohens_d`
(`07_statistical_analysis.py`).

### Table 4 (tests statistiques, `07_statistical_analysis.py`)

| Ville | Topologie | ΔR² std (ddof=0, soumis) | ΔR² std (ddof=1, corrigé) |
|---|---|---|---|
| Beijing | Distance | 0.0001 | 0.0001 |
| Beijing | Correlation | 0.0007 | 0.0008 |
| **London** | **Distance** | **0.0211** | **0.0258 (≈0.026)** |
| **London** | **Correlation** | **0.0050** | **0.0061 (≈0.006)** |
| Madrid | Distance | 0.0050 | 0.0061 |
| Madrid | Correlation | 0.0143 | 0.0175 |

Vérification demandée confirmée numériquement (`tests/test_ddof.py::test_london_table4_table7_reconciliation`) :
Londres passe de 0.021/0.005 à 0.026/0.006 — valeurs désormais identiques à
la Table 7 (k=5), relation exacte `std_ddof1 = std_ddof0 × √(3/2)` pour n=3
seeds (vérifiée bit-à-bit, pas une coïncidence d'arrondi).

### JSON canoniques (`results/{beijing,london,madrid}/multistation_results.json`)

36 champs `MAE_std`/`RMSE_std`/`R2_std` corrigés (2 topologies × 2 modèles ×
3 métriques × 3 villes) — champs dormants, non consommés ailleurs dans le
dépôt (vérifié), mais présents dans la source de vérité canonique. Les
moyennes ne changent pas (vérifié par assertion lors du recalcul).

### `results/external_baselines_tables.md` (Table 3)

Lignes Linear-Transformer / GCN-Transformer : std corrigé (ex. Madrid
Linear-Transformer 0.817±0.001→0.814±0.001 — le ± change de convention,
pas de source ; voir aussi le changement de moyenne documenté plus haut,
cause différente/P1). Lignes Persistence/ARIMA/XGBoost/LSTM : std LSTM
corrigé (seul modèle stochastique sur 3 seeds parmi les baselines rapides) ;
ARIMA/XGBoost/Persistence sont déterministes (n=1, std=0, non affectés).

### `results/sensitivity_k_canonical.csv` (Table 7 canonique, partielle)

6 lignes k=5 recalculées gratuitement depuis le JSON (déjà ddof=1 à la
source après le correctif ci-dessus).

**Incident, détecté et corrigé avant commit** : la fusion de
`08_sensitivity_k.py` supprimait TOUTES les lignes d'une ville listée dans
`--cities` avant de concaténer les nouvelles (`old[~old.city.isin(args.cities)]`)
au lieu de fusionner par clé exacte (city, k, topology). En rafraîchissant
les 6 lignes k=5 (beijing/london/madrid), cette fusion a silencieusement
effacé la ligne historique `beijing k=3 distance (recompute canonique)` —
détecté via `git diff` (7 lignes → 6), pas par une vérification a priori.

Première tentative de réparation (rejetée après revue) : reconstruire
`delta_r2_std` par conversion exacte ×√(3/2) (0.0005 ddof=0 → 0.0006 ddof=1).
Mathématiquement correct, mais **inacceptable comme donnée de table** : c'est
une valeur dérivée d'une valeur déjà affichée (le ddof=0 stocké), sans les
observations par-seed brutes derrière — exactement ce que ce cycle de
révision doit éliminer. Recherche de ces valeurs brutes (git log, logs/,
JSON canoniques) : **aucune trace** — un seul commit (`0413ddb`) a créé le
CSV directement avec mean/std déjà agrégés, aucun fichier annexe avec les 3
R² par seed. Le run a réellement eu lieu (protocole canonique complet,
documenté dans `sensitivity_k_canonical_NOTE.md`) mais sa sortie brute n'a
jamais été persistée.

**Correction finale** : `delta_r2_std`/`gcn_r2_std` mis à vide,
`source = "UNRECOVERABLE (...)"`. `delta_r2_mean` (−0.0134) conservé —
la moyenne ne dépend pas du ddof, ce n'est pas une valeur en question.
Régénération par E9 (P5) requise pour cette cellule.

**Décompte demandé** : sur les 7 lignes de `sensitivity_k_canonical.csv`,
**1 seule** n'a pas de valeurs par-seed brutes retrouvables (celle ci-dessus).
Les 6 lignes `json(k5)` sont recalculées depuis les listes `R2`/`MAE`/`RMSE`
par-seed des JSON canoniques, vérifiées présentes et correctes (test
`tests/test_ddof.py::test_table7_k5_matches_table4_exactly`).

**Correctif structurel** : `src/csv_upsert.py::upsert_rows(csv_path, new_df,
key_cols)` — fusion par clé exacte, ne supprime jamais de ligne dont la clé
n'est pas dans les nouvelles données. `08_sensitivity_k.py` refactoré pour
l'utiliser. Test de régression :
`tests/test_csv_upsert.py::test_upsert_does_not_drop_untargeted_rows_same_city`
(reproduit l'incident exact et vérifie qu'il ne se reproduit plus).

**Suivi — même pattern trouvé et corrigé dans TROIS autres scripts** (recherche
élargie au motif dans tout le dépôt, pas seulement aux deux initialement
signalés) : `10_external_baselines.py`, `13_edge_pruning.py` (script d'E10)
et **`11_diagnostics.py`** (celui-ci découvert uniquement grâce à la
recherche élargie — pas dans le signalement initial). Les trois avaient
exactement `old[~old.city.isin(args.cities)]`. Tous les trois migrés vers
`upsert_rows` avec leur propre clé (`city+model+station+seed` pour E1,
`city+keep_frac+seed+station` pour E10, `city+topology+experiment+seed` pour
E4/E5). `upsert_rows` lui-même durci : comparaison de clé en `str` (une
colonne à valeurs mixtes comme `seed` ∈ {42, "-"} se relit en dtype `object`
depuis le CSV, ce qui aurait cassé une comparaison de tuples typée).

Recherche exhaustive confirmée : tout autre `.city ==`/`isin` restant dans le
dépôt est soit un upsert déjà correct par clé exacte (e6/e8/14_sota_baselines,
préexistants et corrects), soit une lecture/agrégation qui ne réécrit jamais
partiellement un CSV (`results/export_per_station.py`,
`12_per_station_heterophily.py` — reconstruction complète à chaque run, donc
non concernés par cette classe de bug).

Tests de régression : un test par site migré (vérifie l'usage effectif
d'`upsert_rows` avec les bonnes colonnes-clé) + un garde-fou global qui
échoue si le motif `old[~old.city` réapparaît n'importe où dans le dépôt
(`tests/test_csv_upsert.py`).

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
