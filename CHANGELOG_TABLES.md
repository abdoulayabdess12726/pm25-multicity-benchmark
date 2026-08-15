# Journal des changements de valeurs — cycle de révision 20265149

Une ligne par valeur qui change entre la version soumise et la version
révisée : table, ancienne valeur, nouvelle valeur, cause. Alimente la lettre
de réponse aux relecteurs (cf. P11 point 6). Complété au fil des étapes, pas
seulement à la fin.

## P4 — regenerate_tables.py, 24 assertions (17 PASS, 3 FAIL, 4 MISSING DATA)

`scripts/regenerate_tables.py` lit `raw_results.csv` exclusivement (une
exception documentée : h(D)/Table 1 vient de
`results/heterogeneity_index_v2.csv`, hors schéma raw_results depuis P3) et
émet les Tables 1-9 en Markdown + .docx natif (python-docx, table Word
directement copiable) dans `manuscript/tables/`.

**Bug de pipeline trouvé et corrigé avant de committer** (pas l'assertion
qui l'a révélé) : `raw_results.csv` contenait des lignes DOUBLES pour
chaque condition k=5 — une depuis le JSON canonique (pleine précision,
migration P3 source 1) et une depuis `e6_k_sensitivity.csv` (arrondie à 4
décimales lors de son écriture originale par e6/e8), sous deux `run_id`
différents donc jamais détectées comme collision par `append_run`. Cause :
`migrate_k_sensitivity()` migrait k∈{3,5,8} sans exclure k=5, qui est
pourtant un doublon exact de `migrate_canonical()`. Corrigé : k=5 exclu de
cette source de migration. `raw_results.csv` régénéré (816 lignes au lieu
de 834).

**3 FAIL, cause identifiée, pas un mystère** : ancre de pruning
(keep_frac=1.0) vs Table 2 GCN-Transformer/distance pour Madrid, les 3
seeds. `n_stations` diffère (6 vs 7) — c'est exactement le
SUSPECT_6STATION déjà documenté en P1 (MENDEZ ALVARO absente du pruning
Madrid), pas une nouvelle divergence. Le message d'assertion l'identifie
explicitement. Attend le re-run E10.

**4 MISSING DATA** : ancre de pruning Beijing/London seeds 123/777 — cohérent
avec le budget consigné en P3 (20 runs edge-pruning manquants).

Table 3 : 4 décimales + colonne provenance (3seed_mean/primary_seed/
deterministic) ; les lignes Madrid ARIMA/XGBoost/LSTM/Persistence portent
désormais [SUSPECT] dans la cellule R² (le flag manquait dans une première
version — ajouté après avoir remarqué que ces lignes restaient encore à 6
stations sans indication visible dans la table générée).

**Assertion d'unicité structurelle ajoutée** (`assert_no_duplicate_conditions_across_run_ids`,
25e assertion) : le doublon k=5 est passé inaperçu par `append_run` parce
que sa clé de collision inclut `run_id` — deux `run_id` différents pour la
même condition logique (city, model, topology, k, keep_frac, variant, seed,
station, split) ne sont jamais comparés entre eux. Cette assertion groupe
sur l'identité logique SANS `run_id` et échoue si `nunique(run_id) > 1`
pour un groupe. Lancée sur les 816 lignes actuelles : **0 autre doublon
trouvé**. 5 tests dédiés (`tests/test_regenerate_tables.py`), dont la
reproduction exacte de l'incident k=5 et une régression sur le fichier réel.

## P3 — results/raw_results.csv unifié

`src/results_io.py::append_run/load_results` : append-only strict (mode
fichier `"a"` pur, jamais de réécriture), refuse toute clé
(run_id,city,model,topology,k,seed,station) déjà présente. Aucune fonction
de mise à jour/suppression exposée (vérifié par test, pas juste documenté).

**9 scripts instrumentés** (06, 08, 09, 10, 11, 13, 14, e6, e8) : chacun
appelle `append_run` en fin de run avec git_commit, config_path, split_hash
(réel — reconstruit depuis T/t1/t2/SEQ_LEN, pas un placeholder) et
checkpoint_id="no_checkpoint_saved" (aucun script ne sauvegarde de poids,
confirmé en P2). `09_controls_oversmoothing.py` est le seul script qui
n'écrivait AUCUN résultat persistant avant P3 (console only) — instrumenté
quand même, c'est la seule source possible pour la Table 8.

**Deux tensions de schéma, corrigées AVANT commit (relecture) — colonnes
dédiées, pas de champ à double sémantique ni de structure pliée dans une
chaîne** :
- `13_edge_pruning.py` a un axe "niveau d'élagage" absent du schéma initial
  → **colonne `keep_frac` dédiée**, ajoutée (pas réutilisation de `k`, qui
  reste NULL pour ces lignes et réservé au k-NN partout ailleurs). Une
  première version repliait `keep_frac` dans `k` avec une note — rejetée sur
  relecture : un champ à double sémantique dans le fichier canonique aurait
  piégé les assertions de P4 et un relecteur ouvrant le CSV n'aurait pas pu
  deviner la convention sans lire le code.
- `11_diagnostics.py`/`09_controls_oversmoothing.py` (real/shuffled_graph/
  no_meteorology, 1-layer/2-layer) n'ont pas de colonne "expérience"/"variante"
  → **colonne `variant` dédiée**, ajoutée. Une première version pliait
  l'expérience dans le nom du modèle (`GCN-Transformer[shuffled_graph]`) —
  également rejetée : le nom du modèle ne doit pas être un champ structuré
  déguisé.
- `KEY_COLS` étendu en conséquence (`variant`, `keep_frac` ajoutés) — sans
  ça, deux variantes/niveaux différents de la même condition auraient
  partagé une clé et se seraient silencieusement écrasés au lieu d'être
  refusés par `append_run`.
- **Faux-positif détecté et corrigé pendant l'analyse de couverture** :
  l'instrumentation initiale de `09_controls_oversmoothing.py` nommait son
  Linear de référence `"Linear-Transformer"` sans variante — identique au nom
  canonique de la Table 2, alors que c'est un entraînement séparé (topology
  fixée, pas topology=""). Aurait fait croire que 9 conditions Table 8
  existaient déjà. Corrigé : `variant="1layer"` explicite le distingue
  maintenant, sans avoir besoin de décorer le nom du modèle.
- **Bug de robustesse trouvé et corrigé pendant les tests de la correction** :
  `append_run`/`_row_keys` comparaient les clés en `str()` nu — un champ vide
  écrit `""` se relit en `NaN` depuis le CSV, qui se serait casté en la
  chaîne `"nan"` (≠ `""`), faisant manquer de vraies collisions de clé sur
  les lignes sans `variant`/`keep_frac`. `fillna("")` ajouté avant le cast
  (même classe de bug que l'incident de fusion P2, cf. `src/csv_upsert.py`).
- **Nommage des modèles harmonisé** : le JSON canonique stocke en interne
  `"GCN+Transformer"`/`"Linear+Transformer"` (avec `+`, jamais modifié — ce
  n'est pas une logique de calcul), mais `raw_results.csv` normalise en
  `"GCN-Transformer"`/`"Linear-Transformer"` (tiret) partout, pour matcher la
  convention déjà utilisée dans les tableaux/prose du projet (README,
  external_baselines_tables.md). Sans cette normalisation, les lignes issues
  de `06_train_multistation.py` et celles migrées depuis les autres sources
  auraient été traitées comme des modèles différents par toute requête/
  assertion sur `model`.
- **Linear-Transformer dupliqué par topologie** (Table 2) plutôt que
  dédupliqué à `topology=""` : bien que son calcul soit indépendant de la
  topologie (valeurs identiques vérifiées), la boucle live de
  `06_train_multistation.py` l'écrit naturellement une fois par topologie —
  la migration réplique ce comportement plutôt que de le dédupliquer, pour
  que les données migrées et les données live restent structurellement
  identiques.

**Migration (`scripts/migrate_raw_results.py`)** : 744 lignes migrées depuis
7 sources existantes, 300 avec `provenance_note` (147 SUSPECT_6STATION Madrid
pré-P1, 13 UNRECOVERABLE, 140 signalant une granularité réduite mais fiable —
pas de RMSE/MAE ou pas de per-station selon le script d'origine). Aucune
donnée dérivable d'une source déjà migrée n'a été dupliquée (Table 6, Table 4/5,
lignes k=5 de `sensitivity_k_canonical.csv` volontairement exclues de la
migration — recalculables à 100 % depuis le JSON canonique).

**Analyse de couverture (`scripts/gap_analysis.py`)** — voir aussi le rapport
de tâche P3 pour le détail complet : Tables 2, 3, 7, 9 intégralement
couvertes (0 condition manquante). **Table 8 (over-smoothing/GAT) : 36/36
conditions manquantes — aucune ligne, le script n'avait jamais rien persisté
avant ce correctif.** Édition d'arêtes (§6.2.1) : 20/45 manquantes —
Beijing/London n'ont que le seed 42 (Madrid a bien les 3 seeds, mais
SUSPECT_6STATION).

## Budget de calcul restant (mis à jour, à consigner)

En plus d'E9 (k-sensitivity Madrid k∈{3,8}, 12 cellules) et E10 (pruning
Madrid complet, 15 cellules) déjà identifiés en P1 :

- **Table 8 (over-smoothing/GAT) : 36 runs**, jamais lancés — 3 villes × 4
  variantes (linear1L, gcn1L, gcn2L, gat2L) × 3 seeds, topologie distance
  uniquement. Coût par run inconnu (jamais chronométré, script jamais
  exécuté jusqu'au bout avec sortie persistée).
- **Édition d'arêtes (§6.2.1) : 20 runs** — seeds 123/777 pour Beijing et
  London (5 niveaux × 2 seeds × 2 villes), en plus des 15 cellules Madrid
  d'E10 déjà comptées séparément.

**Total identifié à ce stade : 12 (E9) + 15 (E10 Madrid) + 20 (edge pruning
Beijing/London) + 36 (Table 8) = 83 runs GCN restants**, avant toute
expérience nouvelle (E11 parité seeds, E12 contrôles pruning, E13 modèle
post-2024).

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
