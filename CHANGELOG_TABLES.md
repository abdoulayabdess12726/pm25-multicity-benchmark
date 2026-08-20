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

## Budget de calcul restant (mis à jour P5, à consigner)

**E9 et E10 terminés** (P5, cf. section dédiée ci-dessous pour les résultats) :
12 cellules k-sensitivity Madrid k∈{3,8} + 15 cellules pruning Madrid complet
— 27 runs GCN, retirés du budget restant.

Restant, par ordre de dépendance :

- **Table 8 (over-smoothing/GAT) : 36 runs**, jamais lancés — 3 villes × 4
  variantes (linear1L, gcn1L, gcn2L, gat2L) × 3 seeds, topologie distance
  uniquement. Coût par run inconnu (jamais chronométré, script jamais
  exécuté jusqu'au bout avec sortie persistée).
- **Édition d'arêtes (§6.2.1), Beijing/London : 20 runs** — seeds 123/777
  pour Beijing et London (5 niveaux × 2 seeds × 2 villes). Madrid déjà
  complet (E10).
**Numérotation officielle des expériences : voir `REVISION_BRIEF.md`, section
« Numérotation des expériences (E-series) » — référence unique, toutes
sessions. Ne pas attribuer de numéro localement.**

- **E11 — parité seeds STGCN/Graph WaveNet : 12 runs.** `14_sota_baselines.py`
  n'a jamais tourné qu'au seed 42 (périmètre strict déclaré dans le script :
  « SOTA baselines reported at the primary seed given their computational
  cost » — cf. manuscrit). Extension à 123/777 : 2 modèles × 3 villes × 2
  seeds = 12 runs.
- **E12 — contrôles pruning : élagage aléatoire à densité appariée + élagage
  inverse** (garder les arêtes les plus hétérophiles au lieu des moins
  hétérophiles). Nombre de runs à chiffrer une fois le protocole détaillé.
- **E13 — Table 8 (over-smoothing/GAT) : 36 runs**, jamais lancés — 3 villes
  × 4 variantes (linear1L, gcn1L, gcn2L, gat2L) × 3 seeds, topologie distance
  uniquement. Coût par run inconnu (jamais chronométré, script jamais
  exécuté jusqu'au bout avec sortie persistée).
- **E14 — édition d'arêtes Beijing/London : 20 runs** — seeds 123/777 pour
  Beijing et London (5 niveaux × 2 seeds × 2 villes). Madrid déjà complet
  (E10).
- **E15 — AirPhyNet, baseline post-2024** — session parallèle (pas cette
  session). Étude de faisabilité déjà faite (`external/AIRPHYNET_FEASIBILITY.md`,
  commit `51fd7f1` — labellisé « E11 » à l'époque, correspond en réalité à
  E15, cf. note de correspondance dans `REVISION_BRIEF.md`) : reproduction à
  <1 % des chiffres publiés, coûts mesurés, 14 modifications listées. Runs
  non lancés (0/9 : 3 villes × 3 seeds). Le clone `external/AirPhyNet` fait
  par cette session (2026-08-16) était redondant avec ce travail déjà
  existant — sans impact (gitignoré), rien à corriger.
- **E16 — Chang-Zhu-Tan (CZT), 4e réseau de validation externe de h(D)** —
  session parallèle (pas cette session). Pré-enregistrement fait
  (`PREREGISTRATION_CZT.md`, commit `dadb848`) : h(D)=0.312 mesuré avant tout
  entraînement, prédictions P1/P2/P3 datées. Collecte de données démarrée
  (`561f42e`, 22 stations CNEMC 2020-2023, `01h_download_czt.py`) — non
  versionné ici, appartient à cette autre ligne de travail.

**Total chiffré à ce stade : 36 (E13) + 20 (E14) + 12 (E11) = 68 runs
GCN/SOTA restants**, hors E12 (scope à détailler) et E15/E16 (pilotées par
une autre session).

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
| Table 7 | Madrid, k∈{3,8} (2 topologies × 2 k × 3 seeds, 12 lignes) | 6 stations | 7 stations | Ré-entraînement complet (E9, P5). ΔR² 3-seeds : distance k=3 −0.3019→**−0.2828**, k=8 −0.3685→**−0.3478** ; correlation k=3 −0.3437→**−0.3145**, k=8 −0.4540→**−0.4144**. Dans les 4 cas, MENDEZ ALVARO atténue la dégradation (ΔR² moins négatif de 0.019 à 0.040) mais ne change pas la conclusion qualitative (Madrid reste ~15-25× plus dégradée que Beijing selon la condition) | ✅ confirmé (E9, P5) |
| §6.2.1 (pruning) | Madrid, toutes lignes (5 niveaux × 3 seeds, 15 conditions) | 6 stations | 7 stations | Ré-entraînement complet (E10, P5). Ancre pleine-graphe (keep_frac=1.0) : R² moyen 0.4716→**0.4877**, désormais cohérente avec Table 2 (écart <0.008, 3 seeds, tolérance 3×std passée). Courbe de récupération conservée : R² croît de 0.49 (100% arêtes) à 0.75 (0% arêtes, convergence vers Linear-Transformer) — forme et conclusion inchangées | ✅ confirmé (E10, P5) |

**MENDEZ ALVARO — ΔR² par station (Madrid, graphe complet, E10, 3 seeds)** : confirme
l'hypothèse P1 (la station dilue la dégradation Madrid). Classement des 7 stations
par ΔR² (du moins au plus dégradé) : MENDEZ ALVARO **−0.1636** (rang 1/7, la moins
dégradée) < PLAZA ELÍPTICA −0.2369 < CASA DE CAMPO −0.2453 < PLAZA CASTILLA-CANAL
−0.2461 < ESCUELAS AGUIRRE −0.2708 < CUATRO CAMINOS-PABLO −0.2750 < CASTELLANA
**−0.5770** (la plus dégradée). Moyenne 7-stations : −0.2878. MENDEZ ALVARO est
nettement au-dessus de la moyenne (moins négative), confirmant chiffres à l'appui
la prédiction P1.

**Points à consigner pour la lettre aux relecteurs (2026-08-16) :**

1. **MENDEZ ALVARO est la station la moins dégradée de tout le réseau Madrid**
   (−0.1636 vs moyenne −0.2878, rang 1/7). Cohérent avec le mécanisme : série
   d'entraînement constante sur cette station (PM2.5 invariant sur le train,
   cf. `results/mendez_alvaro_diagnostic.md`), donc pas de signal temporel
   propre que l'agrégation spatiale du GCN puisse corrompre. Conséquence
   directe : le chiffre publié à 7 stations est **le plus conservateur des
   deux protocoles possibles** — inclure MENDEZ ALVARO réduit la dégradation
   Madrid rapportée, on ne gonfle pas l'effet en la gardant. **Argument à
   faire figurer dans la réponse R1.5/R2.5.**
2. **Étendue intra-Madrid** : ΔR² va de **−0.164** (MENDEZ ALVARO) à **−0.577**
   (CASTELLANA) sur le même réseau, le même graphe, le même protocole — un
   facteur **3.5×** entre la station la moins et la plus touchée. À réutiliser
   dans l'analyse intra-ville du modèle mixte (**P9 / R2.1**) : l'hétérophilie
   locale par station (cf. Étape 3 / `12_per_station_heterophily.py`) explique
   une partie de cette étendue, pas seulement le h(D) global de la ville.

## E11 — parité seeds STGCN/Graph WaveNet (P5, validé 2026-08-16)

12 runs (2 modèles × 3 villes × seeds 123/777, en plus du seed 42 déjà
présent depuis E7). Table 3 : STGCN et Graph WaveNet passent de
`primary_seed` (une seule ligne, seed 42) à `3seed_mean` (moyenne±écart-type
ddof=1 sur 42/123/777), au même standard que les autres modèles stochastiques
de la table (`scripts/regenerate_tables.py::table3()`, bloc STGCN/GWN
réécrit pour utiliser `agg_mean_std` au lieu de `sub.r2.iloc[0]`).

**Provenance du « au plus 0.011 » (résumé, §5.2, §6.1) — chiffre affiché
inchangé, provenance à corriger.** L'avantage maximal d'un modèle graphe sur
le Linear-Transformer, recalculé depuis les valeurs non arrondies sur
moyennes 3 seeds (tous modèles graphe confondus — GCN-Transformer 2
topologies, STGCN, Graph WaveNet — dans les 3 villes) : **0.010593**
(Beijing / Graph WaveNet), contre 0.011351 au seed 42 seul (même
ville/modèle). Les deux arrondissent à **0.011** — **le chiffre affiché ne
change pas** — mais sa provenance change : ce n'est plus une valeur seed-42,
c'est désormais une moyenne 3 seeds. **À corriger partout où « 0.011 »
apparaît (résumé, §5.2, §6.1)** : remplacer la mention « seed 42 »/valeur
seed-unique par « moyenne 3 seeds (42/123/777) » en note de provenance,
sans toucher au chiffre 0.011 lui-même.

**Nuance à traiter en §6.1, pas encore rédigée (attend E12 + E15) : Madrid /
Graph WaveNet = +0.005622, positif.** Sur le réseau où le GCN-Transformer
s'effondre le plus (ΔR² −0.32 à −0.40 selon la topologie), une architecture
à adjacence *apprise* (Graph WaveNet) fait légèrement mieux que le
Linear-Transformer — seul cas positif hors Beijing parmi les 12 combinaisons
ville×modèle-graphe testées (cf. tableau de classement ci-dessous). C'est
une nuance réelle du résultat négatif du papier : à documenter explicitement
en §6.1 plutôt que de la laisser dans le seul tableau, pour qu'un relecteur
ne la découvre pas de lui-même. **Ne pas rédiger §6.1 avant que E12
(contrôles pruning) et E15 (AirPhyNet) aient donné leurs résultats** — la
rédaction attend une vue d'ensemble, pas cette seule cellule.

Classement complet, avantage vs Linear-Transformer (moyenne 3 seeds, non
arrondi, tous modèles graphe × toutes villes) :

| Rang | Ville / modèle | Avantage vs Linear |
|---|---|---|
| 1 | Beijing / Graph WaveNet | **+0.010593** |
| 2 | Beijing / STGCN | +0.008769 |
| 3 | **Madrid / Graph WaveNet** | **+0.005622** |
| 4 | London / Graph WaveNet | +0.000239 |
| 5 | London / STGCN | −0.000114 |
| 6 | Beijing / GCN-Transformer (distance) | −0.017178 |
| 7 | Madrid / STGCN | −0.019684 |
| 8 | Beijing / GCN-Transformer (correlation) | −0.037517 |
| 9 | Madrid / GCN-Transformer (distance) | −0.321297 |
| 10 | London / GCN-Transformer (distance) | −0.375412 |
| 11 | Madrid / GCN-Transformer (correlation) | −0.379538 |
| 12 | London / GCN-Transformer (correlation) | −0.401444 |

## E13 — Table 8, over-smoothing/GAT (P5, 2026-08-17/18) — CONTREDIT LE MANUSCRIT POUR BEIJING

36 runs (3 villes × 4 variantes × 3 seeds, topologie distance uniquement,
cf. Table 8 caption du manuscrit). **Cette table n'avait jamais eu de ligne
persistée avant ce run** — c'est la première exécution reproductible.
Variantes vérifiées contre le manuscrit (§6.2, paragraphe décrivant le
contrôle : « single-layer GCN, the two-layer GCN, and a GAT variant ») avant
lancement : linear1L/gcn1L/gcn2L/gat2L, correspondance confirmée.

| Ville | Linear (1L) | GCN (1L) | GCN (2L) | GAT (2L) | ΔR² gcn1L vs Linear | Wilcoxon p (gcn1L<Linear) |
|---|---|---|---|---|---|---|
| Beijing | 0.8790±0.0020 | 0.8852±0.0082 | 0.8825±0.0031 | 0.8619±0.0027 | **+0.006** | **0.9939** |
| London | 0.6938±0.0202 | 0.2523±0.0122 | 0.1366±0.0077 | −0.0705±0.0220 | −0.442 | 0.0039 |
| Madrid | 0.6460±0.0050 | 0.4968±0.0086 | 0.4588±0.0037 | 0.3090±0.0045 | −0.149 | 0.0078 |

**Comparaison à l'affirmation du manuscrit** (§6.2 : *« A single-layer GCN
[...] already underperforms the baseline at every city (per-station
Wilcoxon: London ΔR2 = −0.279, p = 0.004; Madrid −0.283, p = 0.008; Beijing
−0.008, p = 0.005) »*) :

- **Beijing — CONTREDIT, pas une simple divergence de magnitude.** Le
  manuscrit affirme ΔR²=−0.008, p=0.005 (significatif, gcn1L pire que
  Linear). Ce run donne ΔR²=**+0.006** (signe opposé), p=**0.9939** (non
  significatif — au sens strict, gcn1L n'est même pas *pire* que Linear ici,
  la comparaison va dans l'autre sens). Cohérent avec h(D) Beijing=0.497
  (le réseau le moins hétérophile, peu d'arêtes hétérophiles à corrompre le
  signal) mais **contredit le chiffre précis déjà écrit dans le texte**
  utilisé pour l'argument central du §6.2 (« underperforms at every city »).
- **London — même sens (significatif, gcn1L pire), magnitude très
  différente.** Manuscrit ΔR²=−0.279 ; ce run ΔR²=−0.442 (58 % plus négatif).
  p quasi identique (0.0039 vs 0.004 manuscrit).
- **Madrid — même sens (significatif, gcn1L pire), magnitude très
  différente.** Manuscrit ΔR²=−0.283 ; ce run ΔR²=−0.149 (quasi moitié moins
  négatif). p quasi identique (0.0078 vs 0.008 manuscrit).

**Point prioritaire (2026-08-18) : les MAGNITUDES divergent pour Londres/Madrid,
pas seulement Beijing — un décalage de protocole donnerait un biais de même
sens partout, ce n'est pas le cas ici (Londres plus négatif que le
manuscrit, Madrid moins négatif).** Investigation menée, sans relancer aucun
entraînement (consigne explicite) :

1. **Recherche de −0.279/−0.283/−0.008 ailleurs dans le dépôt (autre ville,
   variante gcn2L, autre topologie).** Grep textuel sur tout le dépôt
   (hors `data/czt_raw`, `external`, `MSDGNN...`) : aucune occurrence sauf
   dans ce CHANGELOG. `git log --all -S` (pickaxe) sur toute l'histoire :
   aucun commit n'a jamais introduit ces valeurs, et aucun commit n'a jamais
   touché `09_controls_oversmoothing.py` avant E13 — cohérent avec « cette
   table n'a jamais eu de ligne persistée ». Recalcul exhaustif de tous les
   ΔR² possibles (gcn1L/gcn2L/gat2L) × (Linear T8 propre / Linear T2
   canonique) pour les 3 villes : **aucune combinaison ne reproduit les 3
   valeurs du manuscrit simultanément** — seul un rapprochement isolé et
   non systématique apparaît pour Beijing/gat2L vs Linear-T8 (−0.017,
   proche de −0.008 mais pas exact, et ne concerne qu'une ville). Un
   rapprochement numérique notable a été trouvé ailleurs : **Madrid
   Table 7 (k-sensitivity) k=3/distance = −0.2828**, qui arrondit à
   **−0.283** — coïncidence troublante avec la cible Madrid, mais dans une
   table entièrement différente (sensibilité k, pas over-smoothing), et
   surtout **cette valeur Table 7 vient d'être corrigée par E9 dans cette
   même session P5** (ancienne valeur 6-station : −0.3019, pas −0.283) — le
   manuscrit, rédigé avant E9, ne pouvait pas avoir cité ce chiffre. Très
   probablement une coïncidence numérique, pas la source réelle.
2. **Référence Linear vérifiée — confirmé, deux références distinctes
   existent et sont substantiellement différentes.** Table 8 utilise sa
   propre référence (`Linear-Transformer`, `variant=1layer`, une simple
   couche `nn.Linear` + backbone temporel partagé) : R²=0.879/0.694/0.646
   (Beijing/London/Madrid). Table 2 utilise le Linear-Transformer canonique
   (architecture complète) : R²=0.949/0.842/0.814 — un modèle nettement
   plus fort. Les deux références ont été testées comme dénominateur
   possible pour les 3 variantes GCN (point 1 ci-dessus) : aucune ne
   reproduit les chiffres du manuscrit, donc une confusion de référence
   n'explique pas à elle seule l'écart, même si c'est une source d'erreur
   réelle et maintenant documentée pour toute relecture future du script.
3. **Aucune trace de l'exécution console d'origine.** Pas de log, notebook,
   ou historique shell retrouvé dans ce dépôt. Cohérent avec le docstring du
   script lui-même (« aggregate the printed mean +/- std into Table VI/VI.B »
   — la transcription manuelle console→manuscrit était le flux prévu dès
   l'origine, jamais un export automatisé).

**Conclusion de l'investigation (close, 2026-08-18) : source de
−0.279/−0.283/−0.008 non identifiée — trois pistes épuisées.** Ceci est une
conclusion, pas un échec d'investigation : (1) recherche textuelle +
`git log --all -S` sur tout le dépôt et son historique — rien ; (2) toutes
les combinaisons (variante × référence Linear T2/T8, 3 villes) testées —
aucune ne reproduit les 3 valeurs simultanément ; (3) aucune trace de
log/notebook/historique shell de l'exécution console d'origine. Le script
`09_controls_oversmoothing.py` n'a jamais été modifié dans cette révision
hors l'ajout du flag `--cpu` (P5). **§6.2 ne doit pas être mis à jour tant
que cette source n'est pas identifiée — aucune investigation
supplémentaire prévue à ce stade.**

## Point de rédaction §6.3 — h(D) prédit le signe/l'ampleur, pas un classement strict (préexistant, pas un artefact E13)

**Le renversement Madrid/London existe DÉJÀ dans la Table 2 du manuscrit
soumis** — ce n'est pas un artefact du contrôle over-smoothing simplifié
d'E13, découvert en creusant E13 mais présent indépendamment. Valeurs h(D)
(Table 1, `manuscript/tables/table1_h_index.md`) : **Beijing 0.497, London
0.656, Madrid 0.728** — Madrid est le réseau le plus hétérophile des trois,
pas Londres.

Or, dans la Table 2 du manuscrit (GCN-Transformer canonique, ΔR² vs
Linear-Transformer, 3 seeds) : **London dégrade davantage que Madrid dans
les deux topologies** (distance −0.375 vs −0.321 ; correlation −0.401 vs
−0.380), malgré un h(D) plus faible (0.656 < 0.728). Le même renversement
réapparaît dans le contrôle over-smoothing d'E13 (gcn1L : London −0.442 vs
Madrid −0.149) — cohérence entre les deux modèles, qui exclut un artefact
de l'architecture simplifiée du contrôle.

**Portée exacte, à ne pas surstate** : sur 3 réseaux, h(D) sépare
correctement Beijing (le plus homophile, la dégradation la plus faible,
voire nulle/positive) du groupe {London, Madrid} (tous deux nettement
dégradés) — le signe et l'ordre de grandeur de l'effet sont bien prédits.
Ce que h(D) ne produit PAS, c'est un classement strict entre London et
Madrid : leurs h(D) sont proches (0.656 vs 0.728) et leur ΔR² observé va
dans le sens inverse de leur écart de h(D).

**À énoncer nous-mêmes en §6.3 comme limite de l'indice, plutôt que de
laisser un relecteur la trouver** — cohérent avec le repli déjà prévu sur
« indicateur diagnostique préliminaire » (R2.1) plutôt qu'un prédicteur
quantitatif strict. **Rédaction non faite maintenant** (consigne explicite,
2026-08-18) — ce point attend la même vue d'ensemble qu'E11's nuance
Madrid/GraphWaveNet (E12 + E15), pas de modification du manuscrit à ce
stade.

**Correctif de migration associé (P5)** : `migrate_k_sensitivity()` et
`migrate_edge_pruning()` (`scripts/migrate_raw_results.py`) marquaient
inconditionnellement Madrid k∈{3,8}/toutes les lignes de pruning comme
UNRECOVERABLE_6STATION/SUSPECT_6STATION (n_stations=6) — correct avant E9/E10,
obsolète après (les fichiers sources `results/e6_k_sensitivity.csv` et
`results/edge_pruning.csv` contiennent maintenant les vraies valeurs 7-station).
117 lignes placeholder résiduelles dans `raw_results.csv` (12 k-sensitivity +
105 pruning), en collision de run_id avec les nouvelles lignes réelles de même
identité logique, retirées via `scripts/remove_stale_madrid_placeholders.py`
(947→830 lignes). Assertions bloquantes : 17 PASS/4 FAIL/4 MISSING DATA →
**21 PASS/0 FAIL/4 MISSING DATA** (les 4 MISSING DATA restants concernent
Beijing/London seeds 123/777 pour l'ancre de pruning, hors périmètre P5).

**Non modifié** : Beijing et London (aucune exclusion n'y a jamais été
appliquée, tous les scripts chargeaient déjà la totalité des stations pour
ces deux villes).

## E14 — édition d'arêtes Beijing/London, seeds 123/777 (P5, 2026-08-19)

20 runs (2 villes × 5 niveaux × seeds 123/777, script `13_edge_pruning.py`
inchangé). Complète les 4 MISSING DATA de l'ancre de pruning (Beijing/London
seeds 123/777) et met la courbe de pruning à parité de 3 seeds avec Madrid
(déjà acquis depuis E10). **Assertions bloquantes : 25 PASS / 0 FAIL / 0
MISSING DATA (25/25) — complet pour la première fois ce cycle.**

## E12 — contrôles de pruning : aléatoire à densité appariée + inverse (P5, 2026-08-20)

Nouveau script `15_pruning_controls.py` (réutilise `get_city`/`prune`/
`train_eval`/`metrics_rows` de `13_edge_pruning.py` par import, aucune
logique de calcul existante modifiée). Deux stratégies de contrôle,
appariées en densité (même nombre d'arêtes conservées) avec le pruning
guidé : (1) **aléatoire** — sous-ensemble uniforme sans remise, 5 tirages
indépendants (seeds 1001-1005, hors bande des seeds canoniques 42/123/777,
un seed contrôlant à la fois la sélection d'arêtes et l'entraînement du
tirage) ; (2) **inverse** — retire les arêtes les plus homophiles
(corrélées) d'abord, exact opposé du guidé, protocole standard 3 seeds.

**Réduction de budget assumée** (consigne explicite : réduire les niveaux,
pas les tirages) : niveaux intermédiaires réduits à {0.75, 0.25} au lieu
des 5 canoniques — les niveaux 1.0/0.0 sont partagés avec le guidé
(identiques par construction à ces extrêmes, aucune donnée dupliquée). 48
entraînements au total (3 villes × 2 niveaux × (5 tirages + 3 seeds)),
3172 min (~52,9h). Figure : `figures/pruning_controls.png`
(`16_pruning_controls_figure.py`), 3 courbes par ville + Linear-Transformer
en pointillés.

### Ce qui est tenu constant (degré effectif) vs ce qui diffère (magnitude des messages)

Degré effectif identique par construction entre les 3 stratégies à un
niveau donné (densité appariée) ; magnitude moyenne des messages (= moyenne
des edge_weight conservés, le coefficient de pondération GCN) diffère
nettement et dans le sens attendu (guidé > aléatoire > inverse, puisque
guidé garde les arêtes les plus fortement corrélées) :

| Ville | Niveau | Degré effectif (identique) | Magnitude guidé | Magnitude aléatoire (moy. 5 tirages) | Magnitude inverse |
|---|---|---|---|---|---|
| Beijing | 75% | 3.75 | 0.4306 | 0.3311 | 0.2316 |
| Beijing | 25% | 1.25 | 0.6289 | 0.3431 | 0.0318 |
| London | 75% | 3.75 | 0.3050 | 0.3097 | 0.3365 |
| London | 25% | 1.25 | 0.2230 | 0.2694 | 0.3176 |
| Madrid | 75% | 3.71 | 0.2780 | 0.2598 | 0.2289 |
| Madrid | 25% | 1.29 | 0.3683 | 0.2513 | 0.2119 |

(London/Madrid : la magnitude guidé/aléatoire/inverse est plus resserrée
qu'à Beijing — cohérent avec un réseau globalement plus hétérophile, où
même les arêtes "les plus corrélées" du guidé ne sont pas très corrélées
en absolu.)

### R² par ville, niveau, stratégie (moyenne, ddof=1)

| Ville | Niveau | Guidé | Aléatoire (5 tirages) | Inverse (3 seeds) |
|---|---|---|---|---|
| Beijing | 75% | 0.9316±0.0015 | 0.9323±0.0021 | 0.9325±0.0018 |
| Beijing | 25% | 0.9442±0.0003 | 0.9366±0.0033 | 0.9445±0.0008 |
| London | 75% | **0.5859**±0.0161 | 0.4766±0.0349 | 0.4833±0.0090 |
| London | 25% | **0.7357**±0.0040 | 0.6439±0.0580 | 0.5829±0.0074 |
| Madrid | 75% | 0.5051±0.0086 | 0.4937±0.0393 | **0.5827**±0.0216 |
| Madrid | 25% | 0.5896±0.0078 | 0.6088±0.0888 | **0.6409**±0.0614 |

### Résultat par rapport à la prédiction (guidé > aléatoire > inverse) — rapporté tel quel, pas de collapse généralisé mais Madrid contredit franchement

- **Beijing** : les 3 stratégies indiscernables (réseau trop homophile —
  h(D)=0.497 — pour que l'ordre des arêtes retirées fasse une différence
  mesurable). Résultat neutre, ni confirme ni infirme.
- **London (h(D)=0.656)** : **confirme la prédiction** — guidé > aléatoire >
  inverse aux deux niveaux, écarts nets (guidé bat aléatoire de +0.109 à
  75%, +0.092 à 25% ; aléatoire bat inverse de +0.061 à 25%, quasi égal à
  75%). Le mécanisme (récupération vient de la suppression des arêtes
  hétérophiles, pas de la réduction de densité) tient clairement pour cette
  ville.
- **Madrid (h(D)=0.728, le réseau le plus hétérophile) : CONTREDIT
  FRANCHEMENT LA PRÉDICTION.** Inverse (0.5827/0.6409) bat guidé
  (0.5051/0.5896) aux deux niveaux — l'exact opposé de l'ordre prédit.

**Investigation menée avant de rapporter (pas un cadrage qui sauve la
conclusion — vérification que ce n'est pas un artefact avant de conclure) :**
suspicion initiale que MENDEZ ALVARO (dont les corrélations train sont
quasi nulles avec tout le monde, donc classée arête "maximalement
hétérophile" par construction plutôt que par une vraie relation spatiale)
soit isolée sous guidé (degré=0 vérifié dès 75%, contre degré=5 conservé
sous inverse) et fasse basculer l'agrégat. **Écartée** : en excluant
MENDEZ ALVARO, l'écart persiste et **s'aggrandit** sur les 6 autres
stations (guidé 0.459 vs inverse 0.563 à 75% ; 0.556 vs 0.637 à 25%).
MENDEZ ALVARO est en fait la SEULE station de Madrid qui va dans le sens
prédit (guidé meilleur pour elle : R²≈0.72-0.75 isolée vs 0.34-0.64 sous
inverse, cohérent avec le fait qu'elle n'a jamais eu besoin du graphe,
cf. P1) — elle atténue le renversement au niveau agrégat, elle ne
l'explique pas. La cause du renversement Madrid reste ouverte (7 stations
seulement, `n=3` seeds par condition — échantillon petit ; graphe très
clairsemé aux niveaux testés, 9 arêtes sur 7 nœuds à 25% — la structure
fine du sous-graphe pourrait dominer davantage que l'hétérophilie moyenne
à cette taille).

**Conséquence pour §6.2.1** : l'argument causal actuel (« pruning guidé
par hétérophilie récupère la performance, donc l'hétérophilie est bien la
cause ») **ne peut plus être énoncé comme universel sur les 3 villes**.
Il tient pour Londres, est neutre pour Beijing, et est contredit pour
Madrid. §6.2.1 nécessite une révision — **pas une réécriture complète**
(le mécanisme n'est pas invalidé partout, contrairement au scénario du
collapse généralisé qui aurait fait s'effondrer aléatoire au niveau de
guidé dans les 3 villes, ce qui n'est PAS ce qu'on observe), mais la
formulation ne peut plus prétendre à une causalité uniforme. **Rédaction
non faite maintenant** — attend une décision sur la formulation
(nuancer par ville ? approfondir la cause Madrid avant §6.2.1 ? cf.
discussion à avoir).

### Investigation Madrid (2026-08-20) — deux anomalies indépendantes sur le même réseau, source non résolue

Madrid renverse déjà l'ordre h(D) en Table 2 (§6.3, point de rédaction
ci-dessus) et renverse maintenant aussi l'ordre guidé/inverse en E12 — deux
anomalies distinctes sur le même réseau. Quatre points vérifiés, **aucun
entraînement relancé, aucune configuration cherchée pour redresser Madrid** :

**1. Niveaux testés et dispersion inter-seeds.** Seuls 2 niveaux
intermédiaires réentraînés pour Madrid (0.75 et 0.25, réduction de budget
E12 déjà documentée) :

| Niveau | Guidé (3 seeds) | Inverse (3 seeds) | Écart | SE combiné | Écart/SE |
|---|---|---|---|---|---|
| 0.75 | 0.5031, 0.4976, 0.5145 → 0.5051±0.0086 | 0.6073, 0.5666, 0.5742 → 0.5827±0.0216 | +0.0776 | 0.0233 | **3.34** |
| 0.25 | 0.5926, 0.5954, 0.5808 → 0.5896±0.0078 | 0.6964, 0.6514, 0.5750 → 0.6409±0.0614 | +0.0513 | 0.0619 | **0.83** |

À 75 % l'écart est net, ~3,3 fois l'incertitude combinée — pas un
chevauchement de dispersion. À 25 % l'écart n'est **pas** distinguable de
la dispersion inter-seeds (0,83×SE), portée presque entièrement par la
variance d'`inverse` (std=0,0614, 8× celle de guidé à ce niveau) —
un seed (777, R²=0,575) tire la moyenne inverse vers le bas. **Avec 3
seeds seulement, le point à 25 % ne permet pas de conclure seul ; celui à
75 % si.**

**2. MENDEZ ALVARO dans l'ordre d'élagage.** Ses 11 arêtes (sur 35 au
total, 31 %) ont une corrélation train NaN (variance nulle, cf. P1) — dans
le calcul d'hétérophilie (`het = 1 − corr`, NaN traité comme hétérophilie
infinie), **ses 11 arêtes occupent EXACTEMENT les rangs 0-10 (les 11
premières retirées) sous guidé, et EXACTEMENT les rangs 24-34 (les 11
dernières retirées) sous inverse** — sur un total de 35 arêtes. Elle
bascule donc à l'extrême le plus tôt possible dans un ordre et à l'extrême
le plus tard possible dans l'autre, par construction : ce n'est pas une
hétérophilie mesurée qui la place là, c'est une valeur dégénérée (NaN) qui
la place automatiquement à la borne, quel que soit le sens du tri.
Conséquence structurelle : sous guidé elle est isolée (degré=0) dès
keep=75 % ; sous inverse elle garde son degré plein (5) même à keep=25 %,
et devient un hub disproportionné dans le sous-graphe conservé (5 de ses
arêtes sur les 9 arêtes totales conservées à ce niveau, cf. section E12
ci-dessus). **Déjà vérifié (section E12) que l'exclure de l'agrégat ne
supprime pas le renversement** — les 6 autres stations le montrent aussi,
en plus fort. Ceci n'explique donc pas le renversement du R² agrégé, mais
explique pourquoi l'ORDRE d'élagage lui-même est structurellement dominé
par un artefact de corrélation dégénérée plutôt que par une hétérophilie
réellement mesurée, sur près d'un tiers des arêtes.

**3. Magnitude des messages, guidé vs inverse — écart réel, pas un défaut
de discrimination.**

| Ville | Niveau | Magnitude guidé | Magnitude inverse | Écart relatif |
|---|---|---|---|---|
| Beijing | 75% | 0,4306 | 0,2316 | +46 % |
| Beijing | 25% | 0,6289 | 0,0318 | +95 % |
| London | 75% | 0,3050 | 0,3365 | −10 % |
| London | 25% | 0,2230 | 0,3176 | −42 % |
| Madrid | 75% | 0,2780 | 0,2289 | +18 % |
| Madrid | 25% | 0,3683 | 0,2119 | +42 % |

Madrid n'a PAS un écart quasi nul : +18 % à 75 %, +42 % à 25 % — du même
ordre de grandeur que Beijing à 75 % (+46 %) et plus net que London (qui
est même inversé, guidé < inverse en magnitude aux deux niveaux — cohérence
supplémentaire avec le fait que Londres suit la prédiction). **Écarté** :
l'explication « guidé et inverse retirent des arêtes de magnitude quasi
identique à Madrid, donc les deux conditions ne sont pas distinctes » ne
tient pas — les conditions SONT distinctes en magnitude, le renversement
n'est pas un artefact d'absence de contraste.

**4. Distribution des corrélations d'arêtes, base graphe (avant élagage).**

| Ville | n arêtes | n NaN (Mendez) | corr finie : min/méd./max | std | CV (std/moy) | Forme |
|---|---|---|---|---|---|---|
| Beijing | 60 | 0 | 0,814 / 0,931 / 0,963 | 0,043 | 0,047 | unimodale, resserrée en haut |
| London | 40 | 0 | 0,259 / 0,576 / 0,766 | 0,112 | 0,194 | unimodale, large étalement |
| Madrid | 35 | 11 (31 %) | 0,332 / 0,470 / 0,620 | 0,062 | 0,130 | unimodale, **resserrée** |

**Ni bimodale ni plate** : les 24 arêtes à corrélation finie de Madrid
forment une distribution unimodale, resserrée (tout l'intervalle tient
entre 0,33 et 0,62 — pas d'arête franchement anti-corrélée ni quasi-nulle
parmi les arêtes non-dégénérées). Combiné au point 2 : **31 % des arêtes
de Madrid (les 11 de MENDEZ ALVARO) sont placées aux extrêmes de l'ordre
par un artefact NaN, et les 24 arêtes restantes ne se séparent pas
nettement en deux groupes homophile/hétérophile** — elles occupent un
continuum étroit. L'ordre guidé/inverse à Madrid est donc dominé par
l'artefact MENDEZ ALVARO sur près d'un tiers du graphe, et par un
classement à faible contraste sur le reste.

**Bilan, sans redresser Madrid** : le point à 75 % reste une contradiction
statistiquement nette (3,3×SE) de la prédiction, la magnitude des messages
confirme que les conditions sont réellement distinctes (point 3), et la
piste MENDEZ ALVARO reste écartée comme explication du renversement du R²
agrégé (point 2, déjà vérifié en E12). Ce qui EST nouveau ici : la
distribution des corrélations (point 4) montre que l'ordre d'élagage
lui-même repose à Madrid sur un signal plus faible et plus artefactuel
(NaN sur 31 % des arêtes, continuum étroit sur le reste) qu'à Londres —
une raison plausible pour laquelle le classement des arêtes y est moins
fiable, sans que ceci explique le SENS du renversement observé. **Source
du renversement Madrid toujours non identifiée.** Aucune modification du
manuscrit.

### Traitement des corrélations indéfinies (2026-08-20) — investigation, aucun re-run

**1. Confirmé — à 75% à Madrid, guidé et inverse ne retirent pas la même
catégorie d'arêtes du tout.** 35 arêtes de base, keep=75% → 9 retirées :

- **Guidé** : les 9 arêtes retirées sont **9/9 NaN** (MENDEZ ALVARO,
  0 arête finie touchée) — les 9 retraits sont exactement ses 9 arêtes
  restantes à ce niveau (elle en a 11 au total sur 35 ; à 75% guidé a déjà
  retiré 9 des 11).
- **Inverse** : les 9 arêtes retirées sont **9/9 finies** (0 NaN touchée) —
  les plus corrélées réellement mesurées (corr 0,50 à 0,62 : CASTELLANA↔
  PLAZA ELÍPTICA, CASTELLANA↔CUATRO CAMINOS-PABLO, CASTELLANA↔CASA DE CAMPO,
  CASA DE CAMPO↔PLAZA ELÍPTICA, CASTELLANA↔PLAZA CASTILLA-CANAL).

**À ce niveau, « guidé » ne teste quasiment aucune décision d'hétérophilie
réelle — il retire presque exclusivement un artefact de corrélation
dégénérée**, tandis qu'« inverse » retire un signal homophile réellement
mesuré. Le contraste observé à 75% (guidé 0,5051 vs inverse 0,5827, section
E12) compare donc « graphe sans les arêtes dégénérées de MENDEZ ALVARO »
à « graphe sans ses 9 arêtes les plus corrélées réelles » — pas
« hétérophile retiré » vs « homophile retiré » au sens propre voulu par le
contrôle.

**2. Convention « NaN → hétérophilie infinie »** : **explicite et
documentée**, pas implicite — `13_edge_pruning.py`, docstring du module
(lignes 12-15) : *« Les arêtes vers une station à corr NaN [...] sont
traitées comme les plus hétérophiles (élaguées en premier) »* ; implémentée
ligne 101, `key = np.where(np.isnan(het), np.inf, het)`, commentée en
ligne (`# NaN -> plus hétérophile`). `15_pruning_controls.py::prune_inverse`
réutilise exactement la même ligne (copiée, pas importée) mais sans
réexpliquer la convention dans son propre docstring — corrigé au passage
dans ce même correctif de docstring (2026-08-18, cf. section E12).

**3. Recherche exhaustive de propagation — un point non documenté trouvé,
qui affecte la Table 2 canonique, pas seulement le contrôle de pruning :**

| Fichier | Usage de la corrélation | Gestion du NaN | Statut |
|---|---|---|---|
| `13_edge_pruning.py` / `15_pruning_controls.py` | ordre d'élagage (train seul) | NaN→∞, documenté | ✅ intentionnel |
| `06_train_multistation.py::build_correlation_graph()` | **graphe topologie « correlation », Table 2 canonique** (train seul, `data[:train_len]`, appel réel ligne 912) | **aucune** — `np.argsort(corr[i])[::-1]` place NaN en TÊTE (vérifié empiriquement), puis le seuil `corr>0` élimine l'arête NaN mais **consomme un slot de voisin sans le remplacer** | ⚠️ **NON documenté, affecte Table 2** |
| `12_per_station_heterophily.py` | h_i par station (Étape 3) | **masque explicitement** NaN/non-finis avant top-k (`valid = np.isfinite(row)`), h_i=NaN si aucun voisin valide | ✅ correct, documenté en commentaire (lignes 84-87) |
| `09_controls_oversmoothing.py::build_edges()` | topologie correlation (E13, **non exercée** — E13 lancé en `--topology distance` uniquement) | `argsort(-sim[i])` (sans `[::-1]`) — NaN reste en QUEUE, exclu du top-k (vérifié empiriquement sur cas jouet) | ✅ sûr par construction, mais jamais testé en pratique |
| `05_compute_heterogeneity_v2.py` (Table 1, h(D)) | r̄ (corrélation inter-stations moyenne) | `np.nanmean(corr[iu])` — ignore le NaN silencieusement ; Moran's I et CV ne dépendent pas de la matrice de corrélation (moyennes/écarts-types bruts par station) | ✅ sûr, pas de filtre explicite en amont mais résultat correct |

**Finding le plus important : `build_correlation_graph()` (Table 2,
topologie correlation, Madrid) construit en réalité un graphe où MENDEZ
ALVARO est TOTALEMENT ISOLÉE (0 arête entrante, 0 sortante — vérifié
empiriquement avec l'appel réel `data[:train_len]`), et où chacune des 6
autres stations perd 1 de ses 5 voisins k-NN prévus (4 arêtes réelles au
lieu de 5) parce qu'un NaN a occupé son 5e slot avant d'être filtré sans
remplacement.** Total : 24 arêtes au lieu des 35 attendues pour un k-NN à
k=5 sur 7 nœuds. **Ceci n'est PAS propre au contrôle de pruning — c'est
dans le graphe canonique utilisé pour la Table 2, topologie correlation,
Madrid, GCN-Transformer.** Pas encore établi si ceci modifie les chiffres
déjà publiés dans Table 2 (topologie « correlation », Madrid) au-delà de
« MENDEZ ALVARO isolée s'y comporte essentiellement comme un nœud sans
agrégation spatiale » — **non creusé plus loin, hors périmètre de cette
tâche (aucune réécriture demandée)**. À traiter comme item séparé.

**4. Coût d'un re-run avec les 11 arêtes NaN exclues structurellement du
graphe (élagage restreint aux 24 arêtes finies) — chiffré, non lancé.**
Interprétation retenue : reconstruire le graphe de base de pruning Madrid
en retirant d'emblée les 11 arêtes de MENDEZ ALVARO (elle devient isolée à
TOUS les niveaux, y compris 100%) — les 24 arêtes restantes constituent le
seul univers soumis à l'élagage. **Conséquence méthodologique à anticiper** :
l'ancre keep=100% de cette variante ne vaut plus 0,4877 (Table 2, graphe à
35 arêtes) — c'est un graphe différent, donc une nouvelle ancre propre à
calculer, l'assertion « ancre pruning == Table 2 » ne s'appliquerait pas
telle quelle à cette variante.

Budget (aucune donnée réutilisable de E10/E12, le graphe de base change) :
guidé 5 niveaux×3 seeds=15 + aléatoire 2 niveaux×5 tirages=10 + inverse
2 niveaux×3 seeds=6 → **31 runs**. Durée par run observée sur Madrid :
E10 (guidé, avec contention d'autres jobs) ≈57 min/run en moyenne ;
E12 (aléatoire/inverse, sans contention) ≈25-43 min/run. **Estimation :
31 runs × 35-70 min/run ≈ 18 à 36h**, selon contention avec d'autres runs
en parallèle sur la machine.

## Matériau pour §6.2.1 et la lettre R2.6 (2026-08-20) — rassemblé, rien rédigé

**Courbes 3 stratégies par ville, avec écart-types et seuil de
significativité (écart vs dispersion inter-seeds combinée)** — figure :
`figures/pruning_controls.png` ; données complètes :

| Ville | Niveau | Guidé | Aléatoire (5 tirages) | Inverse (3 seeds) | Écart guidé/inverse ÷ SE combiné |
|---|---|---|---|---|---|
| Beijing | 75% | 0,9316±0,0015 | 0,9323±0,0021 | 0,9325±0,0018 | 0,26 (indiscernable) |
| Beijing | 25% | 0,9442±0,0003 | 0,9366±0,0033 | 0,9445±0,0008 | 0,33 (indiscernable) |
| London | 75% | 0,5859±0,0161 | 0,4766±0,0349 | 0,4833±0,0090 | 5,45 (net) |
| London | 25% | 0,7357±0,0040 | 0,6439±0,0580 | 0,5829±0,0074 | 18,3 (net) |
| **Madrid** | **75%** | 0,5051±0,0086 | 0,4937±0,0393 | 0,5827±0,0216 | **3,34 (net, dépasse la dispersion)** |
| **Madrid** | **25%** | 0,5896±0,0078 | 0,6088±0,0888 | 0,6409±0,0614 | **0,83 (NE dépasse PAS la dispersion)** |

**Magnitude des messages par condition** (moyenne des edge_weight
conservés) :

| Ville | Niveau | Guidé | Aléatoire (moy. 5 tirages) | Inverse | Écart relatif guidé→inverse |
|---|---|---|---|---|---|
| Beijing | 75% | 0,4306 | 0,3311 | 0,2316 | +46% |
| Beijing | 25% | 0,6289 | 0,3431 | 0,0318 | +95% |
| London | 75% | 0,3050 | 0,3097 | 0,3365 | −10% |
| London | 25% | 0,2230 | 0,2694 | 0,3176 | −42% |
| Madrid | 75% | 0,2780 | 0,2598 | 0,2289 | +18% |
| Madrid | 25% | 0,3683 | 0,2513 | 0,2119 | +42% |

**Distribution des corrélations d'arêtes (base graphe, avant élagage), les
3 villes :**

| Ville | n arêtes | n NaN | corr finie : min/méd./max | std | CV | Forme |
|---|---|---|---|---|---|---|
| Beijing | 60 | 0 | 0,814 / 0,931 / 0,963 | 0,043 | 0,047 | unimodale, resserrée haut |
| London | 40 | 0 | 0,259 / 0,576 / 0,766 | 0,112 | 0,194 | unimodale, large étalement |
| Madrid | 35 | 11 (31%) | 0,332 / 0,470 / 0,620 | 0,062 | 0,130 | unimodale, resserrée |

**À déclarer dans la lettre (R2.6)** : E12 n'a réentraîné que 2 niveaux
intermédiaires (75%, 25%) au lieu des 5 niveaux canoniques de la courbe de
pruning guidée — réduction de budget assumée, décidée pour préserver les
≥5 tirages aléatoires par niveau plutôt que de les réduire. Les niveaux
100%/0% sont partagés avec le pruning guidé (identiques par construction
à ces extrêmes, non réentraînés). Table 8 (§6.2), Table 2 (§6.1) : pas de
réduction de budget équivalente, protocoles complets aux deux.

Rien rédigé dans le manuscrit ni dans un brouillon de §6.2.1/lettre — ce
sont les données brutes, organisées pour rédaction ultérieure.
