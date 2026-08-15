# AUDIT — reconnaissance P0 (IJIES 20265149)

Reconnaissance uniquement, aucun fichier de pipeline modifié. Seuls
`REVISION_BRIEF.md`, l'ajout d'une ligne dans `CLAUDE.md`, et ce rapport ont
été écrits.

⚠️ **Point d'incertitude à lever avant P4** : aucun fichier manuscrit
(`.docx`/`.tex`) n'existe dans ce dépôt — impossible de vérifier directement
la numérotation actuelle des tables (20265149). `README.md` date du cycle
précédent (Paper ID 20264131, titre différent) et donne une numérotation qui
**contredit** les labels utilisés dans les scripts de cette session (voir
§5). Je me suis appuyé sur les labels les plus récents (docstrings des
scripts E6/E8 de cette session, qui reflètent vraisemblablement la
numérotation du manuscrit actuel) plutôt que sur README.md, mais ça reste une
inférence — à confirmer contre le manuscrit réel avant que P4 écrive des
noms de table en dur.

---

## 1. Listes de stations — définition et tous les points d'exclusion

### Source canonique actuelle (dispersée, pas centralisée)

- **Beijing** : dict codé en dur `BEIJING_COORDS` (`06_train_multistation.py:57-70`),
  12 stations. Dupliqué intégralement dans `05_compute_heterogeneity_v2.py:10`
  (deuxième copie du même dict).
- **London** : dérivée dynamiquement — `06_train_multistation.py:185-236`,
  intersection entre les stations présentes dans
  `data/london_processed/london_full_hourly.parquet` et
  `data/london_laqn/station_coords.csv`. 8 stations, aucune exclusion
  ad hoc.
- **Madrid** : dérivée dynamiquement — `06_train_multistation.py:239-297`,
  intersection entre `data/madrid_processed/madrid_full_hourly.parquet` et
  `data/madrid_openaq/station_coords_valid.csv`. **7 stations, MENDEZ ALVARO
  incluse** dans ce loader canonique (`N_STATIONS=7` confirmé à l'exécution).

### Fichier `data/stations_{beijing,london,madrid}.csv`

Existent (listes correctes, 12/8/7 stations, MENDEZ ALVARO présente pour
Madrid) mais **orphelins** : `grep` ne trouve aucun script qui les lit. Ce
sont des artefacts documentaires de l'Étape 1 du cycle précédent, jamais
branchés au pipeline. À réutiliser ou remplacer par `configs/stations/` en P1
— ne pas les dupliquer en plus.

### Tous les points où MENDEZ ALVARO / une exclusion est appliquée

| Fichier:ligne | Mécanisme | Niveau d'exclusion | Correct ? |
|---|---|---|---|
| `12_per_station_heterophily.py:84-99` | Corrélation NaN → `h_i` devient `NaN`, pas de filtre explicite en amont | Analyse hétérophilie station-level | **Correct** — charge les 7 stations via `b.load_madrid_data()`, laisse `h_i` indéfini pour MENDEZ (comportement voulu, documenté en commentaire) |
| `10_external_baselines.py:59,102-118,323-336` | `EXCLUDE = {"madrid": {"MENDEZ ALVARO"}}`, filtre appliqué dans `metrics_rows()` (per-station ET agrégat ARIMA/XGBoost/LSTM/Persistence) ET dans `agg_from_json()` (ré-agrège Linear/GCN-Transformer sur 6 stations au lieu de lire le champ agrégat 7-stations déjà présent dans le JSON) | **Reporting/agrégation seulement** — les données brutes 7-stations existent (JSON canonique intact), seule la vue Table 3-externe est tronquée | **Faux** — Madrid = 6 stations dans `external_baselines.csv`/`external_baselines_tables.md` |
| `13_edge_pruning.py:23,42,104-108` | `EXCLUDE = {"madrid": {"MENDEZ ALVARO"}}`, entraînement sur le graphe 7-nœuds (correct) mais `metrics_rows()` ne calcule/stocke le per-station **et** l'agrégat que pour les 6 stations `keep` | **Reporting/agrégation seulement**, mais contrairement à E1, les métriques par-station de MENDEZ n'ont jamais été calculées (boucle `for i in keep`, pas de calcul pour l'index exclu) | **Faux** — `edge_pruning.csv` Madrid = 6 stations |
| `e6_k_sensitivity.py:38,76-77,132` | `EXCLUDE = {"madrid": {"MENDEZ ALVARO"}}`, entraînement sur le graphe 7-nœuds (`n_nodes=c["n_nodes"]`, correct) mais le R² agrégé final (`ss_res`/`ss_tot` ou `r2_score`) n'est calculé que sur `kept` (6 stations) | **Reporting/agrégation seulement**, mais **seule la valeur agrégée est persistée** — aucune métrique per-station stockée, donc irrécupérable a posteriori sans ré-entraîner | **Faux** — `results/e6_k_sensitivity.csv` Madrid = 6 stations |
| `e8_k_sensitivity_3seeds.py:48,89` | Identique à e6 (hérite de la même logique `EXCLUDE`/`kept`) | Idem e6 | **Faux** — `results/table7_k_sensitivity.csv` Madrid = 6 stations |
| `08_sensitivity_k.py` (commentaire ligne 19-20) | Aucun `EXCLUDE` dans ce script — agrège explicitement sur les 7 stations (« Table 6 canonique ») | N/A — c'est la version qui ne reproduit PAS le bug | **Correct**, mais **incomplet** : `results/sensitivity_k_canonical.csv` ne contient que 8/54 lignes (Beijing k=3/5, London/Madrid k=5 seulement) — abandonné pour saturation swap MPS (voir `results/sensitivity_k_canonical_NOTE.md`), jamais terminé |
| `09_controls_oversmoothing.py`, `11_diagnostics.py` | Aucun `EXCLUDE` | N/A | **Correct**, 7 stations Madrid partout (vérifié : aucune occurrence de `EXCLUDE`/`MENDEZ` dans ces deux fichiers) |
| `14_sota_baselines.py` (cette session, E7) | Garde-fou explicite `EXPECTED_N["madrid"]=7`, lève une exception sinon | N/A — écrit exprès pour ce cycle de révision | **Correct**, vérifié à l'exécution (3 runs Madrid, `n_nodes=7` loggé à chaque fois) |

**Conclusion Q1** : le brief compte juste — trois familles de scripts
bugguées (E1 externe, E5 pruning, E6/E8 k-sensitivity), toutes par un même
schéma `EXCLUDE = {"madrid": {"MENDEZ ALVARO"}}` copié-collé indépendamment
trois fois (jamais une fonction partagée). **Nuance importante pour le coût
du correctif (P1/P5)** : E1 (baselines Linear/GCN de référence) est réparable
**sans ré-entraîner** — le JSON canonique a déjà les 7 stations, il suffit de
retirer le filtre dans `agg_from_json()`. Les lignes ARIMA/XGBoost/LSTM/
Persistence de E1 nécessitent de ré-exécuter ces baselines rapides (per-
station, pas de GNN) uniquement pour MENDEZ ALVARO — coût faible. E5
(pruning) et E6/E8 (k-sensitivity) n'ont **jamais calculé ni stocké** les
métriques de MENDEZ ALVARO : leur correction exige un ré-entraînement complet
du GCN sur Madrid (7 stations), ce que E9/E10 (P5) couvrent déjà.

---

## 2. Scripts producteurs, par expérience, et stations Madrid chargées

| Expérience | Script | Madrid — stations chargées (modèle) | Madrid — stations rapportées |
|---|---|---|---|
| Benchmark canonique (Tables 2/5 anciennes) | `06_train_multistation.py` | 7 | **7** |
| Statistiques (Wilcoxon, bootstrap, Cohen's d) | `07_statistical_analysis.py` | lit le JSON canonique (7) | **7** |
| Sensibilité k — version canonique (jamais finie) | `08_sensitivity_k.py` | 7 | **7** (mais incomplet, voir §1) |
| Sensibilité k — version effectivement committée (E6/E8, cette session) | `e6_k_sensitivity.py`, `e8_k_sensitivity_3seeds.py` | 7 (le GCN voit les 7 nœuds) | **6** (agrégat tronqué) |
| Baselines externes (ARIMA/XGBoost/LSTM) | `10_external_baselines.py` | 7 | **6** |
| Courbe de pruning hétérophile | `13_edge_pruning.py` | 7 (le GCN voit les 7 nœuds) | **6** (agrégat tronqué) |
| Contrôle shuffled-graph + ablation no-meteorology | `11_diagnostics.py` | 7 | **7** |
| GAT, GCN 1-couche, Dirichlet energy / over-smoothing | `09_controls_oversmoothing.py` | 7 | **7** |
| STGCN + Graph WaveNet (E7, cette session) | `14_sota_baselines.py` | 7 | **7** |
| Export adjacence (livrable reviewers) | `graphs/export_adjacency.py` | à vérifier — pas audité en détail (hors périmètre des 5 questions, mais probablement hérite de `06`, donc 7) | — |
| Export per-station (livrable reviewers) | `results/export_per_station.py` | lit le JSON canonique (7) | **7** (vérifié : `per_station_seed_topology.csv` contient bien 7 stations Madrid dont MENDEZ ALVARO) |

---

## 3. Tous les calculs d'écart-type et leur ddof effectif

| Fichier:ligne | Appel | ddof effectif | Alimente probablement |
|---|---|---|---|
| `06_train_multistation.py:761-763,806,981-985` | `np.std(...)` (pas d'argument) | **0** (défaut numpy) | Champs `*_std` du JSON canonique (`results/{city}/multistation_results.json`) — source du R2_std utilisé par plusieurs scripts en aval |
| `08_sensitivity_k.py:151,153,156` | `.std(ddof=0)` explicite | **0** | `sensitivity_k_canonical.csv` (incomplet) |
| `10_external_baselines.py:308` | `sub[m].std(ddof=0) if len(sub)>1 else 0.0` | **0** explicite | `external_baselines.csv` (ARIMA/XGBoost/LSTM agrégats) |
| `10_external_baselines.py:336` | `np.std(v)` (pas d'argument, `v` = liste Python) | **0** | `agg_from_json()` — lignes Linear/GCN-Transformer dans `external_baselines_tables.md` |
| `07_statistical_analysis.py:86` | `delta_r2_seeds.std()` où `delta_r2_seeds = np.array(...)` | **0** (défaut numpy) | `results/statistical_analysis/*` — probable source de l'actuelle « Table 4 » |
| `05_compute_heterogeneity_v2.py:46` | `df_wide.std(axis=0)` où `df_wide` est un **DataFrame pandas** | **1** (défaut pandas !) | `heterogeneity_index_v2.csv` — donc l'indice h(D) lui-même utilise déjà ddof=1, incohérent avec le reste |
| `09_controls_oversmoothing.py:337` | `np.std(per_seed)` (liste Python) | **0** | Sorties over-smoothing/Dirichlet |
| `e8_k_sensitivity_3seeds.py` (`aggregate()`, cette session) | `df.groupby(...)["delta_R2"].agg(..., std="std")` où `df` est un **DataFrame pandas** | **1** (défaut pandas !) | `results/table7_k_sensitivity.csv` — **c'est très probablement la « Table 7 en ddof=1 » citée dans le brief** |

**Conclusion Q3** : le brief a raison — il y a bien un mélange ddof=0/ddof=1,
mais **la cause n'est pas un choix explicite incohérent, c'est un défaut
implicite différent entre numpy (`.std()` → ddof=0) et pandas (`.std()` →
ddof=1)** selon que le code agrège une liste/array numpy ou un DataFrame. Ce
piège s'est reproduit à trois endroits indépendants (`06`, `07`,
`05_compute_heterogeneity_v2.py` en ddof=1 côté pandas ; `e8_k_sensitivity_3seeds.py`
que **j'ai écrit moi-même dans cette session précédente** est également en
ddof=1 via pandas, sans que je m'en rende compte à l'époque). La correction
P2 (fonction unique `agg_mean_std`) est la bonne réponse structurelle — un
appel centralisé élimine ce piège au lieu de corriger chaque site un par un.
Note : `05_compute_heterogeneity_v2.py` (indice h(D), Table 1) est **déjà**
en ddof=1 — à vérifier si h(D) doit aussi passer par `agg_mean_std` en P2 ou
si son cas (dispersion intra-ville sur les stations, pas sur les seeds) est
hors du périmètre ddof=0→1 décrit dans le brief (le brief ne parle que des
3 seeds, pas de dispersion inter-stations — probablement hors périmètre, à
confirmer).

---

## 4. Stockage actuel des résultats — totalement dispersé, pas de fichier brut unifié

Aucun `results/raw_results.csv` n'existe. Inventaire des artefacts actuels
(hors `results_backup_*/`, hors dossiers de logs de cette session) :

- **3 JSON canoniques** : `results/{beijing,london,madrid}/multistation_results.json`
  (source de vérité pour le benchmark principal, k=5, Linear/GCN-Transformer,
  3 seeds, per-station — mais format JSON imbriqué, pas tabulaire)
- **8 CSV d'expérience indépendants**, schémas tous différents :
  `e6_k_sensitivity.csv`, `e6_k_sensitivity_seed42_only.csv` (archive),
  `table7_k_sensitivity.csv`, `external_baselines.csv`, `edge_pruning.csv`,
  `diagnostics.csv`, `per_station_heterophily.csv`, `per_station_seed_topology.csv`,
  `sensitivity_k_canonical.csv`, `table3_sota_baselines.csv`,
  `heterogeneity_index.csv`, `heterogeneity_index_v2.csv`
- **1 dossier `results/sensitivity_k/`** : reliquat de l'ancienne version
  `--quick` (18 fichiers `.json`/`.txt` par ville×k, plus `summary.csv`) —
  périmé, cf. `sensitivity_k_canonical_NOTE.md` qui documente explicitement
  que ces valeurs `--quick` sont fausses (Beijing k=3 donnait +0.213 au lieu
  de −0.013 sous le protocole complet)
- **1 dossier `results/statistical_analysis/`** : 2 JSON + 2 CSV
  (`stats_results.json`, `stats_results_3cities.json`,
  `summary_for_paper.csv`, `summary_for_paper_3cities.csv`)
- **Logs texte de cette session** (`e7_*.log`, `e8_run.log`) : sorties
  console redirigées, pas structurées, utiles pour l'audit temporel mais pas
  comme source de données

**Conclusion Q4** : consolidation en `raw_results.csv` (P3) est justifiée —
il n'y a aujourd'hui aucune source unique interrogeable, et au moins deux
schémas de CSV divergent déjà sur le sens de la colonne `station`
(`"__aggregate__"` dans `10`/`13`, agrégat implicite via `r2_score` direct
sans ligne dédiée dans `e6`/`e8`/`14`).

---

## 5. Tables du manuscrit : générées par code vs saisies à la main

**Aucun fichier manuscrit (.docx/.tex) n'est présent dans ce dépôt** — le
tableau ci-dessous est donc basé sur (a) les scripts qui *pourraient* nourrir
chaque table et (b) les rares mentions explicites de numéros de table dans le
code/les docs. Deux sources de numérotation **se contredisent** :

- `README.md` (cycle précédent, Paper ID 20264131) : Table 1=h(D),
  Tables 2/5=benchmark principal, **Table 3=stats** (`07_statistical_analysis.py`),
  **Table 6=k-sensitivity** (`08_sensitivity_k.py`), **Table 7=over-smoothing/GAT/
  Dirichlet** (`09_controls_oversmoothing.py`).
- Docstrings des scripts de cette session (cycle actuel probable, Paper ID
  20265149) : `e6_k_sensitivity.py` se réfère à lui-même comme **« R6 / Table 6 »**,
  `e8_k_sensitivity_3seeds.py` comme **« R2.5 / Table 7 »** — cohérent avec le
  brief qui parle de « Table 7 » pour la sensibilité k, **pas** pour
  l'over-smoothing.

Le titre de l'article a changé au moins deux fois entre les sessions
observées dans ce dépôt (« Spatial Graph Encoding for AI-Based PM2.5... » →
« When Does Spatial Graph Encoding Help?... » → « Heterophily Limits Spatial
Graph Encoding... »), ce qui rend une renumérotation des tables tout à fait
plausible d'une resoumission à l'autre. **Je ne peux pas trancher avec
certitude laquelle des deux numérotations est actuellement correcte** sans
le manuscrit lui-même — à confirmer avant que P4 nomme des fichiers de sortie
`table_4.md` / `table_7.md` en dur.

Sous cette réserve, la distinction générée-par-code / saisie-à-la-main :

| Contenu | Généré par code ? |
|---|---|
| `results/*.csv`, `results/*/multistation_results.json` | Oui — sortie directe de chaque script d'expérience |
| `results/*_tables.md`, `results/*_summary.md` (ex. `external_baselines_tables.md`, `diagnostics_summary.md`, `table3_sota_baselines_summary.md`) | Oui, générés par le script correspondant à la fin du run |
| `README.md` — sections « Key results » (tableaux markdown avec chiffres h(D), ΔR², p, d) | **Non** — valeurs copiées-collées à un instant donné, aucun script ne régénère `README.md`. Déjà partiellement périmées (le README date de 20264131 ; les valeurs ΔR² Madrid qu'il affiche, −0.321/−0.380, sont d'ailleurs les bonnes valeurs 7-stations, cohérentes avec le JSON canonique — donc le README n'est pas contaminé par le bug MENDEZ ALVARO, contrairement aux CSV d'expérience aval) |
| `CLAUDE.md` — bloc « Résultats de référence du papier » (h(D), ΔR², 26/27 stations) | **Non** — texte libre, recopié manuellement depuis le papier au moment de la rédaction du brief précédent |
| `results/sensitivity_k_canonical_NOTE.md` | Mixte — généré partiellement (chiffres canoniques) puis complété à la main (comparaison avec l'« ancienne Table 6 `--quick` ») |

**Conclusion Q5** : il n'existe aujourd'hui **aucun script qui écrit
directement dans un document manuscrit** — tout ce qui ressemble à une
« table du papier » dans ce dépôt est soit une sortie CSV/MD brute par
expérience (correcte, mais 8+ fichiers disjoints, cf. Q4), soit un
copié-collé figé dans `README.md`/`CLAUDE.md`. `scripts/regenerate_tables.py`
(P4) n'a donc pas de prédécesseur direct à remplacer — c'est une brique
entièrement nouvelle, pas un refactor.

---

## Recommandation avant P1

Rien ne bloque P1 (la source unique de stations est indépendante de la
question de numérotation des tables). Mais avant P4, confirmer contre le
manuscrit réel (20265149) quels scripts/tables correspondent à « Table 4 »
et « Table 7 » — les deux sources internes au dépôt se contredisent (voir
§5), et une mauvaise correspondance ferait écrire des assertions de
cohérence formellement correctes mais comparant les mauvaises tables entre
elles.
