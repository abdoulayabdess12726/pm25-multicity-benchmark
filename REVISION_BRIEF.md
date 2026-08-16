# Contexte de révision — IJIES 20265149

Article : « Heterophily Limits Spatial Graph Encoding for PM2.5 Forecasting in
IoT Smart Cities ». Décision : major revision, 2 relecteurs. Ce dépôt doit
produire des résultats reproductibles pour la version révisée.

## Deux causes racines identifiées à l'audit

**1. Station MENDEZ ALVARO (Madrid).** Variance nulle sur la période
d'entraînement. Politique correcte, énoncée dans le manuscrit :
- CONSERVÉE dans le benchmark de prévision (Madrid = 7 stations)
- EXCLUE uniquement de l'analyse station-level d'hétérophilie (corrélation
  locale indéfinie)

Trois scripts en aval l'excluent à tort au chargement (sensibilité k,
baselines externes, pruning), ce qui fait tourner Madrid à 6 stations dans les
Tables 3 et 7 et en §6.2.1. C'est l'unique cause de toutes les incohérences
Madrid relevées par les relecteurs. Londres et Beijing sont cohérents partout
parce qu'aucune station n'y est exclue.

**2. Convention ddof.** Table 4 en ddof=0, Table 7 en ddof=1. Facteur
√(3/2) ≈ 1,225 sur 3 seeds. Convention retenue : **ddof=1 partout**.

## Règles non négociables

- `results/raw_results.csv` est **écrit une fois par les runs, jamais édité à
  la main**. Aucun script ne le modifie a posteriori.
- **Aucun chiffre du manuscrit n'est saisi manuellement.** Toutes les tables
  sont émises par `scripts/regenerate_tables.py` depuis `raw_results.csv`.
- **Si une assertion de cohérence échoue, on corrige les données ou le
  pipeline — jamais l'assertion.** Affaiblir un seuil pour faire passer le
  build est un échec, pas une solution.
- **Si un re-run contredit une affirmation du manuscrit, le signaler
  explicitement dans le rapport de tâche.** Ne pas réconcilier, ne pas
  arrondir vers la valeur attendue, ne pas choisir le seed qui arrange. Le
  papier est un résultat négatif : sa valeur tient entièrement à l'honnêteté
  des chiffres.
- Toute liste de stations vient d'une **source unique** (`configs/stations/`).
  Aucun script n'a sa propre liste en dur.
- Entraînements longs : lancer en arrière-plan avec log horodaté, ne jamais
  bloquer la session.
- **Toute nouvelle expérience prend son numéro dans la table ci-dessous
  (« Numérotation des expériences »), jamais attribué localement par une
  session.** Plusieurs sessions tournent en parallèle sur ce dépôt (cf.
  cette section) ; un numéro attribué localement crée une collision dès
  qu'une autre session en choisit un déjà pris.

## Décision d'agrégation MENDEZ ALVARO (P1, confirmée)

**MENDEZ ALVARO entre dans tous les agrégats sans traitement particulier par
modèle** — y compris quand un modèle donné y échoue nettement (XGBoost Madrid
R² individuel = −0.254 sur cette station, agrégat Madrid XGBoost recalculé
0.806→0.676). Aucune exclusion conditionnelle, aucun clipping, aucun
traitement à part.

**Motif** : exclure ou traiter à part une station parce qu'un modèle y
échoue serait une exclusion *conditionnée par le résultat* — exactement la
classe de décision responsable du bug initial (MENDEZ ALVARO exclue de 3
scripts en aval sans base dans le manuscrit). La règle d'agrégation doit
être identique pour toutes les lignes d'une table, indépendamment du chiffre
qu'elle produit.

**Diagnostic qui a précédé la décision** (voir aussi
`results/mendez_alvaro_diagnostic.md`) : train PM2.5 constant (std=0,
valeur=6.0, n=24544h) puis test normal (std=9.93, mean=11.01) — changement de
régime, pas juste une variance nulle isolée. Persistence/ARIMA/LSTM
s'en sortent (R²≈0.78-0.80, mécanisme autorégressif/skip qui réinjecte
l'observation réelle) ; XGBoost s'effondre (R²=−0.254, un seul leaf appris
sur des lags eux-mêmes constants pendant tout le train → aucune capacité
d'extrapolation). **Vérifié unique** : aucune des 26 autres stations (3
villes) n'a train_std < 0.5 avec test_std ≥ 0.5 — pas un mode de défaillance
systématique du filtre qualité §3.2, un cas isolé.

**Le filtre §3.2 n'est PAS modifié** : il ne teste que la variance de la
période de test, jamais celle du train — c'est une faille réelle (elle a
laissé passer MENDEZ ALVARO), mais la corriger en cours de révision serait
elle aussi une décision conditionnée par le résultat, et changerait la
composition d'autres villes de façon non contrôlée. Documentée comme limite
connue du protocole (à faire figurer dans la table de caractérisation R2.7,
voir ci-dessous), pas réparée.

## Table de caractérisation des jeux de données (R2.7, P9) — colonne ajoutée

En plus des colonnes prévues (période de monitoring, nb stations,
fournisseur, source météo, moyenne/variance PM2.5, autocorrélation lag-1,
R² persistance, % manquant, densité de graphe, degré effectif) : **ajouter
la variance PM2.5 de la période d'ENTRAÎNEMENT par station** (pas seulement
la variance globale/test). C'est le prédicteur direct du mode de défaillance
identifié sur MENDEZ ALVARO (train quasi-constant, test normal) — doit être
visible dans la table plutôt qu'enterré dans un diagnostic ponctuel.

## Numérotation des tables (confirmée contre le manuscrit soumis, cycle 20265149)

T1 indice h(D) · T2 benchmark par ville · T3 baselines externes · T4 tests
statistiques · T5 corrélation inter-villes · T6 ΔR² par station · T7
sensibilité k · T8 over-smoothing/GAT · T9 contrôles diagnostiques.

Table 3 (baselines externes) et Table 9 (contrôles diagnostiques) sont des
ajouts de ce cycle de révision, absents de la numérotation précédente
(Paper ID 20264131) — d'où un décalage de +1 sur toutes les tables à partir
de l'ancienne Table 3. Correspondance complète et scripts producteurs :
voir `README.md`, section « Reproducing the paper's tables ».

⚠️ Piège identifié : `e6_k_sensitivity.py` se référait en interne à
« Table 6 » (juste sous l'ancienne numérotation, où Table 6 = sensibilité k).
Sous la numérotation confirmée, Table 6 = ΔR² par station (un tableau
différent) et la sensibilité k est désormais Table 7. Le contenu du script
est correct (c'est bien de la sensibilité k), seul le label documentaire
était périmé — corrigé.

## Numérotation des expériences (E-series) — référence unique, toutes sessions

Ce dépôt est travaillé par plusieurs sessions en parallèle (P5 — sensibilité
k/pruning Madrid — n'est qu'une des lignes de travail en cours ; d'autres
sessions avancent en parallèle sur AirPhyNet et Chang-Zhu-Tan, sur consigne
directe de l'utilisateur, pas de collision de travail). Le numéro E<n> d'une
expérience est fixé **ici** et nulle part ailleurs — un script, un commit ou
un CHANGELOG ne doit jamais inventer son propre numéro localement.

| N° | Expérience | Statut (2026-08-16) |
|---|---|---|
| E9  | Sensibilité k Madrid, 7 stations (k∈{3,8}×2 topo×3 seeds) | **FAIT** (P5) |
| E10 | Pruning Madrid, 7 stations (5 niveaux×3 seeds) | **FAIT** (P5) |
| E11 | Parité seeds STGCN/Graph WaveNet (2 modèles×3 villes×2 seeds, 12 runs) | à faire |
| E12 | Contrôles pruning : élagage aléatoire à densité appariée + élagage inverse (garder les arêtes les plus hétérophiles) | à faire (scope de détail à préciser avant chiffrage des runs) |
| E13 | Table 8, over-smoothing/GAT (3 villes×4 variantes×3 seeds, 36 runs) | jamais lancée |
| E14 | Édition d'arêtes Beijing/London, seeds 123/777 (5 niveaux×2 seeds×2 villes, 20 runs) | à faire (Madrid déjà couvert par E10) |
| E15 | AirPhyNet, baseline post-2024 | étude de faisabilité faite (reproduction <1% d'écart, coûts mesurés — `external/AIRPHYNET_FEASIBILITY.md`), runs non lancés (0/9 : 3 villes×3 seeds) — session parallèle |
| E16 | Chang-Zhu-Tan, 4e réseau de validation externe de h(D) | pré-enregistrement fait (`PREREGISTRATION_CZT.md`, h(D)=0.312 mesuré avant entraînement), collecte de données en cours — session parallèle |

**Note de correspondance** : le commit `51fd7f1` (« E11 faisabilité AirPhyNet »)
utilisait un numéro provisoire attribué localement, avant que cette table ne
soit figée. Il correspond en réalité à **E15**. Le commit n'est **pas
réécrit** (historique git non modifié) — cette note fait foi pour toute
lecture future du log.

## Schéma de `results/raw_results.csv`

```
city, model, topology, k, seed, checkpoint_id, split_hash, n_stations,
station, split, rmse, mae, r2, run_id, config_path, git_commit, timestamp
```

Une ligne par station et une ligne agrégée (`station = "__aggregate__"`) par
condition. `split_hash` = hash des indices de découpe chronologique.

## Conventions statistiques

- Comparaisons agrégées : moyenne sur 3 seeds (42, 123, 777), ± écart-type
  ddof=1
- Comparaisons par station : seed primaire (42)
- Les deux niveaux ne sont **jamais mélangés** dans une même quantité dérivée
- Chaque baseline de référence porte une étiquette de protocole
  (`3seed_mean` ou `primary_seed`)
