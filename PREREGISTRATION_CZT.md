# Pré-enregistrement — Chang-Zhu-Tan comme 4e réseau de validation externe de h(D)

**Date de rédaction : 2026-08-15.**
**Statut au moment de la rédaction : AUCUN entraînement lancé sur ce jeu de
données. Aucun ΔR² Chang-Zhu-Tan n'existe, ni dans ce repo ni ailleurs.**

Ce fichier est daté et versionné *avant* tout calcul de ΔR². Sa seule valeur
tient à ce qu'il ne soit pas révisé après coup. Si le résultat contredit la
prédiction ci-dessous, la prédiction reste dans le manuscrit telle quelle et
la contradiction est rapportée (règle permanente n°5 de `CLAUDE.md`).

## 1. Contexte

Les trois réseaux du papier sont tous des cas **hétérophiles**, où l'encodage
spatial dégrade la performance :

| Réseau  | h(D) | ΔR² (dist) | ΔR² (corr) |
|---------|------|------------|------------|
| Beijing | 0.497 | −0.017 | −0.038 |
| London  | 0.656 | −0.375 | −0.401 |
| Madrid  | 0.728 | −0.321 | −0.380 |

L'indice n'a donc jamais été testé du côté où il prédit un effet **positif ou
neutre**. C'est la critique du relecteur 2 : trois points tous du même côté du
seuil ne testent pas la thèse, ils l'illustrent.

Chang-Zhu-Tan (Changsha–Zhuzhou–Xiangtan, Hunan, 22 stations CNEMC, 2020–2023)
fournit ce point manquant.

## 2. Valeur de h(D) mesurée AVANT tout entraînement

Calculée sur `MSDGNN-Lu et al- PLOS/air_quality_data.npz` (PM2.5, 22 stations),
topologie corrélation, k = 5, période train = 70 % initiaux, convention
identique à `05_compute_heterogeneity_v2.py` :

**h(D) = 0.312** (h_i par station : min 0.207, max 0.437)

Corrélation PM2.5 inter-stations toutes paires : moyenne 0.512 (min 0.317,
max 0.794).

Cette valeur est **inférieure aux trois réseaux du papier**, et notablement
inférieure à Beijing (0.497), le cas le moins hétérophile connu.

## 3. Prédiction (hors échantillon)

Sous l'hypothèse de l'article — le bénéfice de l'encodage spatial décroît de
façon monotone avec h(D) —, h(D) = 0.31 prédit :

**P1 (principale).** ΔR² agrégé (3 seeds, ddof=1) ≥ −0.02 pour les deux
topologies. Autrement dit le GCN-Transformer est **neutre ou meilleur** que le
Linear-Transformer, jamais nettement moins bon.

**P2 (ordonnancement).** ΔR²(Chang-Zhu-Tan) > ΔR²(Beijing) pour chaque
topologie, c.-à-d. > −0.017 (dist) et > −0.038 (corr).

**P3 (par station).** Au moins 50 % des 22 stations ont ΔR² ≥ 0 au seed 42,
topologie distance — à contraster avec 26/27 stations *sous-performantes* sur
les trois réseaux hétérophiles.

**Falsification.** P1 est fausse si ΔR² < −0.02 sur l'une des deux topologies.
Un ΔR² de l'ordre de −0.3 (magnitude London/Madrid) réfuterait l'indice comme
prédicteur utilisable, pas seulement son calibrage.

## 4. Ce qui est figé avant l'entraînement

- **Découpe** : chronologique 70/15/15 (protocole du papier). MSDGNN utilise
  60/20/20 ; on ne s'aligne pas dessus, la comparabilité recherchée est
  interne (Beijing/London/Madrid), pas avec les chiffres publiés de MSDGNN.
- **Seeds** : 42, 123, 777. SEQ_LEN 24, horizon 1 h, MAX_EPOCHS 50, PATIENCE 8,
  K_NEIGHBORS 5, MinMax scaling.
- **Topologies** : distance (si coordonnées récupérées) et corrélation.
- **Agrégation** : identique pour toutes les stations, aucune exclusion
  conditionnée par le résultat (cf. décision MENDEZ ALVARO, `REVISION_BRIEF.md`).
- **Filtre qualité §3.2** : appliqué tel quel, non modifié.
- **h(D) définitif** : recalculé sur le jeu reconstruit (cf. §5) avec la même
  définition. Si la valeur bouge, elle est rapportée ; les prédictions P1–P3
  ci-dessus restent celles associées à h(D) ≈ 0.31 et ne sont pas réécrites.

## 5. Provenance des données (état au 2026-08-15)

Le `.npz` de MSDGNN est identifié : les 22 colonnes correspondent aux codes
CNEMC **1335A–1344A, 1508A, 1511A–1515A, 1518A–1520A, 1524A, 1559A, 1562A**
(vérifié par correspondance exacte des valeurs PM2.5 horaires du 2020-01-01
contre l'archive nationale : 20 stations sur 22 à 24/24 heures identiques).
Les codes 1339A et 1559A, cités comme meilleur et pire cas dans l'article de
Lu et al., figurent bien dans cette liste. Ordre des variables confirmé :
PM2.5, PM10, SO2, NO2, O3, CO, AQI.

Limites du `.npz` identifiées avant usage :
- PM2.5 et AQI sont bruts ; PM10, SO2, NO2, O3 ont subi un lissage
  exponentiel α = 0.5 (vérifié sur 1339A, 2020-01-01).
- Colonne 0 (1335A) n'est pas à valeurs entières, contrairement aux 21 autres :
  série reconstruite/interpolée.
- Aucun horodatage. Des heures manquantes ont été supprimées sans être
  marquées : le décalage index↔horloge s'accumule (≈ −24 h à mi-2020,
  ≈ −522 h fin novembre 2023), et la période diurne effective mesurée en
  index vaut 23.954 h au lieu de 24.

En conséquence, le jeu utilisé sera **reconstruit depuis la source CNEMC**
pour ces 22 codes sur 2020-2023, pas repris du `.npz`. Le `.npz` sert
uniquement à identifier stations et période.

## 6. Ajout daté 2026-08-16 — h(D) pré-enregistré vs h(D) reconstruit

*Section ajoutée après le commit initial `dadb848`. Les sections 2 et 3 sont
inchangées et le restent.*

| Valeur | Source | État |
|--------|--------|------|
| **h(D) = 0.312** | `.npz` MSDGNN, PM2.5, corr k=5, train 70 % | pré-enregistrée §2 |
| **h(D) = à venir** | reconstruction CNEMC 2020-2023, même définition | en cours de collecte |

**Quels artefacts du `.npz` peuvent déplacer h(D), et lesquels ne le peuvent
pas.** L'indice se calcule sur PM2.5 seul, à partir de corrélations
inter-stations. Les deux défauts les plus visibles du `.npz` n'agissent donc
pas dessus :

- **Le lissage EWMA α = 0.5 ne touche pas h(D)** : il porte sur PM10, SO2, NO2
  et O3. PM2.5 et AQI sont bruts (vérifié sur 1339A, 2020-01-01, 24/24 heures
  identiques à la source). L'entrée de l'indice n'est pas lissée.
- **Le décalage temporel ne touche pas h(D) non plus** : les heures supprimées
  le sont pour *toutes* les stations à la fois, puisque c'est le même index de
  ligne. Le décalage index↔horloge (jusqu'à −522 h) casse l'alignement avec des
  covariables externes comme la météo, mais laisse l'alignement
  station-à-station intact — donc les corrélations inter-stations aussi.

Trois mécanismes peuvent en revanche déplacer la valeur, tous à la hausse
attendue de la corrélation (donc à la baisse de h) dans le `.npz` :

1. **1335A est interpolée** (seule colonne non entière) : une série lissée
   corrèle mécaniquement mieux avec ses voisines. Sur la reconstruction elle
   redevient brute, avec ses trous.
2. **Comblement des manquants** dans le `.npz` (zéro NaN sur 34305 × 22) :
   même effet, réparti sur toutes les stations.
3. **Filtre qualité §3.2 et fenêtre de train** : la composition des stations et
   la borne des 70 % changent légèrement après reconstruction.

**Attente, formulée avant le calcul** : h(D) reconstruit reste dans la région
homophile, plutôt légèrement au-dessus de 0.312 (les trois mécanismes vont dans
le même sens), et très en deçà de Beijing (0.497). Une valeur autour de
0.30–0.40 ne change pas le sens des prédictions P1–P3.

**Règle d'arrêt.** Si h(D) reconstruit franchit le seuil au-delà duquel
l'indice prédit une dégradation — opérationnellement, s'il atteint ou dépasse
0.497 (Beijing, ΔR² = −0.017) —, alors la prédiction P1 change de sens et
l'entraînement est suspendu : la valeur est rapportée à l'auteur avant tout
lancement, et P1–P3 sont conservées telles quelles comme prédiction réfutée
par le seul changement de jeu de données.

---

## 7. Ajout daté 2026-08-23 — h(D) reconstruit calculé, résultat du prétraitement

*Section ajoutée après le commit initial `dadb848` et l'ajout du 2026-08-16
(section 6, ci-dessus). Les sections 2 et 3 sont inchangées et le restent.*

**Prétraitement** (`01i_preprocess_czt.py`) : reconstruction depuis
`data/czt_raw/` (CNEMC 2020-2023, 20 stations — cf. `configs/stations/czt.yaml`
pour l'écart 22→20, décommissionnement post-2023 de 1344A/1559A, confirmé
présents dans l'historique 2020-2023). Jonction météo Open-Meteo sur 3 points
ville (centroïdes Changsha/Zhuzhou/Xiangtan, un point par ville plutôt que par
station — 20 stations mais seulement 3 climats locaux distincts). Filtre §3.2
réel (couverture PM2.5 brute ≥ 50 %, non imputée) : **20/20 stations
retenues**, couverture brute 89,0 % à 98,4 % selon la station (aucune
proche du seuil). Aucun mode variance-train-faible/variance-test-normale
détecté (pas d'équivalent MENDEZ ALVARO sur ce réseau).

**h(D) reconstruit** (topologie corrélation, k=5, période train = 70 %
initiaux, convention identique à `05_compute_heterogeneity_v2.py`, comme la
valeur pré-enregistrée) :

| | r̄ | Moran's I | CV (normalisé) | h(D) |
|---|---|---|---|---|
| Pré-enregistré (§2, .npz MSDGNN) | 0.512 | — | — | **0.312** |
| Reconstruit (ce calcul, CNEMC) | 0.904 | 0.262 | 0.404 | **0.413** |

**Écart : +0.101.** Dans le sens anticipé par la section 6 (les trois
mécanismes identifiés — 1335A non lissée, pas de comblement artificiel des
manquants, fenêtre train légèrement différente — poussent tous vers un h(D)
plus élevé après reconstruction) et dans la fourchette annoncée
« 0.30–0.40 » à 0.01 près (légèrement au-dessus, pas en dehors du
raisonnement qui sous-tendait cette fourchette).

**Règle d'arrêt (§6) : NON déclenchée.** h(D) = 0.413 < 0.497 (seuil
Beijing) — la prédiction P1 ne change pas de sens sous ce résultat. CZT
reste, comme prévu, notablement moins hétérophile que les trois réseaux du
manuscrit (Beijing 0.497 le plus proche, London 0.656, Madrid 0.728).

**Correctif build_correlation_graph (P5, 2026-08-21) vérifié sur CZT** :
aucune corrélation indéfinie (0 NaN sur la matrice 20×20, contrairement à
Madrid/MENDEZ ALVARO) — le nombre d'arêtes réelles égale exactement n×k_eff
à k∈{3,5,8} (60/100/160 pour n=20), confirmé par
`tests/test_correlation_graph.py::test_czt_correlation_graph_edge_count_is_exactly_n_times_k`.
Le correctif ne change donc rien pour CZT en tant que tel (rien à corriger
puisqu'aucun NaN) — seule la construction du graphe canonique aurait pu être
affectée si CZT avait eu une station dégénérée, ce qui n'est pas le cas.

**Aucun entraînement lancé.** Ce prétraitement produit `data/czt_processed/`
(PM2.5 + météo, format harmonisé 5 features) et confirme que la règle
d'arrêt ne s'applique pas — la décision de lancer l'entraînement (E16)
reste à prendre séparément.

## 8. Ajout daté 2026-08-23 — résultat de l'entraînement, prédiction P1 réfutée

*Section ajoutée après les sections 6 et 7 ci-dessus. Les sections 2 et 3
sont inchangées et le restent.*

**Protocole** : identique aux 3 autres réseaux (GCN-Transformer et
Linear-Transformer, topologies distance et correlation, k=5, seeds
42/123/777, splits chronologiques 70/15/15, MinMax train seul,
`06_train_multistation.py --city czt --graph both`, 372,6 min).

| Topologie | R² GCN | R² Linear | ΔR² |
|---|---|---|---|
| distance | 0,9375±0,0008 | 0,9586±0,0034 | **−0,0211** |
| correlation | 0,9272±0,0010 | 0,9586±0,0034 | **−0,0314** |

GCN supérieur au Linear-Transformer sur **0/20 stations** (seed primaire).

**P1 réfutée, dans les deux topologies — rapporté tel quel, non lissé.**
P1 prédisait ΔR² ≥ −0,02. Résultat : −0,0211 (distance, juste en dessous du
seuil) et −0,0314 (correlation, plus nettement en dessous). h(D)=0,413
prédisait correctement la **direction relative** (CZT moins dégradée que
Beijing/London/Madrid, confirmé — cf. section suivante) mais pas le **signe
absolu** annoncé par le seuil P1 (« neutre ou meilleur »). C'est la valeur
même de ce pré-enregistrement : la prédiction était fausse sur ce point
précis, et c'est publié tel quel, pas ajusté après coup.

**Position sur la courbe h(D)↔ΔR², 4 réseaux :**

| Ville | h(D) | ΔR² distance | ΔR² correlation |
|---|---|---|---|
| CZT | 0,413 | −0,0211 | −0,0314 |
| Beijing | 0,497 | −0,0172 | −0,0375 |
| London | 0,656 | −0,3754 | −0,4014 |
| Madrid | 0,728 | −0,3213 | −0,4144 |

Spearman h(D)↔ΔR², n=4 : **correlation topologie ρ=−1,000** (ordre
parfaitement monotone — CZT < Beijing < London < Madrid en h(D), même ordre
en ΔR² le moins au plus négatif). **Distance topologie ρ=−0,600, p=0,40**
(non significatif à n=4) : CZT (−0,0211) légèrement plus dégradée que
Beijing (−0,0172) malgré un h(D) plus bas — inversion locale mineure entre
deux valeurs proches de zéro, pas comparable au renversement franc
Madrid/London documenté en Table 2 (cf. `CHANGELOG_TABLES.md`).

---

## 9. Ajout daté 2026-08-24 — incohérence de convention trouvée, h(D)=0,469 (jeu complet) devient la valeur de Table 1/5

En préparant Table 1/Table 5 (4 réseaux, tâche de livraison manuscrit),
`05_compute_heterogeneity_v2.py` — le script qui produit les h(D) publiés de
Beijing/London/Madrid (0,497/0,656/0,728) — a été étendu à CZT (même
fonction `heterogeneity()`, non modifiée ; nouveau bloc de chargement
uniquement, cf. commit de cette session). Résultat sur le **jeu complet** :
**h(D) = 0,469** (r̄=0,9068, Moran's I=0,0930, CV_norm=0,4080), pas 0,413.

**Cause identifiée avec certitude.** La valeur 0,413 de la §7 ci-dessus a
été calculée sur le **slice train uniquement (70% initiaux)**, malgré la
mention « convention identique à `05_compute_heterogeneity_v2.py` » — cette
mention était inexacte : ce script ne slice jamais le train pour
Beijing/London/Madrid, il utilise le jeu complet partout. Vérifié en
recalculant les deux slices sur les mêmes données CZT : train-only reproduit
0,413 quasi exactement (r̄=0,9036, Moran's I=0,2615, h=0,4131) ; jeu complet
donne 0,469. L'écart vient presque entièrement de Moran's I (0,262 vs
0,093) — la structure spatiale des valeurs moyennes par station diffère
sensiblement entre la fenêtre train et la période complète.

**Décision (validée par l'utilisateur, 2026-08-24)** : Table 1, Table 5,
Figure 3 et toute analyse comparant h(D) entre les 4 réseaux utilisent
désormais **h(D)=0,469** pour CZT — cohérent avec la convention jeu-complet
utilisée pour les 3 autres réseaux. Les valeurs 0,312 (§2, pré-enregistrée)
et 0,413 (§6-§8, reconstruite train-only) restent dans ce document telles
qu'écrites à l'époque — ne pas les réécrire — mais ne sont plus la référence
utilisée dans le manuscrit à partir de cette date.

**Effet sur la conclusion de P1 (§8 ci-dessus)** : le sens ne change pas —
0,469 &lt; 0,497 (Beijing), CZT reste le réseau le moins hétérophile des 4,
« P1 réfutée sur le signe absolu, pas sur la direction relative » reste
valide mot pour mot. La marge se réduit cependant nettement (0,084 → 0,028)
et doit être présentée comme telle, pas arrondie vers l'écart précédent.
Le tableau « Position sur la courbe h(D)↔ΔR² » et le Spearman à 4 réseaux
de la §8 doivent être recalculés avec 0,469 avant citation dans le
manuscrit — non refaits ici (cf. matériau §5.9 livré séparément).

---

*Enregistré avant tout entraînement. Ne pas modifier les sections 2 et 3.*
