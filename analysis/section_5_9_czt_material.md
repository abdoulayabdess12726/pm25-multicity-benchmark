# Matériau pour §5.9 (Chang-Zhu-Tan) — faits assemblés, PAS de prose rédigée

> Ce fichier rassemble des faits vérifiés, avec sources exactes. Il ne
> contient aucune phrase destinée à être copiée telle quelle dans le
> manuscrit — c'est la matière première pour rédiger §5.9, référencée
> depuis §5.5 et §6.3.

## 1. Chronologie du pré-enregistrement (`PREREGISTRATION_CZT.md`)

| Date | Événement | Commit(s) |
|---|---|---|
| 2026-08-15 | Rédaction initiale, AVANT tout entraînement. h(D)=0,312 mesuré sur le `.npz` MSDGNN (22 stations), 3 prédictions figées. | `dadb848` (commit initial du fichier) |
| 2026-08-16 | Ajout §6 : distinction h(D) pré-enregistré (0,312, `.npz` MSDGNN) vs h(D) à reconstruire (source CNEMC propre) | — |
| 2026-08-23 | Ajout §7 : prétraitement réel (`01i_preprocess_czt.py`), 20/20 stations retenues (décommissionnement 1344A/1559A post-2023), h(D) reconstruit = 0,413 (calculé sur slice **train-only**, 70% initiaux) | `4367e19` (prétraitement) |
| 2026-08-23 | Intégration CZT dans `06_train_multistation.py`, lancement entraînement | `466046c` |
| 2026-08-23 | Ajout §8 : résultat entraînement (372,6 min), P1 réfutée dans les 2 topologies | `770f083` |
| 2026-08-24 | Ajout §9 : incohérence trouvée — h(D)=0,413 utilisait un slice train-only alors que `05_compute_heterogeneity_v2.py` (source de Beijing/London/Madrid) utilise le jeu complet. Recalculé sur jeu complet : **h(D) = 0,469**. Devient la valeur de référence Table 1/5/Figure 3. | commit de cette session (05_compute_heterogeneity_v2.py étendu à CZT) |

## 2. Les trois valeurs de h(D) CZT — ne pas en confondre une pour une autre

| Valeur | h(D) | Méthode | Statut |
|---|---|---|---|
| Pré-enregistrée | **0,312** | `.npz` MSDGNN (22 stations), slice train 70%, AVANT tout entraînement | Historique, jamais réécrite (§2 du pré-enregistrement) |
| Reconstruite (§7) | **0,413** | Source CNEMC propre (20 stations), slice train 70% — même slice que la valeur pré-enregistrée, mais PAS la même convention que Beijing/London/Madrid | Historique, jamais réécrite ; incohérence documentée §9 |
| **Finale (utilisée manuscrit)** | **0,469** | Source CNEMC propre (20 stations), **jeu complet** — même convention que Beijing (0,497), London (0,656), Madrid (0,728) | **Valeur à citer dans Table 1, Table 5, Figure 3, §5.9** |

Composantes de la valeur finale (jeu complet, `results/heterogeneity_index_v2.csv`) :
r̄=0,9068 · Moran's I=0,0930 · CV_normalisé=0,4080 · h(D)=0,4694

## 3. Les trois prédictions pré-enregistrées et leur statut réel

**P1 (principale)** — texte pré-enregistré : « ΔR² agrégé (3 seeds, ddof=1) ≥ −0,02 pour les deux topologies. »
Résultat : distance −0,0211 ; correlation −0,0314. **Réfutée dans les 2 topologies.**

**P2 (ordonnancement)** — texte pré-enregistré : « ΔR²(CZT) > ΔR²(Beijing) pour chaque topologie, c.-à-d. > −0,017 (dist) et > −0,038 (corr). »
Résultat :
- distance : CZT=−0,0211 vs Beijing=−0,0172 → **−0,0211 n'est PAS > −0,0172 : P2 réfutée en distance** (inversion locale, les deux valeurs sont proches de zéro).
- correlation : CZT=−0,0314 vs Beijing=−0,0375 → **−0,0314 > −0,0375 : P2 confirmée en correlation.**
- **P2 est donc mixte : confirmée en correlation, réfutée en distance** — pas explicitement scoré comme tel dans le pré-enregistrement lui-même (qui note l'inversion sans utiliser l'étiquette « P2 »), à formuler clairement en §5.9.

**P3 (par station)** — texte pré-enregistré : « Au moins 50% des 22 stations ont ΔR² ≥ 0 au seed 42, topologie distance. »
Note : le nombre de stations réel après filtre S3.2 est **20**, pas 22 (2 stations décommissionnées post-2023, cf. §7 du pré-enregistrement) — la prédiction reste formulée sur 22 mais s'évalue sur les 20 stations retenues.
Résultat : **0/20 stations** ont ΔR² ≥ 0 (seed 42, distance) — confirmé aussi en correlation (0/20). **P3 nettement réfutée** (0% au lieu du seuil 50%).

## 4. ΔR² CZT par topologie (3 seeds, ddof=1) — valeurs manuscrit actuelles

| Topologie | R² GCN-Transformer | R² Linear-Transformer | ΔR² | Wilcoxon p (Holm-Bonf) | Cohen's d | GCN<Linear |
|---|---|---|---|---|---|---|
| distance | 0,9375 ± 0,0008 | 0,9586 ± 0,0034 | **−0,0211 ± 0,0041** | 7,629e-06 | −3,07 | 20/20 |
| correlation | 0,9272 ± 0,0010 | 0,9586 ± 0,0034 | **−0,0314 ± 0,0034** | 7,629e-06 | −3,16 | 20/20 |

_Source : `manuscript/tables/table4_statistical_tests.md` (régénéré depuis `raw_results.csv`, commit de cette session). Protocole d'entraînement : `06_train_multistation.py --city czt --graph both`, seeds 42/123/777, k=5, splits chronologiques 70/15/15, MinMax train seul, durée mesurée 372,6 min (`results/e16_run.log`)._

## 5. Spearman h(D) ↔ ΔR², 4 réseaux (valeurs manuscrit actuelles, h(D) CZT=0,469)

| Topologie | ρ | p | n |
|---|---|---|---|
| distance | −0,600 | 0,4000 | 4 |
| correlation | **−1,000** | **0,0000** | 4 |

_Source : `manuscript/tables/table5_cross_city_correlation.md`. Rang h(D) inchangé par la correction 0,413→0,469 : CZT (0,469) < Beijing (0,497) < London (0,656) < Madrid (0,728) — CZT reste le réseau le moins hétérophile des 4 dans les deux calculs de h(D)._

Tableau de position, 4 réseaux (h(D) le plus bas au plus haut) :

| Réseau | h(D) | ΔR² distance | ΔR² correlation |
|---|---|---|---|
| Chang-Zhu-Tan | 0,469 | −0,0211 | −0,0314 |
| Beijing | 0,497 | −0,0172 | −0,0375 |
| London | 0,656 | −0,3754 | −0,4014 |
| Madrid | 0,728 | −0,3213 | −0,4144 |

## 6. Caractérisation du réseau CZT

| Champ | Valeur | Source |
|---|---|---|
| Villes couvertes | Changsha, Zhuzhou, Xiangtan (Hunan) | `PREREGISTRATION_CZT.md` §1 |
| Stations pré-enregistrées | 22 (identifiées via `.npz` MSDGNN) | §5 |
| Stations retenues après filtre S3.2 | **20** (2 décommissionnées post-2023 : 1344A, 1559A) | §7 |
| Fournisseur | CNEMC (China National Environmental Monitoring Centre), historique horaire | `01h_download_czt.py` ; `analysis/p9_3_characterization.csv` |
| Source météo | Open-Meteo Historical Weather, 3 points ville (centroïdes Changsha/Zhuzhou/Xiangtan — 1 point par ville, pas par station) | `PREREGISTRATION_CZT.md` §7 ; `p9_3_characterization.csv` |
| Période | 2020-01-01 → 2023-12-31 (48 mois) | `p9_3_characterization.csv` |
| Couverture PM2.5 brute (avant interpolation) | 89,0% à 98,4% selon la station (aucune proche du seuil de filtre 50%) | `PREREGISTRATION_CZT.md` §7 |
| Taux de manquants brut (moyenne réseau) | 0,0339 (3,39%) | `p9_3_characterization.csv` colonne `raw_missing_rate` |
| PM2.5 moyen / variance | 39,54 / 1043,12 | `p9_3_characterization.csv` |
| Variance train par station (moyenne [min, max]) | 998,76 [852,16, 1105,41] | `p9_3_characterization.csv` — aucun mode variance-train-nulle détecté (pas d'équivalent MENDEZ ALVARO sur ce réseau) |
| Autocorrélation lag-1 | 0,9795 | `p9_3_characterization.csv` |
| R² de persistance | 0,9613 | `p9_3_characterization.csv` |
| r̄ (période train, 70%) | 0,9036 | `p9_3_characterization.csv` — **distinct** de r̄ jeu complet (0,9068, §2 ci-dessus) |
| Densité de graphe réalisée (k=5) | 0,2632 (distance et correlation) | `p9_3_characterization.csv` |
| Degré effectif (k=5) | 5,00 (distance et correlation) — jamais plafonné, 20 stations ≥ k+1 | `p9_3_characterization.csv` |

## 7. Portée du protocole CZT — ce qui n'a jamais été lancé sur ce réseau

Rappel explicite (déjà en légende Table 3/7/8/9, à répéter en §5.9) : CZT n'a
**ni baselines externes** (ARIMA/XGBoost/LSTM/STGCN/Graph WaveNet — E16
limité à GCN-Transformer + Linear-Transformer), **ni contrôle
over-smoothing/GAT** (E13, 3 villes seulement), **ni contrôle diagnostique**
(E4/E5, shuffled-graph/no-meteorology, 3 villes seulement), **ni expérience
de pruning** (Figure 5, déclaré dans la légende), **ni balayage
k-sensitivity** (Table 7, k=5 uniquement). Seul le benchmark canonique
(GCN-Transformer vs Linear-Transformer, 2 topologies, 3 seeds) a été lancé.
