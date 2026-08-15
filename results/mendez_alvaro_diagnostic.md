# Diagnostic MENDEZ ALVARO (Madrid) — P1

Décision d'agrégation : **incluse partout, sans traitement particulier**
(voir `REVISION_BRIEF.md`, section « Décision d'agrégation MENDEZ ALVARO »).
Ce document consigne les chiffres qui ont précédé la décision.

## Variance train vs test (dénormalisée)

| Split | n | mean | std | min | max |
|---|---|---|---|---|---|
| Train | 24544 | 6.0000 | **0.0000** | 6.00 | 6.00 |
| Val | 5260 | 5.9204 | 1.0868 | 1.00 | 30.00 |
| Test | 5260 | 11.0068 | **9.9281** | 1.00 | 49.00 |

PM2.5 parfaitement constant sur tout le train (probable capteur bloqué),
recalibré quelque part entre train et val, régime test normal. Ce n'est pas
une simple "variance nulle" locale : c'est un changement de régime complet
entre les splits.

## Unicité du cas (3 villes, 27 stations)

Le filtre qualité (`01g_preprocess_madrid.py:127-138`, et équivalent pour
Beijing/London) ne teste que `test_std < 0.5` — jamais la variance train.
Vérification systématique sur les 27 stations (12 Beijing + 8 London + 7
Madrid) : **MENDEZ ALVARO est la seule** avec train_std < 0.5 et test_std ≥
0.5. Train_std minimum observé ailleurs : 5.84 (BL0, Londres). Pas un mode
de défaillance systématique du filtre — un cas isolé, filtre non modifié.

## Prédictions par modèle (station seule, seed 42 pour LSTM/XGBoost)

| Modèle | MAE | RMSE | R² (station seule) | Mécanisme |
|---|---|---|---|---|
| Persistence | 2.856 | 4.693 | 0.777 | PM2.5[t] = PM2.5[t−1] réel |
| ARIMA(2,1,2) | 2.856 | 4.693 | 0.777 | état ré-alimenté par val/test réels (`.append(refit=False)`), converge vers la persistance |
| LSTM (pooled 7 stations) | 2.865 | 4.637 | 0.783 | skip-persistance (`pred = PM2.5[t−1] + correction`) ; correction bien calibrée même en extrapolation |
| XGBoost | 7.215 | 11.134 | **−0.254** | prédit une constante 6.0 partout (train : lags ET cible constants → un seul leaf, aucune capacité d'extrapolation) |

## Impact sur les agrégats Madrid de la Table 3 (7 stations, MENDEZ incluse)

Reconstruit sans ré-entraînement pour les 6 autres stations (SS_res = Σ
RMSE²·n depuis `results/external_baselines.csv` existant + le calcul MENDEZ
ci-dessus). Voir `CHANGELOG_TABLES.md` pour le détail avant/après et la
cause de chaque variation.

| Modèle | R² (6 stations, soumis) | R² (7 stations, corrigé) | Δ |
|---|---|---|---|
| Linear-Transformer | 0.817 | 0.8140 | −0.0030 |
| GCN-Transformer | 0.472 | 0.4927 | +0.0207 |
| Persistence | 0.7986 | 0.7961 | −0.0025 |
| ARIMA | 0.8114 | 0.8073 | −0.0041 |
| XGBoost | 0.8061 | **0.6758** | **−0.1304** |
| LSTM (seed 42) | 0.7990 | 0.7971 | −0.0019 |
