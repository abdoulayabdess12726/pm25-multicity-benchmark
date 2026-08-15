# Baselines externes (E1) — agrégat par ville

Métriques sur le test, PM2.5 dénormalisé, agrégat = R² global (toutes stations × temps),
protocole identique à 06 (splits 70/15/15, 5 features, horizon 1h, MinMax fit train).
Moyenne ± SD (ddof=1, src.stats.agg_mean_std — cf. REVISION_BRIEF.md) sur seeds 42/123/777
(LSTM ; Linear/GCN-Transformer). ARIMA/XGBoost/Persistence déterministes.

### Beijing — agrégat 12 stations (protocole benchmark) (test, dénormalisé, moyenne ± SD)
| Model | MAE | RMSE | R² |
|---|---|---|---|
| Persistence (t−1) | 10.624 ± 0.000 | 21.240 ± 0.000 | 0.947 ± 0.000 |
| ARIMA | 10.407 ± 0.000 | 20.389 ± 0.000 | 0.952 ± 0.000 |
| XGBoost | 10.986 ± 0.000 | 22.170 ± 0.000 | 0.943 ± 0.000 |
| LSTM | 10.598 ± 0.005 | 21.172 ± 0.020 | 0.948 ± 0.000 |
| Linear-Transformer | 11.227 ± 0.511 | 20.855 ± 0.134 | 0.949 ± 0.001 |
| GCN-Transformer | 13.151 ± 0.128 | 24.129 ± 0.099 | 0.932 ± 0.001 |

_Persistence (t−1) : PM2.5[t]=PM2.5[t−1]. GCN-Transformer: topologie distance. Linear-Transformer: temporel pur (identique aux 2 topologies)._

### London — agrégat 8 stations (protocole benchmark) (test, dénormalisé, moyenne ± SD)
| Model | MAE | RMSE | R² |
|---|---|---|---|
| Persistence (t−1) | 0.974 ± 0.000 | 1.956 ± 0.000 | 0.829 ± 0.000 |
| ARIMA | 0.954 ± 0.000 | 1.894 ± 0.000 | 0.839 ± 0.000 |
| XGBoost | 1.070 ± 0.000 | 1.921 ± 0.000 | 0.835 ± 0.000 |
| LSTM | 0.988 ± 0.010 | 1.937 ± 0.002 | 0.832 ± 0.000 |
| Linear-Transformer | 1.091 ± 0.076 | 1.876 ± 0.024 | 0.842 ± 0.004 |
| GCN-Transformer | 2.298 ± 0.060 | 3.448 ± 0.079 | 0.467 ± 0.025 |

_Persistence (t−1) : PM2.5[t]=PM2.5[t−1]. GCN-Transformer: topologie distance. Linear-Transformer: temporel pur (identique aux 2 topologies)._

### Madrid — agrégat 7 stations (protocole benchmark) (test, dénormalisé, moyenne ± SD)
| Model | MAE | RMSE | R² |
|---|---|---|---|
| Persistence (t−1) | 2.875 ± 0.000 | 4.860 ± 0.000 | 0.799 ± 0.000 |
| ARIMA | 2.853 ± 0.000 | 4.704 ± 0.000 | 0.811 ± 0.000 |
| XGBoost | 3.015 ± 0.000 | 4.769 ± 0.000 | 0.806 ± 0.000 |
| LSTM | 2.880 ± 0.000 | 4.855 ± 0.000 | 0.799 ± 0.000 |
| Linear-Transformer | 2.871 ± 0.021 | 4.620 ± 0.016 | 0.814 ± 0.001 |
| GCN-Transformer | 4.867 ± 0.016 | 7.629 ± 0.048 | 0.493 ± 0.006 |

_Persistence (t−1) : PM2.5[t]=PM2.5[t−1]. GCN-Transformer: topologie distance. Linear-Transformer: temporel pur (identique aux 2 topologies)._

**Provenance mixte tant que les baselines rapides n'ont pas été relancées** : Linear-Transformer et GCN-Transformer ci-dessus sont recalculés sur 7 stations (MENDEZ ALVARO incluse, sans ré-entraînement). Persistence/ARIMA/XGBoost/LSTM proviennent en revanche de `results/external_baselines.csv` tel qu'il a été écrit par l'ancien code (6 stations, MENDEZ ALVARO jamais évaluée) tant que ce fichier n'a pas été régénéré — voir REVISION_BRIEF.md / rapport de tâche P1 (diagnostic MENDEZ ALVARO avant re-run).

## Notes

- **Persistence (t−1)** : prévision naïve PM2.5[t] = PM2.5[t−1] (calcul direct sur les cibles test, aucun modèle).
- **GCN-Transformer** : topologie distance. **Linear-Transformer** : temporel pur (identique aux 2 topologies).
- **LSTM** : 2 couches, hidden 64, avec skip de persistance (prédit la correction sur PM2.5[t−1]).
- **ARIMA** : (2,1,2) per-station, one-step-ahead sans refit. **XGBoost** : per-station, lags 1–24 + 4 météo à t−1, seed 42.
- **MENDEZ ALVARO (Madrid)** : incluse dans le benchmark de prévision (manuscrit §3.3) — voir
  results/mendez_alvaro_diagnostic.md et REVISION_BRIEF.md. Persistence/ARIMA/LSTM peu affectés ;
  XGBoost s'effondre sur cette station (extrapolation hors-distribution), tirant l'agrégat Madrid vers le bas.
