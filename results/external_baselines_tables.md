# Baselines externes (E1) — agrégat par ville

Métriques sur le test, PM2.5 dénormalisé, agrégat = R² global (toutes stations × temps),
protocole identique à 06 (splits 70/15/15, 5 features, horizon 1h, MinMax fit train).
Moyenne ± SD sur seeds 42/123/777 (LSTM ; Linear/GCN-Transformer). ARIMA/XGBoost déterministes.

## Beijing (12 stations)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| ARIMA | 10.407 ± 0.000 | 20.389 ± 0.000 | 0.952 ± 0.000 |
| XGBoost | 10.986 ± 0.000 | 22.170 ± 0.000 | 0.943 ± 0.000 |
| LSTM | 10.598 ± 0.004 | 21.172 ± 0.017 | 0.948 ± 0.000 |
| Linear-Transformer | 11.227 ± 0.417 | 20.855 ± 0.109 | 0.949 ± 0.001 |
| GCN-Transformer | 13.151 ± 0.105 | 24.129 ± 0.081 | 0.932 ± 0.000 |

## London (8 stations)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| ARIMA | 0.954 ± 0.000 | 1.894 ± 0.000 | 0.839 ± 0.000 |
| XGBoost | 1.070 ± 0.000 | 1.921 ± 0.000 | 0.835 ± 0.000 |
| LSTM | 0.988 ± 0.008 | 1.937 ± 0.002 | 0.832 ± 0.000 |
| Linear-Transformer | 1.091 ± 0.062 | 1.876 ± 0.019 | 0.842 ± 0.003 |
| GCN-Transformer | 2.298 ± 0.049 | 3.448 ± 0.065 | 0.467 ± 0.020 |

## Madrid (7 stations)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| ARIMA | 2.853 ± 0.000 | 4.702 ± 0.000 | 0.807 ± 0.000 |
| XGBoost | 3.615 ± 0.000 | 6.099 ± 0.000 | 0.676 ± 0.000 |
| LSTM | 2.877 ± 0.000 | 4.832 ± 0.000 | 0.797 ± 0.000 |
| Linear-Transformer | 2.871 ± 0.017 | 4.620 ± 0.013 | 0.814 ± 0.001 |
| GCN-Transformer | 4.867 ± 0.013 | 7.629 ± 0.039 | 0.493 ± 0.005 |

## Notes

- **GCN-Transformer** : topologie distance. **Linear-Transformer** : temporel pur (identique aux 2 topologies).
- **LSTM** : 2 couches, hidden 64, avec skip de persistance (prédit la correction sur PM2.5[t−1]).
  Un LSTM vanilla régresse vers la moyenne et sous-performe la persistance triviale (R² 0.17 vs 0.80 sur Madrid).
- **ARIMA** : (2,1,2) per-station, one-step-ahead sans refit. **XGBoost** : per-station, lags 1–24 + 4 météo à t−1, seed 42.
- Constat E1 : dans les villes hétérogènes (London, Madrid) le GCN-Transformer est battu par TOUS les baselines,
  y compris ARIMA et LSTM — l encodage de graphe spatial dégrade la prévision (résultat central du papier).