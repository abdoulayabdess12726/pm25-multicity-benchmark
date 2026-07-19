# Baselines externes (E1) — agrégat par ville

Métriques sur le test, PM2.5 dénormalisé, agrégat = R² global (toutes stations × temps),
protocole identique à 06 (splits 70/15/15, 5 features, horizon 1h, MinMax fit train).
Moyenne ± SD sur seeds 42/123/777 (LSTM ; Linear/GCN-Transformer). ARIMA/XGBoost/Persistence déterministes.

## Beijing (12 stations)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Persistence (t−1) | 10.624 ± 0.000 | 21.240 ± 0.000 | 0.947 ± 0.000 |
| ARIMA | 10.407 ± 0.000 | 20.389 ± 0.000 | 0.952 ± 0.000 |
| XGBoost | 10.986 ± 0.000 | 22.170 ± 0.000 | 0.943 ± 0.000 |
| LSTM | 10.598 ± 0.004 | 21.172 ± 0.017 | 0.948 ± 0.000 |
| Linear-Transformer | 11.227 ± 0.417 | 20.855 ± 0.109 | 0.949 ± 0.001 |
| GCN-Transformer | 13.151 ± 0.105 | 24.129 ± 0.081 | 0.932 ± 0.000 |

## London (8 stations)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Persistence (t−1) | 0.974 ± 0.000 | 1.956 ± 0.000 | 0.829 ± 0.000 |
| ARIMA | 0.954 ± 0.000 | 1.894 ± 0.000 | 0.839 ± 0.000 |
| XGBoost | 1.070 ± 0.000 | 1.921 ± 0.000 | 0.835 ± 0.000 |
| LSTM | 0.988 ± 0.008 | 1.937 ± 0.002 | 0.832 ± 0.000 |
| Linear-Transformer | 1.091 ± 0.062 | 1.876 ± 0.019 | 0.842 ± 0.003 |
| GCN-Transformer | 2.298 ± 0.049 | 3.448 ± 0.065 | 0.467 ± 0.020 |

## Madrid — 6 stations (MENDEZ ALVARO exclue)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Persistence (t−1) | 2.875 ± 0.000 | 4.860 ± 0.000 | 0.799 ± 0.000 |
| ARIMA | 2.853 ± 0.000 | 4.704 ± 0.000 | 0.811 ± 0.000 |
| XGBoost | 3.015 ± 0.000 | 4.769 ± 0.000 | 0.806 ± 0.000 |
| LSTM | 2.880 ± 0.000 | 4.855 ± 0.000 | 0.799 ± 0.000 |
| Linear-Transformer | 2.875 ± 0.018 | 4.638 ± 0.013 | 0.817 ± 0.001 |
| GCN-Transformer | 5.011 ± 0.014 | 7.872 ± 0.037 | 0.472 ± 0.005 |

## Notes

- **Persistence (t−1)** : prévision naïve PM2.5[t] = PM2.5[t−1] (calcul direct sur les cibles test, aucun modèle).
- **GCN-Transformer** : topologie distance. **Linear-Transformer** : temporel pur (identique aux 2 topologies).
- **LSTM** : 2 couches, hidden 64, avec skip de persistance (prédit la correction sur PM2.5[t−1]).
- **ARIMA** : (2,1,2) per-station, one-step-ahead sans refit. **XGBoost** : per-station, lags 1–24 + 4 météo à t−1, seed 42.
- **MENDEZ ALVARO (Madrid)** : PM2.5 constant sur le train, h_i indéfini → station exclue de TOUTES les expériences
  (Expérience A et baselines E1). L'agrégat Madrid ci-dessus porte donc sur 6 stations pour tous les modèles.
- Constat E1 : dans les villes hétérogènes (London, Madrid) le GCN-Transformer est battu par TOUS les baselines,
  y compris la persistance triviale, ARIMA et LSTM — l'encodage de graphe spatial dégrade la prévision.