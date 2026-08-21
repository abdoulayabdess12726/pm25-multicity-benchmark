# Table 3 — Baselines externes + SOTA (4 décimales)

| City | Model | Topology | MAE | RMSE | R² | Provenance |
|---|---|---|---|---|---|---|
| Beijing | Persistence (t−1) |  | 10.6240 | 21.2405 | 0.9474 ± 0.0000 | deterministic |
| Beijing | ARIMA |  | 10.4071 | 20.3889 | 0.9515 ± 0.0000 | deterministic |
| Beijing | XGBoost |  | 10.9860 | 22.1699 | 0.9427 ± 0.0000 | primary_seed |
| Beijing | LSTM |  | 10.5984 | 21.1720 | 0.9477 ± 0.0001 | 3seed_mean |
| Beijing | GCN-Transformer | distance |  |  | 0.9321 ± 0.0006 | 3seed_mean |
| Beijing | GCN-Transformer | correlation |  |  | 0.9118 ± 0.0009 | 3seed_mean |
| Beijing | Linear-Transformer | n/a (topology-independent) |  |  | 0.9493 ± 0.0007 | 3seed_mean |
| Beijing | STGCN | correlation |  |  | 0.9580 ± 0.0015 | 3seed_mean |
| Beijing | Graph WaveNet | correlation |  |  | 0.9599 ± 0.0018 | 3seed_mean |
| London | Persistence (t−1) |  | 0.9740 | 1.9557 | 0.8286 ± 0.0000 | deterministic |
| London | ARIMA |  | 0.9538 | 1.8944 | 0.8391 ± 0.0000 | deterministic |
| London | XGBoost |  | 1.0701 | 1.9209 | 0.8346 ± 0.0000 | primary_seed |
| London | LSTM |  | 0.9884 | 1.9367 | 0.8319 ± 0.0003 | 3seed_mean |
| London | GCN-Transformer | distance |  |  | 0.4668 ± 0.0247 | 3seed_mean |
| London | GCN-Transformer | correlation |  |  | 0.4408 ± 0.0037 | 3seed_mean |
| London | Linear-Transformer | n/a (topology-independent) |  |  | 0.8422 ± 0.0040 | 3seed_mean |
| London | STGCN | correlation |  |  | 0.8421 ± 0.0055 | 3seed_mean |
| London | Graph WaveNet | correlation |  |  | 0.8425 ± 0.0081 | 3seed_mean |
| Madrid | Persistence (t−1) |  | 2.8753 | 4.8603 | 0.7986 ± 0.0000 [SUSPECT] | deterministic |
| Madrid | ARIMA |  | 2.8527 | 4.7038 | 0.8114 ± 0.0000 [SUSPECT] | deterministic |
| Madrid | XGBoost |  | 3.0147 | 4.7688 | 0.8061 ± 0.0000 [SUSPECT] | primary_seed |
| Madrid | LSTM |  | 2.8801 | 4.8551 | 0.7990 ± 0.0000 [SUSPECT] | 3seed_mean |
| Madrid | GCN-Transformer | distance |  |  | 0.4927 ± 0.0064 | 3seed_mean |
| Madrid | GCN-Transformer | correlation |  |  | 0.3996 ± 0.0335 | 3seed_mean |
| Madrid | Linear-Transformer | n/a (topology-independent) |  |  | 0.8140 ± 0.0013 | 3seed_mean |
| Madrid | STGCN | correlation |  |  | 0.7962 ± 0.0152 | 3seed_mean |
| Madrid | Graph WaveNet | correlation |  |  | 0.8193 ± 0.0010 | 3seed_mean |

_Provenance : 3seed_mean (moyenne±SD ddof=1 sur seeds 42/123/777) ou primary_seed (seed 42 seul, coût de calcul) ou deterministic (ARIMA/Persistence). [SUSPECT] : au moins une ligne source porte un provenance_note (SUSPECT_6STATION, Madrid pré-E1 re-run) — cf. CHANGELOG_TABLES.md._
