# P9.3 (R2.7) — Table de caractérisation des 4 réseaux

Aucun entraînement — métriques calculées depuis les données prétraitées (load_*_data) et raw_results.csv (Persistence Beijing/London/Madrid ; recalculée pour CZT, baseline sans apprentissage). Taux de manquants calculé sur les fichiers SOURCE bruts, avant toute interpolation.

| city | period | n_stations | provider | weather_source | pm25_mean | pm25_var | train_var_mean | train_var_min | train_var_max | lag1_autocorr | r2_persistence | raw_missing_rate | n_raw_files_checked | density_distance | density_correlation | degree_eff_distance | degree_eff_correlation | r_bar |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| beijing | 2013-03-01 -> 2017-02-28 (48 mois) | 12 | UCI Multi-Site Air-Quality Data Set (#501) | native (dataset UCI, TEMP/PRES/DEWP/WSPM déjà inclus) | 79.84 | 6552.94 | 6047.81 | 4878.71 | 7121.38 | 0.969 | 0.9474 | 0.0208 | 12 | 0.4545 | 0.4545 | 5.0 | 5.0 | 0.879 |
| london | 2020-01-01 -> 2023-12-31 (48 mois) | 8 | London Air Quality Network (LAQN) | Open-Meteo Historical Weather | 9.86 | 52.6 | 56.62 | 34.14 | 88.41 | 0.9272 | 0.8286 | 0.4207 | 12 | 0.7143 | 0.7143 | 5.0 | 5.0 | 0.5673 |
| madrid | 2020-01-01 -> 2023-12-31 (48 mois) | 7 | OpenAQ API v3 | Open-Meteo Historical Weather | 9.82 | 71.42 | 63.95 | 0.0 | 117.23 | 0.9357 | 0.7986 | 0.5391 | 8 | 0.8333 | 0.7143 | 5.0 | 4.29 | 0.4784 |
| czt | 2020-01-01 -> 2023-12-31 (48 mois) | 20 | CNEMC (historique horaire, cf. 01h_download_czt.py) | Open-Meteo Historical Weather (3 points ville, centroïdes) | 39.54 | 1043.12 | 998.76 | 852.16 | 1105.41 | 0.9795 | 0.9613 | 0.0339 | 20 | 0.2632 | 0.2632 | 5.0 | 5.0 | 0.9036 |