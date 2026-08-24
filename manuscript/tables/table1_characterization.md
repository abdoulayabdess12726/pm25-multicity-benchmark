# Table 1 — Caractérisation des réseaux (R2.7)

| City | Period | Stations | Provider | Weather source | PM2.5 mean | PM2.5 var | Train var/station [min,max] | Lag-1 autocorr | R² persistence | Raw missing rate | Density (distance) | Density (correlation) | Degree eff. (distance) | Degree eff. (correlation) | r̄ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Beijing | 2013-03-01 -> 2017-02-28 (48 mois) | 12 | UCI Multi-Site Air-Quality Data Set (#501) | native (dataset UCI, TEMP/PRES/DEWP/WSPM déjà inclus) | 79.84 | 6552.94 | 6047.81 [4878.71, 7121.38] | 0.9690 | 0.9474 | 0.0208 | 0.4545 | 0.4545 | 5.00 | 5.00 | 0.8790 |
| Chang-Zhu-Tan | 2020-01-01 -> 2023-12-31 (48 mois) | 20 | CNEMC (historique horaire, cf. 01h_download_czt.py) | Open-Meteo Historical Weather (3 points ville, centroïdes) | 39.54 | 1043.12 | 998.76 [852.16, 1105.41] | 0.9795 | 0.9613 | 0.0339 | 0.2632 | 0.2632 | 5.00 | 5.00 | 0.9036 |
| London | 2020-01-01 -> 2023-12-31 (48 mois) | 8 | London Air Quality Network (LAQN) | Open-Meteo Historical Weather | 9.86 | 52.60 | 56.62 [34.14, 88.41] | 0.9272 | 0.8286 | 0.4207 | 0.7143 | 0.7143 | 5.00 | 5.00 | 0.5673 |
| Madrid | 2020-01-01 -> 2023-12-31 (48 mois) | 7 | OpenAQ API v3 | Open-Meteo Historical Weather | 9.82 | 71.42 | 63.95 [0.00, 117.23] | 0.9357 | 0.7986 | 0.5391 | 0.8333 | 0.7143 | 5.00 | 4.29 | 0.4784 |

_Source : analysis/p9_3_characterization.csv (P9.3). Train var/station : variance PM2.5 par station sur la période train (70% initiaux), moyenne puis [min, max] inter-stations — Madrid min=0.00 correspond à MENDEZ ALVARO (PM2.5 constant sur train, cf. REVISION_BRIEF.md). r̄ et densité/degré effectif calculés sur la période train (70% initiaux), à ne pas confondre avec h(D) (jeu complet, Table 2)._
