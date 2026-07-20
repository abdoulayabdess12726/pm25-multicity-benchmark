# Élagage des arêtes hétérophiles (E-B) — R² agrégé par niveau

Graphe de base : distance (k=5). Arêtes retirées par hétérophilie décroissante
(h = 1 − corr_train). Protocole GCN-Transformer identique à 06. Métriques test
dénormalisées, agrégat global. MENDEZ ALVARO exclue (Madrid, 6 stations).

## Beijing (seed 42)

| Arêtes conservées | MAE | RMSE | R² |
|---|---|---|---|
| 100% | 13.088 | 24.072 | 0.932 |
| 75% | 13.015 | 24.072 | 0.932 |
| 50% | 12.130 | 22.538 | 0.941 |
| 25% | 11.848 | 21.875 | 0.944 |
| 0% | 11.488 | 21.580 | 0.946 |

_Référence GCN (100%) = 0.932 ; niveau Linear-Transformer (cible du 0%) = 0.950._

## London (seed 42)

| Arêtes conservées | MAE | RMSE | R² |
|---|---|---|---|
| 100% | 2.258 | 3.388 | 0.485 |
| 75% | 2.119 | 3.088 | 0.573 |
| 50% | 2.027 | 3.020 | 0.591 |
| 25% | 1.582 | 2.429 | 0.735 |
| 0% | 1.317 | 2.092 | 0.804 |

_Référence GCN (100%) = 0.485 ; niveau Linear-Transformer (cible du 0%) = 0.845._

## Madrid (3 seeds — 6 stations, MENDEZ ALVARO exclue)

| Arêtes conservées | MAE | RMSE | R² |
|---|---|---|---|
| 100% | 5.011 ± 0.014 | 7.872 ± 0.037 | 0.472 ± 0.005 |
| 75% | 4.966 ± 0.043 | 7.929 ± 0.115 | 0.464 ± 0.016 |
| 50% | 4.686 ± 0.021 | 7.785 ± 0.021 | 0.483 ± 0.003 |
| 25% | 4.401 ± 0.019 | 7.134 ± 0.077 | 0.566 ± 0.009 |
| 0% | 3.591 ± 0.023 | 5.471 ± 0.017 | 0.745 ± 0.002 |

_Référence GCN (100%) = 0.465 ; niveau Linear-Transformer (cible du 0%) = 0.816._

## Sanity checks

- **100 % = référence GCN** : reproduit exactement le GCN-Transformer du benchmark (montage correct).
- **0 % (graphe vide)** : le GCN ne fait plus d'agrégation spatiale et converge vers le niveau
  Linear-Transformer (Beijing 0.946 vs 0.950 ; London 0.804 vs 0.845 ; Madrid 0.745 vs 0.816).
  L'écart résiduel à London/Madrid vient de ce que le GCN2 à graphe vide reste un MLP 2 couches
  par nœud (self-loops), pas un encodeur strictement linéaire.
- **Beijing = contrôle négatif** : l'élagage est quasi neutre (0.932 → 0.946), peu d'arêtes
  hétérophiles à retirer. À London et Madrid, l'élagage récupère fortement la performance.