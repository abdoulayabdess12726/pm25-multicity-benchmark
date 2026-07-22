# E4/E5 — Contrôles diagnostiques (seed 42)

ΔR² = R²(GCN-Transformer) − R²(Linear-Transformer), agrégat global, protocole 06.
`real` réutilise le seed 42 des JSON canoniques. Agrégat sur toutes les stations.

| Ville | Topologie | ΔR² real | ΔR² shuffled-graph | ΔR² no-meteorology |
|---|---|---|---|---|
| Beijing | distance | −0.017 | −0.036 | −0.026 |
| Beijing | correlation | −0.037 | −0.070 | −0.041 |
| London | distance | −0.360 | −0.370 | −0.329 |
| London | correlation | −0.402 | −0.462 | −0.434 |
| Madrid | distance | −0.328 | −0.415 | −0.248 |
| Madrid | correlation | −0.391 | −0.400 | −0.288 |

## (a) Shuffled-graph (E4) — permutation d'arêtes préservant les degrés

Dans **les 6 conditions**, ΔR²_shuffled ≤ ΔR²_real : un graphe aléatoire de mêmes
degrés est **aussi mauvais ou pire** que le graphe réel. La dégradation du GCN
n'est donc PAS due à un câblage réel *spécifiquement* adverse — la simple
agrégation spatiale sur un graphe dense de cette densité nuit, et un recâblage
aléatoire (qui connecte des stations en moyenne encore moins corrélées) nuit
davantage. Cohérent avec l'Étape 5 (élaguer les arêtes hétérophiles récupère la
performance) : le facteur est l'hétérophilie des arêtes, pas la topologie exacte.

## (b) No-meteorology (E5) — PM2.5 seul (1 feature)

- **Linear** : no-meteo ≈ complet partout (Beijing 0.948/0.949 ; London 0.842 ;
  Madrid 0.813/0.814) → la météo n'apporte quasi rien au modèle temporel.
- **GCN** : le gap **persiste sans météo** dans les 3 villes → le cœur du
  problème est l'agrégation spatiale du PM2.5, pas la météo. Mais dans la ville
  la plus hétérophile (Madrid), retirer la météo **réduit nettement** le gap
  (distance −0.328 → −0.248 ; correlation −0.391 → −0.288) : l'agrégation
  spatiale des covariables météo entre stations dissemblables ajoute du mal.
  Effet mixte/faible à London et Beijing.

## Conclusion factuelle

Ni le câblage spécifique du graphe (shuffled ≈/pire que real) ni la météo
(gap persistant sans elle) n'expliquent seuls la sous-performance du GCN : elle
provient de l'agrégation spatiale elle-même en régime hétérophile. Les deux
contrôles sont cohérents avec le résultat central du papier.
