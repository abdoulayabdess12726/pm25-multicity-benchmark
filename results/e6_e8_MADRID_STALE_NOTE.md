# E6/E8 (Table 7 — sensibilité k) : Madrid k∈{3,8} non utilisable, en attente d'E9

**MISE À JOUR** : les 6 cellules k=5 Madrid (2 topologies × 3 seeds) sont
**réparées** (pull direct depuis le JSON canonique, zéro ré-entraînement —
voir `CHANGELOG_TABLES.md`). Seules les 12 cellules k∈{3,8} (2 topologies ×
2 k × 3 seeds) restent non utilisables ci-dessous.

**Fichiers concernés** : `e6_k_sensitivity.csv`, `e6_k_sensitivity_seed42_only.csv`,
`table7_k_sensitivity.csv` — uniquement les lignes k∈{3,8} pour Madrid.

Ces trois fichiers ont été produits avant le correctif source-unique de
stations (cycle de révision 20265149, cf. `REVISION_BRIEF.md` et
`AUDIT.md` §1). L'ancien code de `e6_k_sensitivity.py` /
`e8_k_sensitivity_3seeds.py` excluait MENDEZ ALVARO (Madrid) de la liste
`kept` **avant** le calcul du R² agrégé, pour les 3 seeds (42, 123, 777) :

- seed 42 : calculé par `e6_k_sensitivity.py`, archivé dans
  `e6_k_sensitivity_seed42_only.csv` — 6 stations.
- seeds 123/777 : calculés par `e8_k_sensitivity_3seeds.py`, qui **relit**
  l'archive seed-42 ci-dessus pour la cellule seed=42 (donc hérite du
  problème) et appliquait la même exclusion `EXCLUDE`/`kept` pour ses
  propres calculs seed 123/777 — 6 stations partout.

Contrairement à `10_external_baselines.py` (Table 3), il n'existe **aucune**
donnée par-station de MENDEZ ALVARO à ré-agréger a posteriori : la boucle
d'agrégation ne calculait le R² que sur les stations `kept`, jamais sur la
station exclue. **Correction impossible sans ré-entraîner** les 18 cellules
Madrid (3 seeds × 2 topologies × k∈{3,5,8}, sachant que k=5 est un pull JSON
gratuit et redevient correct automatiquement une fois le JSON canonique
relu avec la bonne liste de stations — seuls k∈{3,8} × 3 seeds = 6 cellules
nécessitent un vrai ré-entraînement GCN).

Le code de `e6_k_sensitivity.py` et `e8_k_sensitivity_3seeds.py` est
désormais corrigé (passe par `src.stations.load_stations(city, "benchmark")`,
Madrid = 7 stations). Un re-run produira des lignes Madrid correctes ; en
attendant, **ne pas utiliser les valeurs Madrid actuelles de ces trois
fichiers** pour la Table 7 / §6.2.1 du manuscrit. Beijing et London ne sont
pas affectés (aucune exclusion n'y a jamais été appliquée).

Voir `REVISION_BRIEF.md`, expérience E9 (P5) pour le re-run prévu.
