# DÉPRÉCIÉ — sensibilité k, version `--quick` (v1)

**Ne pas charger ces fichiers dans un pipeline de résultats.** Conservés
uniquement pour comparaison archéologique avec les re-runs canoniques.

## Pourquoi ces chiffres sont faux

Ces `.json`/`.txt` (par ville × k) et `summary.csv` ont été produits sous un
schedule d'entraînement **réduit** (`--quick`) : 1 seed, 10 epochs, modèle
réduit (D_MODEL=32, 1 couche au lieu de 2), SEQ_LEN=12 au lieu de 24 — pas le
protocole complet du benchmark principal (50 epochs, D_MODEL=64, 2 couches,
SEQ_LEN=24, 3 seeds).

Sous ce schedule réduit, **Beijing k=3 donnait un ΔR² positif** (GCN meilleur
que Linear, +0.213 en distance / +0.147 en corrélation) — en contradiction
directe avec le résultat central du papier (le GCN sous-performe partout).
Cause identifiée : sous `--quick`, le Linear-Transformer de Beijing
s'effondrait (R² ≈ 0.65 au lieu de 0.949) tandis que le GCN k=3 restait à
≈ 0.87, produisant un ΔR² faussement positif — un artefact du schedule
réduit, pas un effet réel de k. Voir
`../../sensitivity_k_canonical_NOTE.md` pour le détail complet et la
comparaison chiffrée avec le protocole canonique.

## Où trouver les bonnes valeurs

- `../../sensitivity_k_canonical.csv` — protocole canonique (3 seeds, 50
  epochs), partiel (Beijing k=3/5, London/Madrid k=5 seulement — abandonné
  pour saturation swap MPS, jamais terminé).
- `../../e6_k_sensitivity.csv` / `../../table7_k_sensitivity.csv` — grille
  complète (3 villes × 2 topologies × k∈{3,5,8} × 3 seeds), protocole
  canonique. **Madrid actuellement non utilisable** dans ces deux fichiers
  non plus (cause différente : exclusion erronée de MENDEZ ALVARO, voir
  `../e6_e8_MADRID_STALE_NOTE.md` et `REVISION_BRIEF.md`) — en attente du
  re-run E9 (P5).
