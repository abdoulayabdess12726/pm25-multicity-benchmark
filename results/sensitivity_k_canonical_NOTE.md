# E3 — k-sensitivity canonique : statut

## Résultat principal — l'anomalie Beijing k=3 disparaît sous le protocole complet

L'ancienne Table 6 avait été produite sous `--quick` (1 seed, 10 epochs, modèle
réduit D_MODEL=32/1 couche, SEQ_LEN=12). Sous ce schedule réduit, Beijing k=3
donnait un ΔR² **positif** (GCN meilleur que Linear), en contradiction avec le
résultat central du papier.

| Beijing k=3 | ancien (`--quick`, 1 seed) | canonique (3 seeds, 50 epochs) |
|---|---|---|
| distance | **+0.2133** | **−0.0134 ± 0.0005** |
| correlation | +0.1471 | *(non recalculé — voir statut)* |

Cause : sous `--quick`, le Linear-Transformer de Beijing s'effondrait
(R² ≈ 0.65 au lieu de 0.949), tandis que le GCN k=3 restait à ≈ 0.87 → ΔR²
faussement positif. Au protocole complet, Linear remonte à 0.949 et Beijing k=3
redevient **légèrement négatif** (−0.013), cohérent avec k=5 (−0.017). L'anomalie
était donc un **artefact du schedule réduit**, pas un effet réel de k.

## Contenu de `sensitivity_k_canonical.csv`

- **k=5, 3 villes, 2 topologies** : valeurs CANONIQUES (3 seeds, protocole complet)
  réutilisées des `results/{city}/multistation_results.json` (le benchmark de
  référence, = Table 3 du papier).
- **Beijing k=3, distance** : recalculé au protocole canonique (3 seeds) → −0.0134.

## Statut : partiel (arrêt volontaire)

La grille complète (k=3 et k=8 pour les 3 villes × 2 topologies × 3 seeds = 33
entraînements restants) n'a PAS été terminée : la machine était en saturation de
swap (fuite mémoire MPS du processus long-lived, ~4 h/entraînement). Le script
`08_sensitivity_k.py` a été corrigé (libération MPS + `gc` entre entraînements,
protocole canonique, k plafonné à N−1, réutilise k=5+Linear des JSON). Pour
compléter la Table 6 : `python 08_sensitivity_k.py --cities beijing london madrid`
(idéalement après un reboot pour repartir d'un état MPS/swap sain).

## Ancienne Table 6 (`--quick`) — pour mémoire

k=3 : Beijing +0.213 / +0.147 ; London −0.325 / −0.349 ; Madrid −0.290 / −0.311.
k=5 : Beijing −0.020 / −0.041 ; London −0.381 / −0.408 ; Madrid −0.343 / −0.376.
k=8 : Beijing −0.011 / −0.047 ; London −0.398 / −0.483 ; Madrid −0.356 / −0.406.
(distance / correlation ; source `results/sensitivity_k/summary.csv`.)
