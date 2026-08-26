# P9.2 (R2.4) — Test d'équivalence pratique, marge ±0.02 ΔR²

Marge validée le 2026-08-24, fixée AVANT ce calcul — cf. `PREREGISTRATION_CZT.md` §3 (seuil déjà engagé, ΔR² ≥ −0.02). Différence appariée par station, IC bootstrap 10000 tirages, 95%. **Protocole** : seed primaire 42 par défaut ; repli 2-seeds (123/777, moyenne par station) pour 4 comparaisons Beijing/London STGCN/GraphWaveNet où le seed primaire n'a pas de per-station (migration E7/P2 historique, aggregate-only) — explicitement étiqueté dans la colonne Protocole, jamais mélangé avec le seed primaire dans une même quantité.

| Réseau | Modèle | Topologie | Protocole | n stations | Diff. moyenne | IC95% bootstrap | Verdict |
|---|---|---|---|---|---|---|---|
| beijing | GCN-Transformer | distance | seed 42 (primaire) | 12 | -0.0181 | [-0.0291, -0.0095] | NON CONCLUANT (IC chevauche la frontière de la marge) |
| beijing | GCN-Transformer | correlation | seed 42 (primaire) | 12 | -0.0406 | [-0.0618, -0.0222] | NON ÉQUIVALENT — dégrade au-delà de la marge |
| london | GCN-Transformer | distance | seed 42 (primaire) | 8 | -0.7016 | [-1.1422, -0.3357] | NON ÉQUIVALENT — dégrade au-delà de la marge |
| london | GCN-Transformer | correlation | seed 42 (primaire) | 8 | -0.8088 | [-1.2979, -0.4177] | NON ÉQUIVALENT — dégrade au-delà de la marge |
| madrid | GCN-Transformer | distance | seed 42 (primaire) | 7 | -0.2909 | [-0.3906, -0.2206] | NON ÉQUIVALENT — dégrade au-delà de la marge |
| madrid | GCN-Transformer | correlation | seed 42 (primaire) | 7 | -0.3918 | [-0.5226, -0.2707] | NON ÉQUIVALENT — dégrade au-delà de la marge |
| czt | GCN-Transformer | distance | seed 42 (primaire) | 20 | -0.0216 | [-0.0246, -0.0185] | NON CONCLUANT (IC chevauche la frontière de la marge) |
| czt | GCN-Transformer | correlation | seed 42 (primaire) | 20 | -0.0309 | [-0.0351, -0.0268] | NON ÉQUIVALENT — dégrade au-delà de la marge |
| beijing | stgcn | correlation | 2-seeds (123/777, moy./station) | 12 | +0.0079 | [+0.0049, +0.0104] | ÉQUIVALENT (IC entièrement dans ±0.02) |
| beijing | graphwavenet | correlation | 2-seeds (123/777, moy./station) | 12 | +0.0101 | [+0.0083, +0.0116] | ÉQUIVALENT (IC entièrement dans ±0.02) |
| london | stgcn | correlation | 2-seeds (123/777, moy./station) | 8 | -0.0148 | [-0.0310, +0.0010] | NON CONCLUANT (IC chevauche la frontière de la marge) |
| london | graphwavenet | correlation | 2-seeds (123/777, moy./station) | 8 | -0.0033 | [-0.0130, +0.0049] | ÉQUIVALENT (IC entièrement dans ±0.02) |
| madrid | stgcn | correlation | seed 42 (primaire) | 7 | -0.0195 | [-0.0665, +0.0099] | NON CONCLUANT (IC chevauche la frontière de la marge) |
| madrid | graphwavenet | correlation | seed 42 (primaire) | 7 | +0.0046 | [+0.0015, +0.0077] | ÉQUIVALENT (IC entièrement dans ±0.02) |

**Récapitulatif** : 14 comparaisons testées — 4 équivalentes, 6 dégradent au-delà de la marge, 0 améliorent au-delà de la marge, 4 non conclusives.
