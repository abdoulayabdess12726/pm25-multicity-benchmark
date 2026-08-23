# P9.2 (R2.4) — Test d'équivalence pratique, marge ±0.02 ΔR²

Marge validée le 2026-08-24, fixée AVANT ce calcul — cf. `PREREGISTRATION_CZT.md` §3 (seuil déjà engagé, ΔR² ≥ −0.02). Différence appariée par station, seed primaire 42, IC bootstrap 10000 tirages, 95%.

| Réseau | Modèle | Topologie | n stations | Diff. moyenne | IC95% bootstrap | Verdict |
|---|---|---|---|---|---|---|
| beijing | GCN-Transformer | distance | 12 | -0.0181 | [-0.0291, -0.0095] | NON CONCLUANT (IC chevauche la frontière de la marge) |
| beijing | GCN-Transformer | correlation | 12 | -0.0406 | [-0.0618, -0.0222] | NON ÉQUIVALENT — dégrade au-delà de la marge |
| london | GCN-Transformer | distance | 8 | -0.7016 | [-1.1422, -0.3357] | NON ÉQUIVALENT — dégrade au-delà de la marge |
| london | GCN-Transformer | correlation | 8 | -0.8088 | [-1.2979, -0.4177] | NON ÉQUIVALENT — dégrade au-delà de la marge |
| madrid | GCN-Transformer | distance | 7 | -0.2909 | [-0.3906, -0.2206] | NON ÉQUIVALENT — dégrade au-delà de la marge |
| madrid | GCN-Transformer | correlation | 7 | -0.3988 | [-0.5364, -0.2671] | NON ÉQUIVALENT — dégrade au-delà de la marge |
| czt | GCN-Transformer | distance | 20 | -0.0216 | [-0.0246, -0.0185] | NON CONCLUANT (IC chevauche la frontière de la marge) |
| czt | GCN-Transformer | correlation | 20 | -0.0309 | [-0.0351, -0.0268] | NON ÉQUIVALENT — dégrade au-delà de la marge |
| beijing | stgcn | correlation | — | — | — | MISSING DATA (seed 42 (aggregate-only, migration E7/P2 historique) — per-station disponible aux seeds ['123', '777'], pas au primaire) |
  - *(hors convention seed primaire, à titre informatif : seed 123, n=12, diff=+0.0070, IC=[+0.0038,+0.0096], ÉQUIVALENT (IC entièrement dans ±0.02))*
  - *(hors convention seed primaire, à titre informatif : seed 777, n=12, diff=+0.0087, IC=[+0.0059,+0.0111], ÉQUIVALENT (IC entièrement dans ±0.02))*
| beijing | graphwavenet | correlation | — | — | — | MISSING DATA (seed 42 (aggregate-only, migration E7/P2 historique) — per-station disponible aux seeds ['123', '777'], pas au primaire) |
  - *(hors convention seed primaire, à titre informatif : seed 123, n=12, diff=+0.0079, IC=[+0.0060,+0.0094], ÉQUIVALENT (IC entièrement dans ±0.02))*
  - *(hors convention seed primaire, à titre informatif : seed 777, n=12, diff=+0.0123, IC=[+0.0106,+0.0138], ÉQUIVALENT (IC entièrement dans ±0.02))*
| london | stgcn | correlation | — | — | — | MISSING DATA (seed 42 (aggregate-only, migration E7/P2 historique) — per-station disponible aux seeds ['123', '777'], pas au primaire) |
  - *(hors convention seed primaire, à titre informatif : seed 123, n=8, diff=-0.0042, IC=[-0.0176,+0.0103], ÉQUIVALENT (IC entièrement dans ±0.02))*
  - *(hors convention seed primaire, à titre informatif : seed 777, n=8, diff=-0.0255, IC=[-0.0486,-0.0059], NON CONCLUANT (IC chevauche la frontière de la marge))*
| london | graphwavenet | correlation | — | — | — | MISSING DATA (seed 42 (aggregate-only, migration E7/P2 historique) — per-station disponible aux seeds ['123', '777'], pas au primaire) |
  - *(hors convention seed primaire, à titre informatif : seed 123, n=8, diff=+0.0218, IC=[+0.0149,+0.0283], NON CONCLUANT (IC chevauche la frontière de la marge))*
  - *(hors convention seed primaire, à titre informatif : seed 777, n=8, diff=-0.0284, IC=[-0.0494,-0.0095], NON CONCLUANT (IC chevauche la frontière de la marge))*
| madrid | stgcn | correlation | 7 | -0.0195 | [-0.0665, +0.0099] | NON CONCLUANT (IC chevauche la frontière de la marge) |
| madrid | graphwavenet | correlation | 7 | +0.0046 | [+0.0015, +0.0077] | ÉQUIVALENT (IC entièrement dans ±0.02) |

**Récapitulatif** : 10 comparaisons testées — 1 équivalentes, 6 dégradent au-delà de la marge, 0 améliorent au-delà de la marge, 3 non conclusives.
