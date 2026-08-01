# SOTA GNN spatiotemporelles — STGCN + Graph WaveNet (R1.11 / R2.1-2)

2 modèles × 3 villes × seed 42 (6 entraînements). Protocole identique à
`06_train_multistation.py` (splits 70/15/15, 5 features, SEQ_LEN=24, horizon 1h,
MinMax fit train, R² agrégé dénormalisé, device=cpu). Adjacence prédéfinie =
topologie CORRÉLATION du papier principal (`build_correlation_graph`, k=5).

**Madrid = 7 stations, MENDEZ ALVARO INCLUSE** (manuscrit §3.3 : exclue
seulement de l'analyse station-level §5.5, pas du benchmark de prévision —
contrairement aux scripts E6/E8 de k-sensitivity qui l'excluaient). Vérifié :
n_nodes=7 sur les 3 runs Madrid ; le graphe de corrélation l'isole
naturellement (PM2.5 constant sur train → corrélation indéfinie → 0 arête),
comportement déjà présent dans le GCN-Transformer canonique de la Table 3
originale (Linear-Transformer R2=0.814 sur 7 stations, seed-mean 3 seeds,
reconstruit exactement ΔR² Madrid/distance = −0.321 du papier).

## Table 3 — extension SOTA

| City | Model | R² | Linear-Transformer ref (seed 42) | Bat Linear ? |
|---|---|---|---|---|
| Beijing | STGCN | 0.9598 | 0.9495 | Oui (+0.0103) |
| Beijing | Graph WaveNet | 0.9609 | 0.9495 | Oui (+0.0114) |
| London | STGCN | 0.8479 | 0.8451 | Oui (+0.0028, marginal) |
| London | Graph WaveNet | 0.8396 | 0.8451 | Non (−0.0055) |
| Madrid | STGCN | 0.7882 | 0.8135 | Non (−0.0253) |
| Madrid | Graph WaveNet | 0.8184 | 0.8135 | Oui (+0.0049) |

## Constat scientifique (règle d'arrêt déclenchée à 2 reprises)

Contrairement au GCN-Transformer du papier (ΔR² ≈ −0.36 à −0.40 sur
London/Madrid), STGCN et Graph WaveNet restent compétitifs voire
légèrement supérieurs au Linear-Transformer y compris en régime hétérophile :
STGCN bat le Linear à London (+0.0028) et Graph WaveNet le bat à Madrid
(+0.0049). Aucun des deux ne s'effondre comme le GCN-Transformer. Chiffres
bruts, non lissés — à traiter explicitement dans la réponse aux reviewers :
la dégradation en régime hétérophile documentée dans le papier semble
spécifique à l'architecture GCN-Transformer testée (agrégation spectrale
simple, 2 couches), pas généralisable à toute architecture spatiotemporelle
graphique — les architectures SOTA testées ici combinent gating temporel
(GLU/WaveNet), convolution spectrale/diffusion multi-support, et pour Graph
WaveNet une adjacence adaptative apprise qui peut compenser une topologie
prédéfinie sous-optimale.

## Temps d'exécution (device=cpu, confirmé sur les 6 runs)

| Run | Durée | RSS |
|---|---|---|
| Beijing / STGCN (GATE 1) | 13.1 min | 0.57 GB |
| London / STGCN | 16.4 min | 0.58 GB |
| Madrid / STGCN | 30.3 min | 0.51 GB |
| Beijing / Graph WaveNet (GATE 2, retry avec `caffeinate`) | 67.9 min | 0.62 GB |
| London / Graph WaveNet | 14.3 min | 0.61 GB |
| Madrid / Graph WaveNet | 42.5 min | 0.55 GB |
| **Total (calcul productif)** | **184.6 min (≈3h05)** | — |

Note : la première tentative Beijing/Graph WaveNet (Gate 2) a été tuée après
11h47 d'horloge (>> limite 3h) sans atteindre le premier checkpoint — la
machine s'est mise en veille en l'absence de `caffeinate` (≈96 min de temps
CPU réel accumulées sur ces 11h47, donc la majeure partie était de la veille,
pas du calcul). Toutes les tentatives suivantes ont été enveloppées dans
`caffeinate -i` et ont terminé dans les temps attendus.

## Implémentations — sources citées

- **STGCN** : Yu, B., Yin, H., & Zhu, Z. (2018). *Spatio-Temporal Graph
  Convolutional Networks: A Deep Learning Framework for Traffic Forecasting*.
  IJCAI 2018. Convolution spectrale de Chebyshev : Defferrard, M., Bresson, X.,
  & Vandergheynst, P. (2016). *Convolutional Neural Networks on Graphs with
  Fast Localized Spectral Filtering*. NeurIPS 2016.
- **Graph WaveNet** : Wu, Z., Pan, S., Long, G., Jiang, J., & Zhang, C. (2019).
  *Graph WaveNet for Deep Spatial-Temporal Graph Modeling*. IJCAI 2019.
  Convolution de diffusion : Li, Y., Yu, R., Shahabi, C., & Liu, Y. (2018).
  *Diffusion Convolutional Recurrent Neural Network*. ICLR 2018.
- Seule adaptation vs. papiers originaux : sortie horizon=1 (notre tâche) au
  lieu du seq2seq multi-step natif — gating, dilatations, conv
  spectrale/diffusion et adjacence adaptative sont fidèles aux papiers
  (implémentation : `14_sota_baselines.py`).
