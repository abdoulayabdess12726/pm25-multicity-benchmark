# P9.3 — Modèle mixte étendu, covariables station-level

Covariables ajoutées à h_i (toutes station-level, aucun ajustement réseau-level — 4 réseaux, pas de puissance pour ça) : log(variance PM2.5 train), autocorrélation lag-1, taux de manquants brut.

## Topologie distance (n=46)

**Non ajusté** : β(h_i) = -1.0646, SE=0.5494, IC95%=[-2.1415, +0.0123], p=0.05267

**Ajusté** (+ log_train_var + lag1_autocorr + raw_missing_rate) : β(h_i) = +2.7705, SE=1.0002, IC95%=[+0.8102, +4.7308], p=0.005606

**Corrélation h_i × covariables ajoutées (n=46)** :

  - h_i vs log_train_var : r=-0.843, p=2.04e-13
  - h_i vs lag1_autocorr : r=-0.788, p=8.2e-11
  - h_i vs raw_missing_rate : r=+0.928, p=1.65e-20

**⚠️ LE COEFFICIENT CHANGE DE SIGNE (-1.065 → +2.770) ET DEVIENT PLUS SIGNIFICATIF (p=0.0527 → p=0.00561) EN L'AJUSTANT — CE N'EST PAS UN RENFORCEMENT RÉEL DE L'EFFET, C'EST UNE COLINÉARITÉ SÉVÈRE.** h_i corrèle à r=+0.93 avec le taux de manquants, r=-0.84 avec log(variance train), r=-0.79 avec l'autocorrélation lag-1 — et ces trois covariables varient presque exclusivement ENTRE réseaux, pas AU SEIN d'un même réseau (Beijing/CZT : h_i bas, peu de manquants, forte autocorrélation ; London/Madrid : h_i élevé, beaucoup de manquants, autocorrélation plus faible — le même clivage à 4 points). Avec seulement 4 groupes, ces covariables « station-level » capturent en réalité presque la même variation ENTRE réseaux que h_i lui-même : le modèle ajusté n'isole rien, il redistribue un signal quasi-confondu entre des prédicteurs colinéaires. **CE RÉSULTAT N'EST PAS INTERPRÉTABLE COMME « le coefficient d'hétérophilie survit, renforcé, à l'ajustement » — c'est un artefact statistique, pas une conclusion sur h_i.**

  - log_train_var : β=+0.0651, SE=0.0991, p=0.5113
  - lag1_autocorr : β=+12.3340, SE=2.5525, p=1.351e-06
  - raw_missing_rate : β=-2.3581, SE=0.7558, p=0.001808

## Topologie correlation (n=46)

**Non ajusté** : β(h_i) = -1.0732, SE=0.6276, IC95%=[-2.3034, +0.1569], p=0.08727

**Ajusté** (+ log_train_var + lag1_autocorr + raw_missing_rate) : β(h_i) = +3.8229, SE=0.9922, IC95%=[+1.8783, +5.7675], p=0.0001167

**Corrélation h_i × covariables ajoutées (n=46)** :

  - h_i vs log_train_var : r=-0.843, p=2.04e-13
  - h_i vs lag1_autocorr : r=-0.788, p=8.2e-11
  - h_i vs raw_missing_rate : r=+0.928, p=1.65e-20

**⚠️ LE COEFFICIENT CHANGE DE SIGNE (-1.073 → +3.823) ET DEVIENT PLUS SIGNIFICATIF (p=0.0873 → p=0.000117) EN L'AJUSTANT — CE N'EST PAS UN RENFORCEMENT RÉEL DE L'EFFET, C'EST UNE COLINÉARITÉ SÉVÈRE.** h_i corrèle à r=+0.93 avec le taux de manquants, r=-0.84 avec log(variance train), r=-0.79 avec l'autocorrélation lag-1 — et ces trois covariables varient presque exclusivement ENTRE réseaux, pas AU SEIN d'un même réseau (Beijing/CZT : h_i bas, peu de manquants, forte autocorrélation ; London/Madrid : h_i élevé, beaucoup de manquants, autocorrélation plus faible — le même clivage à 4 points). Avec seulement 4 groupes, ces covariables « station-level » capturent en réalité presque la même variation ENTRE réseaux que h_i lui-même : le modèle ajusté n'isole rien, il redistribue un signal quasi-confondu entre des prédicteurs colinéaires. **CE RÉSULTAT N'EST PAS INTERPRÉTABLE COMME « le coefficient d'hétérophilie survit, renforcé, à l'ajustement » — c'est un artefact statistique, pas une conclusion sur h_i.**

  - log_train_var : β=+0.1057, SE=0.1095, p=0.3347
  - lag1_autocorr : β=+12.6732, SE=2.4656, p=2.746e-07
  - raw_missing_rate : β=-3.4742, SE=0.7073, p=9.026e-07


**Conclusion honnête sur ce point du plan (« le coefficient d'hétérophilie survit-il à l'ajustement ? »)** : NON TESTABLE PROPREMENT avec ces covariables à n=4 réseaux. Les covariables station-level naturelles (variance train, autocorrélation, taux de manquants) se sont révélées quasi-confondues avec l'appartenance réseau elle-même — colinéarité sévère (r jusqu'à 0.93 avec h_i), pas un problème de puissance sur h_i mais un problème d'identification du modèle ajusté. Un ajustement fiable demanderait soit plus de réseaux (variation indépendante entre et au sein des groupes), soit des covariables qui varient vraiment au sein d'un même réseau sans suivre le clivage inter-réseau — non trouvées parmi les candidats naturels ici.

**Rappel à faire figurer dans le manuscrit** : aucun ajustement réseau-level testé — avec n=4 réseaux, un tel ajustement n'a pas de puissance statistique interprétable.