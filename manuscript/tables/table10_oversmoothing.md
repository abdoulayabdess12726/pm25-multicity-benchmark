# Table 10 — Over-smoothing / GAT

| City | Linear (1L) | GCN (1L) | GCN (2L) | GAT (2L) |
|---|---|---|---|---|
| Beijing | 0.8790±0.0020 | 0.8852±0.0082 | 0.8825±0.0031 | 0.8619±0.0027 |
| London | 0.6938±0.0202 | 0.2523±0.0122 | 0.1366±0.0077 | -0.0705±0.0220 |
| Madrid | 0.6460±0.0050 | 0.4968±0.0086 | 0.4588±0.0037 | 0.3090±0.0045 |

_Beijing/London/Madrid uniquement (E13) — Chang-Zhu-Tan (CZT) n'a pas de contrôle over-smoothing/GAT : jamais lancé sur ce réseau (E16 : protocole limité à GCN-Transformer + Linear-Transformer), pas une omission. Non iso-capacité avec le GCN-Transformer canonique (Table 4) : cette réimplémentation (09_controls_oversmoothing.py) utilise un FFN Transformer de largeur 128 au lieu de 256, soit 35,6 % de paramètres en moins (dont 89 % imputables au FFN, pas à la profondeur du GCN testée ici) — comparaisons internes à cette table valides, comparaison directe avec Table 4 confondue par cet écart. Cf. CHANGELOG_TABLES.md pour le détail chiffré._
