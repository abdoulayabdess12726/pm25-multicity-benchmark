#!/usr/bin/env bash
# E12 (P5) — controles de pruning : aleatoire a densite appariee (>=5
# tirages) + inverse (retire les aretes les plus homophiles d'abord), 3
# villes. Niveaux intermediaires reduits a {0.75, 0.25} (au lieu des 5
# canoniques) pour tenir le budget, consigne explicite : reduire les
# niveaux plutot que les tirages. 1.0/0.0 partages avec le pruning guide
# (identiques par construction, pas reentraines).
#
# Compute : (2 niveaux x 5 tirages + 2 niveaux x 3 seeds) x 3 villes
#         = (10 + 6) x 3 = 48 entrainements.
set -euo pipefail
cd "$(dirname "$0")"

t0=$(date +%s)
for city in beijing london madrid; do
  for level in 0.75 0.25; do
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') $city/keep=$level/random (5 tirages) ==="
    caffeinate -i python3 15_pruning_controls.py --cities "$city" \
      --strategies random --levels "$level" --n_draws 5 --cpu
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') $city/keep=$level/inverse (3 seeds) ==="
    caffeinate -i python3 15_pruning_controls.py --cities "$city" \
      --strategies inverse --levels "$level" --cpu
  done
done
t1=$(date +%s)
echo "E12_TOTAL_DUREE_S=$((t1 - t0)) E12_TOTAL_DUREE_MIN=$(( (t1 - t0) / 60 ))"
