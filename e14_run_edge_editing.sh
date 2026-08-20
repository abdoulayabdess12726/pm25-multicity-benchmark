#!/usr/bin/env bash
# E14 (P5) — edition d'aretes Beijing/London, seeds 123/777.
# 20 cellules : 2 villes x 5 niveaux x 2 seeds. Complete les 4 MISSING DATA
# de l'ancre de pruning (Beijing/London seeds 123/777) et met la courbe a
# parite de seeds avec Madrid (deja 3 seeds depuis E10, P5).
set -euo pipefail
cd "$(dirname "$0")"

t0=$(date +%s)
for city in beijing london; do
  for level in 1.0 0.75 0.5 0.25 0.0; do
    for seed in 123 777; do
      echo "=== $(date '+%Y-%m-%d %H:%M:%S') $city/keep=$level/seed=$seed ==="
      caffeinate -i python3 13_edge_pruning.py --cities "$city" --seeds "$seed" \
        --levels "$level" --cpu
    done
  done
done
t1=$(date +%s)
echo "E14_TOTAL_DUREE_S=$((t1 - t0)) E14_TOTAL_DUREE_MIN=$(( (t1 - t0) / 60 ))"
