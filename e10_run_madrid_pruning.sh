#!/usr/bin/env bash
# E10 (P5) — Madrid, courbe de pruning, graphe complet 7 stations.
# 15 cellules : 5 niveaux x 3 seeds. Un sous-processus frais par cellule
# (isolation, cf. incident E3 — fuite mémoire MPS sur process long) ;
# --cpu force le device ; caffeinate -i empêche la machine de dormir.
set -euo pipefail
cd "$(dirname "$0")"

t0=$(date +%s)
for level in 1.0 0.75 0.5 0.25 0.0; do
  for seed in 42 123 777; do
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') madrid/keep=$level/seed=$seed ==="
    caffeinate -i python3 13_edge_pruning.py --cities madrid --seeds "$seed" \
      --levels "$level" --cpu
  done
done
t1=$(date +%s)
echo "E10_TOTAL_DUREE_S=$((t1 - t0)) E10_TOTAL_DUREE_MIN=$(( (t1 - t0) / 60 ))"
