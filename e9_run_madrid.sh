#!/usr/bin/env bash
# E9 (P5) — Madrid, k-sensitivity, configuration canonique 7 stations.
# 12 cellules : {distance,correlation} x {3,8} x {42,123,777}.
# k=5 déjà correct (pull JSON, 7 stations, cf. rapport de tâche P5) — pas relancé.
# --force-retrain : seed=42 ne doit PAS relire l'archive 6-stations (P1).
# caffeinate -i : empêche la machine de dormir (cause du dépassement 3h en E7).
set -euo pipefail
cd "$(dirname "$0")"

t0=$(date +%s)
for topo in distance correlation; do
  for k in 3 8; do
    for seed in 42 123 777; do
      echo "=== $(date '+%Y-%m-%d %H:%M:%S') madrid/$topo/k=$k/seed=$seed ==="
      caffeinate -i python3 e8_k_sensitivity_3seeds.py --city madrid --topology "$topo" \
        --k "$k" --seed "$seed" --cpu --force-retrain
    done
  done
done
t1=$(date +%s)
echo "E9_TOTAL_DUREE_S=$((t1 - t0)) E9_TOTAL_DUREE_MIN=$(( (t1 - t0) / 60 ))"
