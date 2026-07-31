#!/usr/bin/env bash
# E8 — driver : populate les 54 cellules (3 villes x 2 topo x 3 k x 3 seeds) de
# results/e6_k_sensitivity.csv. Seules les 24 cellules k∈{3,8} x seed∈{123,777}
# entraînent réellement un GCN (device=cpu) ; les 30 autres sont des pulls
# instantanés (json k=5, archive seed=42). Chaque cellule = un sous-processus
# Python frais (isole toute fuite mémoire éventuelle d'un run à l'autre).
# Arrêt immédiat (set -e) si une cellule lève une exception (STOP protocole).
set -euo pipefail
cd "$(dirname "$0")"

t0=$(date +%s)
for city in beijing london madrid; do
  for topo in distance correlation; do
    for k in 3 5 8; do
      for seed in 42 123 777; do
        python3 e8_k_sensitivity_3seeds.py --city "$city" --topology "$topo" \
          --k "$k" --seed "$seed" --cpu
      done
    done
  done
done
t1=$(date +%s)
echo "TOTAL_DUREE_S=$((t1 - t0)) TOTAL_DUREE_MIN=$(( (t1 - t0) / 60 ))" >&2

python3 e8_k_sensitivity_3seeds.py --aggregate
