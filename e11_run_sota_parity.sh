#!/usr/bin/env bash
# E11 (P5) — parité de seeds STGCN / Graph WaveNet.
# 12 cellules : {stgcn,graphwavenet} x 3 villes x seeds {123,777}.
# Seed 42 déjà présent (E7, primary seed) — pas relancé.
# --cpu : 14_sota_baselines.py force déjà cpu par défaut (store_true,
# default=True) ; flag passé explicitement pour rester cohérent avec E9/E10.
# caffeinate -i : empêche la machine de dormir pendant le run.
set -euo pipefail
cd "$(dirname "$0")"

t0=$(date +%s)
for model in stgcn graphwavenet; do
  for city in beijing london madrid; do
    for seed in 123 777; do
      echo "=== $(date '+%Y-%m-%d %H:%M:%S') $city/$model/seed=$seed ==="
      caffeinate -i python3 14_sota_baselines.py --city "$city" --model "$model" \
        --seed "$seed" --cpu
    done
  done
done
t1=$(date +%s)
echo "E11_TOTAL_DUREE_S=$((t1 - t0)) E11_TOTAL_DUREE_MIN=$(( (t1 - t0) / 60 ))"
