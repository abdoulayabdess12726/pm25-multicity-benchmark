#!/usr/bin/env bash
# Correctif build_correlation_graph (P5, 2026-08-20) — re-run des 15 cellules
# Madrid/correlation affectées par le bug de tri NaN (cf. CHANGELOG_TABLES.md).
# Table 2 k=5 (3 seeds) + Table 7 k=3/k=8 (3 seeds chacun) + Table 3
# STGCN/Graph WaveNet (3 seeds chacun). --force-retrain sur e8 débloque
# aussi le recalcul k=5 (avant : toujours pull JSON, jamais recalculé).
set -euo pipefail
cd "$(dirname "$0")"

t0=$(date +%s)

for k in 5 3 8; do
  for seed in 42 123 777; do
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') madrid/correlation/k=$k/seed=$seed (Table 2/7) ==="
    caffeinate -i python3 e8_k_sensitivity_3seeds.py --city madrid --topology correlation \
      --k "$k" --seed "$seed" --cpu --force-retrain
  done
done

for model in stgcn graphwavenet; do
  for seed in 42 123 777; do
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') madrid/$model/seed=$seed (Table 3) ==="
    caffeinate -i python3 14_sota_baselines.py --city madrid --model "$model" \
      --seed "$seed" --cpu
  done
done

t1=$(date +%s)
echo "CORRFIX_TOTAL_DUREE_S=$((t1 - t0)) CORRFIX_TOTAL_DUREE_MIN=$(( (t1 - t0) / 60 ))"
