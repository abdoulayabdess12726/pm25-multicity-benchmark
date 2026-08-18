#!/usr/bin/env bash
# E13 (P5) — Table 8, over-smoothing/GAT, jamais lancée jusqu'ici.
# 36 runs : 3 villes x 4 variantes (linear1L, gcn1L, gcn2L, gat2L) x 3 seeds,
# topologie distance uniquement (cf. manuscrit, Table 8 caption). Chaque
# invocation --city couvre déjà les 4 variantes x 3 seeds en interne
# (09_controls_oversmoothing.py::main), donc 3 invocations = 36 runs.
# --cpu : ajouté cette session (auparavant device auto MPS/CUDA/CPU, jamais
# testé sur un run long — cf. incident E3/NOTE.md).
set -euo pipefail
cd "$(dirname "$0")"

t0=$(date +%s)
for city in beijing london madrid; do
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') $city/distance (4 variantes x 3 seeds) ==="
  caffeinate -i python3 09_controls_oversmoothing.py --city "$city" \
    --topology distance --cpu
done
t1=$(date +%s)
echo "E13_TOTAL_DUREE_S=$((t1 - t0)) E13_TOTAL_DUREE_MIN=$(( (t1 - t0) / 60 ))"
