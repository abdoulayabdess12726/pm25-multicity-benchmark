# Spatial Graph Encoding for AI-Based PM2.5 Forecasting in IoT Smart Cities — Three-City Benchmark

Supplementary code and data for:

> **A. Badouch and K. Belhoucine**, "Spatial Graph Encoding for AI-Based PM2.5 Forecasting in IoT Smart Cities," *International Journal of Intelligent Engineering and Systems* (IJIES), under review (Paper ID 20264131).

Companion single-city (Beijing) study: Badouch & Krit, *IJACSA*, 2026, [DOI 10.14569/IJACSA.2026.0170595](https://doi.org/10.14569/IJACSA.2026.0170595) — code at [`pm25-beijing-benchmark`](https://github.com/abdoulayabdess12726/pm25-beijing-benchmark).

## What this repository contains

A **reproducible three-city benchmark** (Beijing, London, Madrid) comparing a two-layer **GCN-Transformer** against a temporal-only **Linear-Transformer** for 1-hour-ahead PM2.5 forecasting, together with a composite **spatial heterogeneity index h(D)**. The central finding is a *negative result*: the GCN-Transformer underperforms the Linear-Transformer at 26 of 27 stations (distance topology, primary seed; 53 of 54 station–topology pairs), and the magnitude of underperformance is broadly associated with network heterogeneity.

## Key results (paper Tables 1 and 3)

Spatial heterogeneity index:

| City | Stations | h(D) | Regime |
|---|---|---|---|
| Beijing | 12 | 0.497 | homogeneous (basin-bound) |
| London | 8 | 0.656 | moderately heterogeneous |
| Madrid | 7 | 0.728 | highly heterogeneous (traffic-dominated) |

Aggregate ΔR² = GCN − Linear (3 seeds); Wilcoxon p Holm–Bonferroni-corrected; Cohen's d on per-station differences:

| City | Topology | ΔR² (3 seeds) | p (Holm) | d |
|---|---|---|---|---|
| Beijing | Distance | −0.017 ± 0.0001 | 0.0024 | −1.02 |
| Beijing | Correlation | −0.038 ± 0.0007 | 0.0015 | −1.15 |
| London | Distance | −0.375 ± 0.021 | 0.0156 | −1.17 |
| London | Correlation | −0.401 ± 0.005 | 0.0117 | −1.26 |
| Madrid | Distance | −0.321 ± 0.005 | 0.0156 | −2.42 |
| Madrid | Correlation | −0.380 ± 0.014 | 0.0078 | −2.17 |

All six per-city tests are significant at the corrected 0.05 level; the aggregate gap is ≈18.9× (distance) and ≈10.0× (correlation) larger in Madrid than in Beijing.

## Datasets

- **Beijing**: UCI Multi-Site Air-Quality Data Set (#501), 12 stations, 2013–2017
- **London**: London Air Quality Network (LAQN) + Open-Meteo Historical Weather, 8 stations after quality filtering, 2020–2023
- **Madrid**: OpenAQ API v3 + Open-Meteo Historical Weather, 7 stations after quality filtering

See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) for sources, licenses, and access details.

## Pipeline

Two tiers, depending on what you want to reproduce:

- **Tables and figures from the already-collected results** (fast, a couple
  of minutes, no training): `results/raw_results.csv` is the single source
  of truth for every number in the manuscript. If you only need to verify
  that the reported tables/figures follow from it, skip straight to
  [Regenerating tables and figures](#regenerating-tables-and-figures) below.
- **Every experiment from raw data** (slow, real wall-clock time on the
  order of days serially — see [Reproducing all experiments from
  scratch](#reproducing-all-experiments-from-scratch) for measured
  per-experiment durations): follow this section in full.

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt   # exact pinned versions: see Reproducibility below

# 1. Data acquisition and preprocessing (4 networks: Beijing, London, Madrid,
#    Chang-Zhu-Tan)
python 01a_download_beijing.py            # UCI #501
python 01b_download_london.py             # LAQN
python 01d_download_london_weather.py     # Open-Meteo covariates
python 01c_preprocess_london.py
python 01e_download_madrid.py             # OpenAQ v3
python 01f_download_madrid_weather.py     # Open-Meteo covariates
python 01g_preprocess_madrid.py
python 01h_download_czt.py                # CNEMC 2020-2023, 20 stations (~93 min, one-time)
python 01i_preprocess_czt.py              # coverage filter S3.2 + Open-Meteo join

# 2. Heterogeneity index h(D) (Table 1)
python 05_compute_heterogeneity_v2.py

# 3. Canonical benchmark: one call per city (seeds 42/123/777 are hardcoded
#    module-level constants, not a CLI flag), 2 topologies x 2 models each (Table 2)
python 06_train_multistation.py --city beijing
python 06_train_multistation.py --city london
python 06_train_multistation.py --city madrid
python 06_train_multistation.py --city czt

# 4. Every other reported experiment (external baselines, SOTA baselines,
#    k-sensitivity, pruning, diagnostics, over-smoothing/GAT controls, ...):
#    configs/experiments/*.yaml documents the exact CLI invocation, seeds,
#    and hyperparameters actually used for each one, cross-checked against
#    raw_results.csv. Run the `cli_invocation:` field of each file, or the
#    shell wrapper it points to (e9_run_madrid.sh, e13_run_oversmoothing.sh, ...).
ls configs/experiments/
```

Every run appends to `results/raw_results.csv` via `src/results_io.append_run`
(append-only — existing rows are never rewritten; a corrected rerun for a
condition that already has a flawed row, e.g. a pre-fix run, lands as new
rows under a new `run_id` and is reconciled at table-generation time, see
`scripts/regenerate_tables.py::resolve_superseded_suspect_rows`).

### Regenerating tables and figures

Once `raw_results.csv` is populated (either from this repo's own history, or
from your own reruns above), every table and figure is a pure function of
that one file — no number is ever hand-typed into the manuscript:

```bash
python3 scripts/regenerate_tables.py    # Tables 1-9 -> manuscript/tables/*.{md,docx}
python3 regenerate_figures.py           # Figures 1-5 -> figures/*.{pdf,svg}
```

`regenerate_tables.py` ends with a block of blocking assertions (structural
uniqueness of run conditions, Table 7(k=5) == Table 4, pruning anchor ==
Table 2, single Linear-Transformer reference per city/topology, effective
degree capping, etc.) — it exits non-zero if any assertion fails. As of the
current `HEAD`, both scripts run clean from a fresh `git clone` and a venv
resolved to the exact pinned versions below: **25/25 assertions PASS, 0
FAIL, 0 MISSING DATA**, all 5 figures render without error.

## Reproducing the paper's tables

**Numbering below is for Paper ID 20265149 (current revision cycle,
confirmed against the submitted manuscript).** It differs from the table
numbers used in this repo's history up to and including Paper ID 20264131
— see the correspondence table further down if you are cross-referencing
older commits, issues, or notes.

| Paper table | Content | Script |
|---|---|---|
| Table 1 | h(D) components per city | `05_compute_heterogeneity_v2.py` |
| Table 2 | Per-city benchmark (Linear/GCN-Transformer, 3 seeds) | `06_train_multistation.py` |
| Table 3 | External baselines (ARIMA, XGBoost, LSTM, Persistence, STGCN, Graph WaveNet) | `10_external_baselines.py`, `14_sota_baselines.py` |
| Table 4 | Statistical tests (Wilcoxon, Holm-Bonferroni, bootstrap CI, Cohen's d) | `07_statistical_analysis.py` |
| Table 5 | Cross-city Spearman correlation, h(D) vs ΔR² | `07_statistical_analysis.py` |
| Table 6 | ΔR² per station | `results/export_per_station.py` → `results/per_station_seed_topology.csv` |
| Table 7 | k-sensitivity (k ∈ {3,5,8}, capped at N−1) | `e8_k_sensitivity_3seeds.py` (canonical, full 3-seed protocol — `08_sensitivity_k.py` is an earlier, abandoned attempt under a reduced `--quick` schedule that produced a Beijing-k=3 anomaly later shown to be a schedule artifact, see `results/sensitivity_k_canonical_NOTE.md`; not used for any reported number) |
| Table 8 | Over-smoothing controls (1-layer GCN, GAT, Dirichlet energy) | `09_controls_oversmoothing.py` |
| Table 9 | Diagnostic controls (shuffled-graph, no-meteorology ablation) | `11_diagnostics.py` |

### Reproducing all experiments from scratch

`configs/experiments/*.yaml` is the source of truth for the exact CLI
invocation, seeds, and hyperparameters behind every reported experiment.
Timings below are **measured wall-clock**, not estimates — from this
project's own run logs (`results/e*.log`, `logs/*.log`), Apple M1 (CPU
backend forced via `--cpu` for every experiment run after the original
canonical benchmark, for cross-machine determinism; the canonical benchmark
itself used MPS). A full from-scratch reproduction run serially is on the
order of a week of wall-clock time — run experiments in the background and
plan accordingly; no experiment depends on another's raw data, only on the
preprocessed per-city datasets from step 1 above.

| Experiment | Config | Measured duration |
|---|---|---|
| CZT data download (one-time) | — (`01h_download_czt.py`) | 93 min |
| Canonical benchmark, Beijing | `canonical_benchmark.yaml` | 2h 16m |
| Canonical benchmark, London | `canonical_benchmark.yaml` | 2h 21m |
| Canonical benchmark, Madrid | `canonical_benchmark.yaml` | 50 min |
| Canonical benchmark, CZT (E16) | `e16_czt_validation.yaml` | 6h 13m (372.6 min) |
| External baselines (E1), Madrid only, measured on rerun | `external_baselines.yaml` | 51 min (7 stations; Beijing/London not individually logged, scale roughly with station count) |
| k-sensitivity, general 3-city grid | `k_sensitivity_table7.yaml` | 45h 5m |
| E9 — k-sensitivity, Madrid (MENDEZ ALVARO fix) | `e9_k_sensitivity_madrid.yaml` | 13h 40m |
| E10 — edge pruning, Madrid | `e10_edge_pruning_madrid.yaml` | 14h 3m |
| E11 — STGCN/Graph WaveNet seed parity | `sota_stgcn_graphwavenet.yaml` | 10h 22m |
| E12 — pruning controls (random-matched, inverse) | `e12_pruning_controls.yaml` | 52h 52m |
| E13 — over-smoothing / GAT controls | `e13_oversmoothing_gat_control.yaml` | 30h 51m |
| E14 — edge editing, Beijing/London | `e14_edge_editing_beijing_london.yaml` | 32h 48m |
| E17 — correlation-graph NaN-sort bugfix rerun | — (not a distinct experiment, see `configs/experiments/*.yaml` notes) | 9h 7m |

`regenerate_tables.py` and `regenerate_figures.py` themselves run in
seconds to low minutes once `raw_results.csv` exists — see [Regenerating
tables and figures](#regenerating-tables-and-figures) above.

### Correspondence with the previous numbering (Paper ID 20264131)

Table 3 (external baselines) and Table 9 (diagnostic controls) were added
during this revision cycle in response to reviewers and did not exist as
numbered tables before; everything from the old Table 3 onward shifts by
+1. Table 5 (cross-city Spearman correlation) is the new number for
content that existed in the prior manuscript but wasn't separately
reproducible from this repo's README at the time.

| Old table (20264131) | New table (20265149) |
|---|---|
| Table 1 | Table 1 |
| Table 2 | Table 2 |
| — (new) | **Table 3** |
| Table 3 | Table 4 |
| — (new / split out) | **Table 5** |
| Table 5 | Table 6 |
| Table 6 | Table 7 |
| Table 7 | Table 8 |
| — (new) | **Table 9** |

Full per-station results: [`results/per_station_seed_topology.csv`](results/) (27 stations × 3 seeds × 2 topologies = 162 rows). Adjacency matrices used in the paper: [`graphs/adjacency/`](graphs/) — `{city}_{topology}_k{3|5|8}.npy`.

## Graph construction details

- Neighbours are other stations; self-loops are not counted in neighbour selection.
- k is capped at N−1: nominal k = 8 yields fully connected graphs for London (7 effective neighbours) and Madrid (6).
- Distance topology: inverse Haversine edge weights (min-max normalized). Correlation topology: training-period PM2.5 Pearson correlation (clipped, normalized).

## Reproducibility

- Random seeds: 42 (primary), 123, 777
- Chronological splits (70/15/15); SEQ_LEN = 24 h, horizon = 1 h, BATCH = 64, D_MODEL = 64, N_HEADS = 4, N_LAYERS = 2, DROPOUT = 0.1, LR = 1e-3, WEIGHT_DECAY = 1e-5, MAX_EPOCHS = 50, PATIENCE = 8, K_NEIGHBORS = 5 (default; k-sensitivity sweeps k ∈ {3,5,8} capped at N−1)
- Hardware: Apple M1. The canonical benchmark (step 3 above) used the MPS backend; every experiment run afterward forces `--cpu` for cross-run determinism (MPS gives small non-reproducible run-to-run variation on this hardware) — see [Reproducing all experiments from scratch](#reproducing-all-experiments-from-scratch) for measured timings under each backend.
- **Pinned environment** (exact versions the working results in this repo were produced with — `pip install -r requirements.txt` resolves loosely-pinned major dependencies; use these exact versions for a bit-for-bit environment match):

  ```
  python==3.13.0
  numpy==2.4.4
  pandas==3.0.2
  scipy==1.17.1
  scikit-learn==1.8.0
  matplotlib==3.10.8
  requests==2.33.1
  pyyaml==6.0.3
  pyarrow==24.0.0
  libpysal==4.14.1
  esda==2.9.0
  statsmodels==0.14.6
  xgboost==3.2.0
  torch==2.11.0
  torch-geometric==2.7.0
  torchvision==0.26.0
  torchaudio==2.11.0
  pymupdf==1.28.2      # PDF->PNG rasterization for figure QA only, not required for regenerate_*.py
  python-docx==1.2.0    # manuscript/tables/*.docx output only
  ```

  Verified 2026-08-24: a clean `git clone` + a venv resolved to exactly
  these versions runs `scripts/regenerate_tables.py` (25/25 assertions
  PASS) and `regenerate_figures.py` (5/5 figures) without modification, and
  the full `pytest` suite (67/67) passes.

## License

Code: MIT. Data: per the licenses of the original providers (see DATA_AVAILABILITY.md).

## Citation

```bibtex
@article{badouch2026multicity,
  author  = {Badouch, Abdessamad and Belhoucine, Kaoutar},
  title   = {Spatial Graph Encoding for {AI}-Based {PM2.5} Forecasting in {IoT} Smart Cities},
  journal = {International Journal of Intelligent Engineering and Systems},
  year    = {2026},
  note    = {Under review, Paper ID 20264131}
}
```
