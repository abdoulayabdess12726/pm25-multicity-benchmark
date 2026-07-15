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

```bash
pip install -r requirements.txt

# 1. Data acquisition and preprocessing
python 01a_download_beijing.py            # UCI #501
python 01b_download_london.py             # LAQN
python 01d_download_london_weather.py     # Open-Meteo covariates
python 01c_preprocess_london.py
python 01e_download_madrid.py             # OpenAQ v3
python 01f_download_madrid_weather.py     # Open-Meteo covariates
python 01g_preprocess_madrid.py

# 2. Heterogeneity index (paper Table 1)
python 05_compute_heterogeneity_v2.py

# 3. Full benchmark: 3 cities x 2 topologies x 2 models x 3 seeds (Tables 2, 5)
python 06_train_multistation.py --seeds 42 123 777

# 4. Statistical tests: Wilcoxon + Holm-Bonferroni, bootstrap CIs, Cohen's d (Table 3)
python 07_statistical_analysis.py

# 5. Graph-density sensitivity, k in {3, 5, 8} capped at N-1 (Table 6)
python 08_sensitivity_k.py
```

## Reproducing the paper's tables

| Paper table | Script |
|---|---|
| Table 1 — h(D) components per city | `05_compute_heterogeneity_v2.py` |
| Tables 2, 5 — per-city / per-station benchmark | `06_train_multistation.py` |
| Table 3 — statistical tests | `07_statistical_analysis.py` |
| Table 6 — k-sensitivity | `08_sensitivity_k.py` |
| Table 7 — over-smoothing controls (1-layer GCN, GAT, Dirichlet energy) | `09_controls_oversmoothing.py` |

Full per-station results: [`results/per_station_seed_topology.csv`](results/) (27 stations × 3 seeds × 2 topologies = 162 rows). Adjacency matrices used in the paper: [`graphs/adjacency/`](graphs/) — `{city}_{topology}_k{3|5|8}.npy`.

## Graph construction details

- Neighbours are other stations; self-loops are not counted in neighbour selection.
- k is capped at N−1: nominal k = 8 yields fully connected graphs for London (7 effective neighbours) and Madrid (6).
- Distance topology: inverse Haversine edge weights (min-max normalized). Correlation topology: training-period PM2.5 Pearson correlation (clipped, normalized).

## Reproducibility

- Random seeds: 42 (primary), 123, 777
- Chronological splits; SEQ_LEN = 24 h, horizon = 1 h, BATCH = 64, D_MODEL = 64, MAX_EPOCHS = 50, PATIENCE = 8
- Hardware: Apple M1 (MPS backend), Python 3.13, PyTorch 2.11; full benchmark ≈5.5 h wall-clock

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
