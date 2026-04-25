# PM₂.₅ Urban Air Quality Forecasting — Reproducible Benchmark

Supplementary code for:

> **Badouch A., Belhoucine K.** — *Reproducible Benchmark for IoT 
> Urban Air Quality Forecasting: Transformer Superiority and GCN 
> Redundancy Under Spatially Homogeneous Conditions*  
> IJACSA, 2025 (under review)

---

## Results Summary

| Model | MAE | RMSE | R² |
|---|---|---|---|
| ARIMA | 4.17 ±1.2 | 6.51 ±0.8 | 0.973 |
| XGBoost | 6.35 ±0.22 | 11.67 ±0.35 | 0.963 |
| LSTM | 12.00 ±0.10 | 23.81 ±0.48 | 0.847 |
| CNN-LSTM | 21.10 ±4.82 | 33.04 ±5.37 | 0.698 |
| **GCN+Transformer** | **8.17 ±1.00** | **15.07 ±0.67** | **0.939** |

## Dataset

Beijing Multi-site Air Quality Dataset — UCI #501  
→ https://archive.ics.uci.edu/dataset/501  
Download and place CSV files in `beijing+multi+site+air+quality+data/`

## Usage

```bash
pip install -r requirements.txt

# 1. Benchmark all models (Table III)
python 02_train_all_models.py

# 2. Per-station Dongsi (Table IV)  
python 04_dongsi_experiment.py

# 3. Component ablation (Table V)
python 03_ablation_real.py

# 4. Multi-node topology ablation (Table VI)
python solution_A_multinode_gcn.py \
  --data_dir beijing+multi+site+air+quality+data/PRSA_Data_20130301-20170228
```

## Hardware

Apple M1 (MPS) — PyTorch 2.11 — Python 3.13

## Seeds

All experiments use seeds {42, 123, 777}.  
Results reported as mean ± SD across 3 seeds.