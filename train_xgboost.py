"""
train_xgboost.py — Version autonome
Génère le dataset synthétique Beijing-style et entraîne XGBoost.
Pas besoin de fichier CSV externe.
"""

import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json, time

# ── 1. Générer le dataset synthétique (seed=42, identique à l'article) ──
np.random.seed(42)
N = 35_064   # enregistrements horaires 2013–2017

t = np.arange(N)

# PM2.5 : saisonnalité annuelle + cycle journalier + pics de pollution + bruit
pm25 = (
    45
    + 30 * np.sin(2 * np.pi * t / (24 * 365))
    + 15 * np.sin(2 * np.pi * t / 24)
    + 10 * np.sin(2 * np.pi * t / (24 * 7))
    + np.random.exponential(scale=20, size=N)
    + np.random.normal(0, 8, size=N)
).clip(0, 500)

# Covariables météo + NO2
no2  = (30 + 0.3 * pm25 + 10 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 8, N)).clip(0, 200)
temp = 13 + 15 * np.sin(2 * np.pi * (t - 24*90) / (24*365)) + np.random.normal(0, 3, N)
pres = 1013 + 8 * np.sin(2 * np.pi * t / (24*365)) + np.random.normal(0, 2, N)
dewp = temp - 8 + np.random.normal(0, 2, N)
wspm = np.abs(3 + np.random.normal(0, 2, N))

X = np.column_stack([no2, temp, pres, dewp, wspm])   # 5 features (PM2.5 = target)
y = pm25

# ── 2. Split chronologique 70 / 15 / 15 ──
n_train = int(0.70 * N)
n_val   = int(0.85 * N)
X_train, y_train = X[:n_train],      y[:n_train]
X_val,   y_val   = X[n_train:n_val], y[n_train:n_val]
X_test,  y_test  = X[n_val:],        y[n_val:]

print(f"Dataset: {N:,} enregistrements")
print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
print()

# ── 3. Entraîner sur 3 seeds pour avoir mean ± SD ──
SEEDS   = [42, 123, 777]
results = {"MAE": [], "RMSE": [], "R2": []}

for seed in SEEDS:
    model = XGBRegressor(
        n_estimators          = 500,
        max_depth             = 6,
        learning_rate         = 0.05,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        reg_lambda            = 1.0,
        early_stopping_rounds = 20,
        eval_metric           = "rmse",
        random_state          = seed,
        device                = "cpu",
        verbosity             = 0,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_pred = model.predict(X_test)
    results["MAE"].append(mean_absolute_error(y_test, y_pred))
    results["RMSE"].append(np.sqrt(mean_squared_error(y_test, y_pred)))
    results["R2"].append(r2_score(y_test, y_pred))
    print(f"  seed={seed} → MAE={results['MAE'][-1]:.2f}  RMSE={results['RMSE'][-1]:.2f}  R²={results['R2'][-1]:.4f}")

# ── 4. Résultats finaux ──
mae_mean  = float(np.mean(results["MAE"]))
mae_std   = float(np.std(results["MAE"]))
rmse_mean = float(np.mean(results["RMSE"]))
rmse_std  = float(np.std(results["RMSE"]))
r2_mean   = float(np.mean(results["R2"]))

print()
print("=" * 45)
print("  XGBoost — Résultats finaux (test set)")
print("=" * 45)
print(f"  MAE  : {mae_mean:.2f} ± {mae_std:.2f}  µg/m³")
print(f"  RMSE : {rmse_mean:.2f} ± {rmse_std:.2f}  µg/m³")
print(f"  R²   : {r2_mean:.4f}")
print("=" * 45)
print()
print("→ Valeur pour Table 7 de l'article :")
print(f"  XGBoost | PM₂.₅ | {mae_mean:.2f} ±{mae_std:.2f} | {rmse_mean:.2f} ±{rmse_std:.2f} | {r2_mean:.4f}")

# ── 5. Sauvegarder ──
output = {
    "model":     "XGBoost",
    "MAE_mean":  round(mae_mean,  2),
    "MAE_std":   round(mae_std,   2),
    "RMSE_mean": round(rmse_mean, 2),
    "RMSE_std":  round(rmse_std,  2),
    "R2_mean":   round(r2_mean,   4),
    "n_test":    len(y_test),
    "seeds":     SEEDS,
}
with open("xgboost_results.json", "w") as f:
    json.dump(output, f, indent=2)

print()
print("✓ Résultats sauvegardés dans xgboost_results.json")