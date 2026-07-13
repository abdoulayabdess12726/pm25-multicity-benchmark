"""Statistical analysis v2 of multi-station benchmark results.
Computes:
- Wilcoxon signed-rank test (Linear vs GCN, per-station, per-city)
- Bootstrap 95% CI on Delta R-squared aggregates
- Cohen's d effect size
- Holm-Bonferroni correction across all comparisons (now 6 with 3 cities)
- Spearman correlation between h(D) and Delta R-squared (across cities)
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

OUT = Path("results/statistical_analysis")
OUT.mkdir(parents=True, exist_ok=True)

# Heterogeneity index from 05_compute_heterogeneity_v2
HETEROGENEITY_INDEX = {
    "beijing": 0.497,
    "london":  0.656,
    "madrid":  0.728,
}

def bootstrap_ci(data, n_bootstrap=10000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    data = np.asarray(data)
    boot_means = np.array([
        rng.choice(data, size=len(data), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    alpha = (1 - ci) / 2
    return np.percentile(boot_means, [100*alpha, 100*(1-alpha)])

def cohens_d(a, b):
    diff = np.array(a) - np.array(b)
    return diff.mean() / diff.std() if diff.std() > 0 else 0.0

def analyze_city(json_path, city_name):
    print(f"\n{'='*70}")
    print(f"=== Statistical analysis: {city_name.upper()}")
    print(f"{'='*70}")
    
    with open(json_path) as f:
        data = json.load(f)
    
    n_stations = data["n_stations"]
    print(f"Stations: {n_stations}")
    
    results = {"city": city_name, "n_stations": n_stations,
               "h_index": HETEROGENEITY_INDEX.get(city_name, np.nan),
               "tests": []}
    
    for topology in ["distance", "correlation"]:
        gcn = data["graphs"][topology]["GCN+Transformer"]
        lin = data["graphs"][topology]["Linear+Transformer"]
        
        # Per-station metrics (seed 42)
        per_st_gcn = data["graphs"][topology]["per_station"]["GCN+Transformer"]
        per_st_lin = data["graphs"][topology]["per_station"]["Linear+Transformer"]
        
        gcn_r2_per_station = [per_st_gcn[s]["R2"] for s in per_st_gcn]
        lin_r2_per_station = [per_st_lin[s]["R2"] for s in per_st_lin]
        gcn_rmse_per_station = [per_st_gcn[s]["RMSE"] for s in per_st_gcn]
        lin_rmse_per_station = [per_st_lin[s]["RMSE"] for s in per_st_lin]
        
        # Wilcoxon signed-rank test (one-sided: GCN < Linear in R2)
        wilcoxon_r2 = stats.wilcoxon(gcn_r2_per_station, lin_r2_per_station,
                                      alternative='less')
        wilcoxon_rmse = stats.wilcoxon(gcn_rmse_per_station, lin_rmse_per_station,
                                        alternative='greater')
        
        # Bootstrap 95% CI on Delta R-squared
        delta_r2_per_st = np.array(gcn_r2_per_station) - np.array(lin_r2_per_station)
        ci_lo, ci_hi = bootstrap_ci(delta_r2_per_st)
        
        # Cohen's d
        d = cohens_d(gcn_r2_per_station, lin_r2_per_station)
        
        # Aggregate Delta R-squared (across 3 seeds)
        delta_r2_seeds = np.array(gcn["R2"]) - np.array(lin["R2"])
        
        result = {
            "topology": topology,
            "delta_r2_aggregate_mean": float(delta_r2_seeds.mean()),  # the value reported in the paper
            "delta_r2_aggregate_std": float(delta_r2_seeds.std()),
            "delta_r2_perstation_mean": float(delta_r2_per_st.mean()),
            "delta_r2_perstation_ci95_lo": float(ci_lo),
            "delta_r2_perstation_ci95_hi": float(ci_hi),
            "wilcoxon_r2_pvalue": float(wilcoxon_r2.pvalue),
            "wilcoxon_r2_stat": float(wilcoxon_r2.statistic),
            "wilcoxon_rmse_pvalue": float(wilcoxon_rmse.pvalue),
            "cohens_d": float(d),
            "n_stations_gcn_worse": int((np.array(gcn_r2_per_station) < np.array(lin_r2_per_station)).sum()),
            "n_stations_total": len(gcn_r2_per_station),
        }
        results["tests"].append(result)
        
        print(f"\n--- Topology: {topology} ---")
        print(f"  Delta R-squared (aggregate, 3 seeds mean): {result['delta_r2_aggregate_mean']:+.4f} +/- {result['delta_r2_aggregate_std']:.4f}")
        print(f"  Delta R-squared (per-station mean): {result['delta_r2_perstation_mean']:+.4f}")
        print(f"  95% CI (bootstrap): [{ci_lo:+.4f}, {ci_hi:+.4f}]")
        print(f"  Wilcoxon (R-squared, GCN<Linear): W={result['wilcoxon_r2_stat']:.1f}, p={result['wilcoxon_r2_pvalue']:.4e}")
        print(f"  Wilcoxon (RMSE, GCN>Linear): p={result['wilcoxon_rmse_pvalue']:.4e}")
        print(f"  Cohen's d: {d:+.3f} ({'large' if abs(d)>0.8 else 'medium' if abs(d)>0.5 else 'small'})")
        print(f"  GCN worse than Linear: {result['n_stations_gcn_worse']}/{result['n_stations_total']} stations")
    
    return results

# Run analysis on all 3 cities
results_all = []
city_paths = [
    ("beijing", "results/beijing/multistation_results.json"),
    ("london",  "results/london/multistation_results.json"),
    ("madrid",  "results/madrid/multistation_results.json"),
]

for city, path in city_paths:
    if Path(path).exists():
        r = analyze_city(path, city)
        results_all.append(r)

# === Holm-Bonferroni correction (now 6 tests = 3 cities x 2 topologies) ===
print(f"\n{'='*70}")
print(f"=== Holm-Bonferroni correction across all 6 comparisons")
print(f"{'='*70}")

all_pvalues = []
for r in results_all:
    for test in r["tests"]:
        all_pvalues.append({
            "city": r["city"],
            "topology": test["topology"],
            "pvalue_raw": test["wilcoxon_r2_pvalue"],
        })

all_pvalues.sort(key=lambda x: x["pvalue_raw"])
n = len(all_pvalues)
for i, item in enumerate(all_pvalues):
    item["pvalue_corrected"] = min(item["pvalue_raw"] * (n - i), 1.0)
    item["significant_005"] = item["pvalue_corrected"] < 0.05

print(f"\n{'City':<10} {'Topology':<12} {'p (raw)':>12} {'p (Holm-Bonf)':>15} {'Sig.':>6}")
for item in all_pvalues:
    print(f"{item['city']:<10} {item['topology']:<12} "
          f"{item['pvalue_raw']:>12.4e} {item['pvalue_corrected']:>15.4e} "
          f"{'YES' if item['significant_005'] else 'no':>6}")

# === Spearman correlation: h(D) vs aggregate Delta R-squared (across cities) ===
print(f"\n{'='*70}")
print(f"=== Spearman correlation: h(D) vs Delta R-squared (cross-city)")
print(f"{'='*70}")
print("(Tests if larger heterogeneity correlates with worse GCN performance)")

for topology in ["distance", "correlation"]:
    h_vals = []
    delta_vals = []
    cities = []
    for r in results_all:
        for test in r["tests"]:
            if test["topology"] == topology:
                h_vals.append(r["h_index"])
                delta_vals.append(test["delta_r2_aggregate_mean"])
                cities.append(r["city"])
    
    print(f"\n--- Topology: {topology} ---")
    print(f"  {'City':<10} {'h(D)':>8} {'Delta R2':>12}")
    for c, h, d in zip(cities, h_vals, delta_vals):
        print(f"  {c:<10} {h:>8.3f} {d:>+12.4f}")
    
    if len(h_vals) >= 3:
        rho, pval = stats.spearmanr(h_vals, delta_vals)
        print(f"  Spearman rho = {rho:+.3f}, p = {pval:.4f}")
        # Also Pearson for completeness
        r_p, p_p = stats.pearsonr(h_vals, delta_vals)
        print(f"  Pearson r    = {r_p:+.3f}, p = {p_p:.4f}")
        print(f"  Note: with n=3 cities, statistical power is limited; values are descriptive.")

# Save
with open(OUT / "stats_results_3cities.json", "w") as f:
    json.dump({"per_city": results_all, "holm_bonferroni": all_pvalues}, f, indent=2)
print(f"\nSaved to {OUT}/stats_results_3cities.json")

# Clean summary table
summary_rows = []
for r in results_all:
    for test in r["tests"]:
        match = next((p for p in all_pvalues 
                      if p["city"]==r["city"] and p["topology"]==test["topology"]), None)
        summary_rows.append({
            "City": r["city"].capitalize(),
            "h(D)": f"{r['h_index']:.3f}",
            "Topology": test["topology"].capitalize(),
            "Delta R2 (agg, 3 seeds)": f"{test['delta_r2_aggregate_mean']:+.4f} +/- {test['delta_r2_aggregate_std']:.4f}",
            "Delta R2 (per-st, 95% CI)": f"[{test['delta_r2_perstation_ci95_lo']:+.3f}, {test['delta_r2_perstation_ci95_hi']:+.3f}]",
            "Wilcoxon p (Holm-Bonf)": f"{match['pvalue_corrected']:.3e}",
            "Cohen's d": f"{test['cohens_d']:+.2f}",
            "GCN<Linear / total": f"{test['n_stations_gcn_worse']}/{test['n_stations_total']}",
        })

summary_df = pd.DataFrame(summary_rows)
print(f"\n=== CLEAN SUMMARY TABLE FOR PAPER (3 CITIES) ===")
print(summary_df.to_string(index=False))
summary_df.to_csv(OUT / "summary_for_paper_3cities.csv", index=False)
print(f"\nSaved clean summary to {OUT}/summary_for_paper_3cities.csv")
