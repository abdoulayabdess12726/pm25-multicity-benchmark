"""Sensitivity analysis on graph density (k neighbors).
Tests k=3, k=5, k=8 on the three cities in --quick mode (~30 min total).
Saves results in results/sensitivity_k/.
"""
import subprocess, json, shutil, re, os
from pathlib import Path

SAVE_DIR = Path("results/sensitivity_k")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

orig = "06_train_multistation.py"
backup = "06_train_multistation_BACKUP_for_k.py"
shutil.copy(orig, backup)
print(f"Backup: {backup}")

def patch_k(file_path, new_k):
    with open(file_path) as f:
        content = f.read()
    new_content = re.sub(r'K_NEIGHBORS\s*=\s*\d+', f'K_NEIGHBORS = {new_k}', content)
    with open(file_path, 'w') as f:
        f.write(new_content)

try:
    for k in [3, 5, 8]:
        for city in ['beijing', 'london', 'madrid']:
            print(f"\n{'='*60}\n  k={k}, city={city}\n{'='*60}")
            patch_k(orig, k)
            
            cmd = ["python", "-u", orig, "--city", city, "--quick"]
            if city == 'beijing':
                cmd += ["--data_dir", "data/beijing_real/PRSA_Data_20130301-20170228"]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            
            log_path = SAVE_DIR / f"{city}_k{k}_log.txt"
            with open(log_path, 'w') as f:
                f.write(result.stdout + "\n--STDERR--\n" + result.stderr)
            
            json_src = Path(f"results/{city}/multistation_results.json")
            if json_src.exists():
                json_dst = SAVE_DIR / f"{city}_k{k}_results.json"
                shutil.copy(json_src, json_dst)
                print(f"  saved {json_dst}")

finally:
    shutil.copy(backup, orig)
    os.remove(backup)
    print(f"\nRestored {orig}")

# Summary
print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
print(f"{'City':<10} {'k':<5} {'Topo':<12} {'GCN R2':<12} {'Lin R2':<12} {'DeltaR2':<10}")
rows = []
for k in [3, 5, 8]:
    for city in ['beijing', 'london', 'madrid']:
        json_path = SAVE_DIR / f"{city}_k{k}_results.json"
        if not json_path.exists():
            continue
        with open(json_path) as f:
            d = json.load(f)
        for topo in ['distance', 'correlation']:
            gcn = d['graphs'][topo]['GCN+Transformer']['R2_mean']
            lin = d['graphs'][topo]['Linear+Transformer']['R2_mean']
            delta = gcn - lin
            print(f"{city:<10} {k:<5} {topo:<12} {gcn:<12.4f} {lin:<12.4f} {delta:+.4f}")
            rows.append({'city': city, 'k': k, 'topology': topo,
                        'gcn_r2': gcn, 'lin_r2': lin, 'delta_r2': delta})

import csv
with open(SAVE_DIR / "summary.csv", 'w', newline='') as f:
    if rows:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
print(f"\nSaved {SAVE_DIR}/summary.csv")
