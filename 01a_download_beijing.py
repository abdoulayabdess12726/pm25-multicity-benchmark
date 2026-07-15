"""
01_download_beijing.py — Gère le zip imbriqué UCI Beijing
Structure réelle du zip UCI :
  PRSA2017_Data_20130301-20170228.zip  ← zip dans le zip !
  data.csv / test.csv                  ← ignorer
"""
import os, zipfile, urllib.request
import numpy as np
import pandas as pd

DATA_DIR   = "data/beijing_real"
OUTPUT_CSV = "data/beijing_real_combined.csv"
UCI_URL    = "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip"
ZIP_PATH   = "data/beijing_raw.zip"
FEATURES   = ["NO2", "TEMP", "PRES", "DEWP", "WSPM"]
TARGET     = "PM2.5"

os.makedirs("data", exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ── 1. Télécharger si absent ──────────────────────────────────────
if not os.path.exists(ZIP_PATH):
    print("Téléchargement...")
    urllib.request.urlretrieve(UCI_URL, ZIP_PATH)
else:
    print(f"Zip déjà présent : {ZIP_PATH}")

# ── 2. Extraire le zip principal ──────────────────────────────────
print("\nExtraction zip principal...")
inner_zip_path = None
with zipfile.ZipFile(ZIP_PATH, "r") as z:
    for name in z.namelist():
        print(f"  {name}")
        # Extraire le zip imbriqué
        if name.endswith(".zip") and "PRSA" in name:
            z.extract(name, DATA_DIR)
            inner_zip_path = os.path.join(DATA_DIR, name)
            print(f"  → zip imbriqué extrait : {inner_zip_path}")

# ── 3. Extraire le zip imbriqué ───────────────────────────────────
if inner_zip_path and os.path.exists(inner_zip_path):
    print(f"\nExtraction zip imbriqué : {inner_zip_path}")
    with zipfile.ZipFile(inner_zip_path, "r") as z2:
        z2.extractall(DATA_DIR)
        extracted = z2.namelist()
        print(f"  {len(extracted)} fichiers extraits")
        for f in extracted[:5]:
            print(f"  {f}")
else:
    print("Pas de zip imbriqué trouvé — extraction directe")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(DATA_DIR)

# ── 4. Trouver tous les CSV Beijing (PRSA_Data_*.csv) ─────────────
csv_files = []
for root, dirs, files in os.walk(DATA_DIR):
    for f in sorted(files):
        if f.endswith(".csv") and "PRSA_Data" in f:
            csv_files.append(os.path.join(root, f))

if not csv_files:
    # Fallback : tous les CSV
    for root, dirs, files in os.walk(DATA_DIR):
        for f in sorted(files):
            if f.endswith(".csv"):
                path = os.path.join(root, f)
                sample = pd.read_csv(path, nrows=2)
                if "PM2.5" in sample.columns or "PM25" in sample.columns:
                    csv_files.append(path)

print(f"\n{len(csv_files)} stations Beijing trouvées :")
for f in csv_files:
    print(f"  {os.path.basename(f)}")

if not csv_files:
    print("\nTous les fichiers disponibles :")
    for root, dirs, files in os.walk(DATA_DIR):
        for f in files:
            print(f"  {os.path.join(root, f)}")
    raise SystemExit("Aucun CSV Beijing trouvé")

# ── 5. Charger et concaténer ──────────────────────────────────────
dfs = []
for f in csv_files:
    df = pd.read_csv(f)
    # Ajouter colonne station depuis le nom de fichier si absente
    if "station" not in df.columns:
        station = os.path.basename(f).replace("PRSA_Data_","").split("_")[0]
        df["station"] = station
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)
print(f"\nDataset brut : {len(df_all):,} lignes")
print(f"Colonnes : {list(df_all.columns)}")

# ── 6. Timestamp ──────────────────────────────────────────────────
df_all["datetime"] = pd.to_datetime(df_all[["year","month","day","hour"]])

# ── 7. Sélectionner et nettoyer ───────────────────────────────────
COLS = [TARGET] + FEATURES + ["datetime", "station"]
df_sel = df_all[COLS].copy()

for col in [TARGET] + FEATURES:
    df_sel.loc[df_sel[col] < 0, col] = np.nan

df_sel = df_sel.sort_values(["station", "datetime"])
df_sel[[TARGET]+FEATURES] = (
    df_sel.groupby("station")[[TARGET]+FEATURES]
    .transform(lambda x: x.interpolate(method="linear", limit=6))
)
before = len(df_sel)
df_sel = df_sel.dropna(subset=[TARGET]+FEATURES)
print(f"Nettoyage : {before:,} → {len(df_sel):,} lignes")

# ── 8. Agréger par heure (moyenne toutes stations) ────────────────
df_hourly = (
    df_sel.groupby("datetime")[[TARGET]+FEATURES]
    .mean().reset_index().sort_values("datetime")
)

print(f"\n✓ Dataset final : {len(df_hourly):,} enregistrements horaires")
print(f"  Période  : {df_hourly['datetime'].min()} → {df_hourly['datetime'].max()}")
print(f"  PM2.5    : mean={df_hourly[TARGET].mean():.1f}  std={df_hourly[TARGET].std():.1f}  max={df_hourly[TARGET].max():.1f}")

df_hourly.to_csv(OUTPUT_CSV, index=False)
print(f"\n✓ Sauvegardé : {OUTPUT_CSV}")
print("→ Lancer : python 02_train_all_models.py")
