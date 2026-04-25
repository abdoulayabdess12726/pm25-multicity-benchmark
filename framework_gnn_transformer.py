"""
Proposed Framework: GraphEncoder (GCN) + TransformerEncoder + LightweightHead
Ablation study + comparison with baselines.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time, warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Device ─────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print(f"Using device: {DEVICE}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA GENERATION & PREPROCESSING  (same as baselines)
# ══════════════════════════════════════════════════════════════════════════════
np.random.seed(42)
N = 35_000
hours   = np.arange(N)
daily   = np.sin(2 * np.pi * hours / 24)
weekly  = np.sin(2 * np.pi * hours / (24 * 7))
yearly  = np.sin(2 * np.pi * hours / (24 * 365))

pm25 = np.clip(
    60 + 40*yearly + 15*daily + 10*weekly
    + np.random.normal(0, 12, N)
    + np.cumsum(np.random.normal(0, 0.3, N)),
    2, 500
)
no2  = 40 + 20*daily  + np.random.normal(0, 8,  N)
so2  = 25 + 15*yearly + np.random.normal(0, 5,  N)
o3   = 80 - 30*daily  + np.random.normal(0, 10, N)
temp = 15 + 20*yearly + 5*daily + np.random.normal(0, 2, N)
pres = 1013 + 5*yearly + np.random.normal(0, 2, N)
dewp = temp - 10 + np.random.normal(0, 3, N)
rain = np.where(np.random.rand(N) < 0.05, np.random.exponential(2, N), 0)
wspm = np.abs(np.random.normal(2, 1.5, N))

timestamps = pd.date_range('2010-01-01', periods=N, freq='h')
df = pd.DataFrame({
    'year': timestamps.year, 'month': timestamps.month,
    'day':  timestamps.day,  'hour':  timestamps.hour,
    'PM2.5': pm25, 'NO2': no2, 'SO2': so2, 'O3': o3,
    'TEMP': temp,  'PRES': pres, 'DEWP': dewp, 'RAIN': rain, 'WSPM': wspm,
})

FEATURES = ['PM2.5','NO2','SO2','O3','TEMP','PRES','DEWP','RAIN','WSPM']
TARGET    = 'PM2.5'
SEQ_LEN   = 24
BATCH     = 64

scaler = MinMaxScaler()
data_s = scaler.fit_transform(df[FEATURES].values)

n         = len(data_s)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

pm25_idx    = FEATURES.index(TARGET)
pm25_scaler = MinMaxScaler()
pm25_scaler.fit(df[[TARGET]].values)

def inv(arr):
    return pm25_scaler.inverse_transform(arr.reshape(-1,1)).ravel()

class AQDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data    = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len
    def __len__(self):
        return len(self.data) - self.seq_len
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len, pm25_idx]
        return x, y

train_ds = AQDataset(data_s[:train_end],       SEQ_LEN)
val_ds   = AQDataset(data_s[train_end:val_end], SEQ_LEN)
test_ds  = AQDataset(data_s[val_end:],          SEQ_LEN)
train_loader = DataLoader(train_ds, BATCH, shuffle=True,  num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_ds,   BATCH, shuffle=False, num_workers=0, pin_memory=False)
test_loader  = DataLoader(test_ds,  BATCH, shuffle=False, num_workers=0, pin_memory=False)
print(f"Train/Val/Test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. GRAPH TOPOLOGY  (5 fictitious stations, fully connected)
# ══════════════════════════════════════════════════════════════════════════════
N_NODES = 5
src, dst = [], []
for i in range(N_NODES):
    for j in range(N_NODES):
        if i != j:
            src.append(i); dst.append(j)
EDGE_INDEX = torch.tensor([src, dst], dtype=torch.long).to(DEVICE)   # (2, 20)

# ══════════════════════════════════════════════════════════════════════════════
# 3. MODEL MODULES
# ══════════════════════════════════════════════════════════════════════════════
N_FEAT = len(FEATURES)   # 9
D_MODEL = 64

# ── 3a. GraphEncoder (GCN) ────────────────────────────────────────────────────
class GraphEncoder(nn.Module):
    """
    Projects each timestep into N_NODES virtual station embeddings,
    applies 2 GCN layers, then mean-pools across nodes → (B, T, D_MODEL).
    """
    def __init__(self, in_feat=N_FEAT, hidden=32, out=D_MODEL, n_nodes=N_NODES):
        super().__init__()
        self.n_nodes   = n_nodes
        self.proj      = nn.Linear(in_feat, n_nodes * hidden)  # time→node features
        self.gcn1      = GCNConv(hidden, hidden)
        self.gcn2      = GCNConv(hidden, out)
        self.act       = nn.ReLU()

    def forward(self, x, edge_index):
        # x: (B, T, in_feat)
        B, T, F = x.shape
        # project to node features
        h = self.proj(x)                        # (B, T, n_nodes*hidden)
        h = h.view(B * T, self.n_nodes, -1)     # (B*T, n_nodes, hidden)
        hidden = h.shape[-1]

        # flatten into a single big batch for PyG
        # node features: (B*T*n_nodes, hidden)
        node_feat = h.reshape(B * T * self.n_nodes, hidden)

        # build batched edge_index: shift indices per graph
        ei_list = []
        for g in range(B * T):
            ei_list.append(edge_index + g * self.n_nodes)
        big_edge = torch.cat(ei_list, dim=1)    # (2, B*T*20)

        node_feat = self.act(self.gcn1(node_feat, big_edge))
        node_feat = self.act(self.gcn2(node_feat, big_edge))  # (B*T*n_nodes, D_MODEL)

        # mean-pool over nodes
        node_feat = node_feat.view(B * T, self.n_nodes, D_MODEL)
        pooled    = node_feat.mean(dim=1)       # (B*T, D_MODEL)
        return pooled.view(B, T, D_MODEL)       # (B, T, D_MODEL)


# ── 3b. TransformerEncoder ────────────────────────────────────────────────────
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model=D_MODEL, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x):
        return self.encoder(x)          # (B, T, D_MODEL)


# ── 3c. LightweightHead ───────────────────────────────────────────────────────
class LightweightHead(nn.Module):
    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)   # (B,)


# ── Full proposed framework ───────────────────────────────────────────────────
class ProposedFramework(nn.Module):
    def __init__(self, use_gnn=True, use_transformer=True):
        super().__init__()
        self.use_gnn         = use_gnn
        self.use_transformer = use_transformer

        if use_gnn:
            self.graph_enc = GraphEncoder()
        else:
            # ablation: simple linear projection instead of GCN
            self.graph_enc = nn.Sequential(
                nn.Linear(N_FEAT, D_MODEL),
                nn.ReLU(),
            )

        if use_transformer:
            self.seq_enc = TransformerEncoderBlock()
        else:
            # ablation: LSTM instead of Transformer
            self.seq_enc = nn.LSTM(D_MODEL, D_MODEL, batch_first=True)

        self.head = LightweightHead()

    def forward(self, x, edge_index):
        # graph encoding
        if self.use_gnn:
            h = self.graph_enc(x, edge_index)         # (B, T, D_MODEL)
        else:
            h = self.graph_enc(x)                      # (B, T, D_MODEL)

        # sequential encoding
        if self.use_transformer:
            h = self.seq_enc(h)                        # (B, T, D_MODEL)
        else:
            h, _ = self.seq_enc(h)                     # (B, T, D_MODEL)

        # use last timestep
        return self.head(h[:, -1, :])                  # (B,)


# ══════════════════════════════════════════════════════════════════════════════
# 4. TRAINING & EVALUATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def train_framework(model, train_loader, val_loader,
                    epochs=60, patience=7, lr=1e-3):
    opt  = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_val, best_state, wait = np.inf, None, 0

    for epoch in range(1, epochs+1):
        model.train()
        tr_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb, EDGE_INDEX)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(train_loader.dataset)

        model.eval()
        vl_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                vl_loss += loss_fn(model(xb, EDGE_INDEX), yb).item() * len(xb)
        vl_loss /= len(val_loader.dataset)

        if epoch % 10 == 0:
            print(f"  epoch {epoch:3d}  train={tr_loss:.5f}  val={vl_loss:.5f}")

        if vl_loss < best_val:
            best_val, best_state, wait = vl_loss, {k: v.clone() for k,v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model


def evaluate_framework(model, loader):
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            p  = model(xb, EDGE_INDEX).cpu().numpy()
            preds.append(p); actuals.append(yb.numpy())
    p = inv(np.concatenate(preds))
    a = inv(np.concatenate(actuals))
    return (mean_absolute_error(a, p),
            np.sqrt(mean_squared_error(a, p)),
            r2_score(a, p),
            p, a)


def measure_latency(model, loader, n_runs=50):
    model.eval()
    xb, _ = next(iter(loader))
    xb = xb.to(DEVICE)
    # warm-up
    with torch.no_grad():
        for _ in range(5):
            model(xb, EDGE_INDEX)
    if DEVICE.type == 'mps':
        torch.mps.synchronize()
    elif DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(xb, EDGE_INDEX)
            if DEVICE.type == 'mps':
                torch.mps.synchronize()
            elif DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAIN PROPOSED FRAMEWORK (GNN + Transformer)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Proposed Framework (GCN + Transformer) ──")
model_full = ProposedFramework(use_gnn=True, use_transformer=True).to(DEVICE)
model_full = train_framework(model_full, train_loader, val_loader)
mae_full, rmse_full, r2_full, p_full, a_full = evaluate_framework(model_full, test_loader)
lat_full = measure_latency(model_full, test_loader)
print(f"  MAE={mae_full:.4f}  RMSE={rmse_full:.4f}  R²={r2_full:.4f}  Latency={lat_full:.2f}ms")

torch.save(model_full.state_dict(), 'model_framework.pt')
print("  Saved: model_framework.pt")

# ── Quantize head (CPU only, not available on all platforms) ──────────────────
try:
    model_full_cpu = ProposedFramework(use_gnn=True, use_transformer=True)
    model_full_cpu.load_state_dict({k: v.cpu() for k, v in model_full.state_dict().items()})
    model_full_cpu.eval()
    q_model = torch.quantization.quantize_dynamic(
        model_full_cpu, {nn.Linear}, dtype=torch.qint8
    )
    print("  LightweightHead quantized (qint8) on CPU")
except Exception as e:
    print(f"  Quantization skipped (not supported on this platform: {e})")

# ══════════════════════════════════════════════════════════════════════════════
# 6. ABLATION STUDY
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Ablation: No GNN (Linear + Transformer) ──")
model_no_gnn = ProposedFramework(use_gnn=False, use_transformer=True).to(DEVICE)
model_no_gnn = train_framework(model_no_gnn, train_loader, val_loader)
mae_ng, rmse_ng, r2_ng, p_ng, a_ng = evaluate_framework(model_no_gnn, test_loader)
lat_ng = measure_latency(model_no_gnn, test_loader)
print(f"  MAE={mae_ng:.4f}  RMSE={rmse_ng:.4f}  R²={r2_ng:.4f}  Latency={lat_ng:.2f}ms")

print("\n── Ablation: No Transformer (GCN + LSTM) ──")
model_no_tr = ProposedFramework(use_gnn=True, use_transformer=False).to(DEVICE)
model_no_tr = train_framework(model_no_tr, train_loader, val_loader)
mae_nt, rmse_nt, r2_nt, p_nt, a_nt = evaluate_framework(model_no_tr, test_loader)
lat_nt = measure_latency(model_no_tr, test_loader)
print(f"  MAE={mae_nt:.4f}  RMSE={rmse_nt:.4f}  R²={r2_nt:.4f}  Latency={lat_nt:.2f}ms")

# ══════════════════════════════════════════════════════════════════════════════
# 7. RESULTS TABLE
# ══════════════════════════════════════════════════════════════════════════════
# Baselines (from previous run)
baselines = {
    'ARIMA(5,1,2)': dict(MAE=16.8129, RMSE=194.9429, R2=-47.2937, latency_ms=None),
    'LSTM':         dict(MAE=11.6851, RMSE=15.1270,  R2=0.6437,   latency_ms=None),
    'CNN-LSTM':     dict(MAE=11.5390, RMSE=15.1511,  R2=0.6426,   latency_ms=None),
}
framework_results = {
    'Framework (GCN+Transformer)': dict(MAE=mae_full, RMSE=rmse_full, R2=r2_full, latency_ms=lat_full),
    'Ablation: No GNN':            dict(MAE=mae_ng,   RMSE=rmse_ng,   R2=r2_ng,   latency_ms=lat_ng),
    'Ablation: No Transformer':    dict(MAE=mae_nt,   RMSE=rmse_nt,   R2=r2_nt,   latency_ms=lat_nt),
}
all_results = {**baselines, **framework_results}

rows = []
for name, m in all_results.items():
    rows.append({'Model': name, 'MAE': round(m['MAE'],4),
                 'RMSE': round(m['RMSE'],4), 'R2': round(m['R2'],4),
                 'latency_ms': round(m['latency_ms'],2) if m['latency_ms'] else 'N/A'})
df_res = pd.DataFrame(rows)
df_res.to_csv('results_framework.csv', index=False)

print("\n" + "="*72)
print("                    FULL RESULTS TABLE")
print("="*72)
print(df_res.to_string(index=False))
print("="*72)
print("Saved: results_framework.csv")

# ══════════════════════════════════════════════════════════════════════════════
# 8. FIGURES
# ══════════════════════════════════════════════════════════════════════════════

# ── figure_ablation.png ───────────────────────────────────────────────────────
abl_names = ['Framework\n(GCN+Transformer)', 'Ablation:\nNo GNN', 'Ablation:\nNo Transformer']
abl_mae   = [mae_full, mae_ng, mae_nt]
abl_rmse  = [rmse_full, rmse_ng, rmse_nt]
abl_r2    = [r2_full, r2_ng, r2_nt]
abl_lat   = [lat_full, lat_ng, lat_nt]

x = np.arange(3)
w = 0.3
COLORS = ['#1a5276', '#2980b9', '#85c1e9']

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, vals, ylabel, title in zip(
    axes,
    [abl_mae, abl_rmse, abl_r2],
    ['MAE (µg/m³)', 'RMSE (µg/m³)', 'R²'],
    ['MAE — Ablation', 'RMSE — Ablation', 'R² — Ablation'],
):
    bars = ax.bar(x, vals, color=COLORS, edgecolor='white', linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(abl_names, fontsize=8)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=11)
    ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=8)
    ax.set_ylim(0, max(vals)*1.25 if min(vals) >= 0 else min(vals)*1.4)

plt.suptitle('Ablation Study — GCN + Transformer Framework', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figure_ablation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figure_ablation.png")

# ── figure_comparison_all_models.png ─────────────────────────────────────────
all_names  = ['ARIMA', 'LSTM', 'CNN-LSTM', 'Framework\n(GCN+Tr)', 'No GNN', 'No Tr']
all_mae    = [16.8129, 11.6851, 11.5390, mae_full, mae_ng, mae_nt]
all_rmse   = [194.9429, 15.1270, 15.1511, rmse_full, rmse_ng, rmse_nt]
all_r2     = [-47.2937, 0.6437, 0.6426, r2_full, r2_ng, r2_nt]

PALETTE = ['#7f8c8d','#2980b9','#27ae60','#e74c3c','#f39c12','#9b59b6']
x6 = np.arange(len(all_names))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# MAE (exclude ARIMA for scale clarity — add annotation)
ax = axes[0]
bars = ax.bar(x6, all_mae, color=PALETTE, edgecolor='white')
ax.set_xticks(x6); ax.set_xticklabels(all_names, fontsize=8)
ax.set_ylabel('MAE (µg/m³)'); ax.set_title('MAE — All Models')
ax.set_ylim(0, max(all_mae) * 1.15)
ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=7)
ax.axhline(11.54, color='gray', linestyle='--', linewidth=0.8, label='CNN-LSTM baseline')
ax.legend(fontsize=7)

# RMSE — clip ARIMA to show deep models better, annotate
ax = axes[1]
clip_rmse = [min(v, 30) for v in all_rmse]  # clip for visibility
bars = ax.bar(x6, clip_rmse, color=PALETTE, edgecolor='white')
ax.set_xticks(x6); ax.set_xticklabels(all_names, fontsize=8)
ax.set_ylabel('RMSE (µg/m³, clipped at 30)'); ax.set_title('RMSE — All Models')
for i, (b, orig) in enumerate(zip(bars, all_rmse)):
    label = f'{orig:.2f}' if orig > 30 else f'{orig:.2f}'
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
            label, ha='center', va='bottom', fontsize=7)
ax.set_ylim(0, 35)

# R² — clip negative for visibility
ax = axes[2]
clip_r2 = [max(v, -1.0) for v in all_r2]
bars = ax.bar(x6, clip_r2, color=PALETTE, edgecolor='white')
ax.set_xticks(x6); ax.set_xticklabels(all_names, fontsize=8)
ax.set_ylabel('R² (clipped at -1)'); ax.set_title('R² — All Models')
ax.axhline(0, color='black', linewidth=0.6, linestyle='-')
for i, (b, orig) in enumerate(zip(bars, all_r2)):
    ypos = b.get_height() + 0.02 if b.get_height() >= 0 else b.get_height() - 0.05
    ax.text(b.get_x()+b.get_width()/2, ypos,
            f'{orig:.4f}', ha='center', va='bottom', fontsize=7)
ax.set_ylim(-1.2, 1.1)

plt.suptitle('All Models Comparison — Air Quality PM2.5 Forecasting',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figure_comparison_all_models.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figure_comparison_all_models.png")

print("\nDone.")
