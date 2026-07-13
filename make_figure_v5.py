import matplotlib.pyplot as plt
import os

os.makedirs('figures', exist_ok=True)

# v5 results (no NO2)
data = {
    'Beijing':  {'h': 0.497, 'delta_dist': -0.0172, 'delta_corr': -0.0375, 'n_stations': 12},
    'London':   {'h': 0.656, 'delta_dist': -0.3754, 'delta_corr': -0.4014, 'n_stations': 8},
    'Madrid':   {'h': 0.728, 'delta_dist': -0.3213, 'delta_corr': -0.3795, 'n_stations': 7},
}

cities = list(data.keys())
h_vals = [data[c]['h'] for c in cities]
delta_dist = [data[c]['delta_dist'] for c in cities]
delta_corr = [data[c]['delta_corr'] for c in cities]

fig, ax = plt.subplots(figsize=(8.5, 5.8))

ax.plot(h_vals, delta_dist, 'o-', label=r'Distance topology (Pearson $r=-0.90$)',
        markersize=14, linewidth=2.5, color='#1f77b4')
ax.plot(h_vals, delta_corr, 's--', label=r'Correlation topology (Pearson $r=-0.93$)',
        markersize=12, linewidth=2, color='#ff7f0e', alpha=0.85)

ax.axhline(0, color='grey', linestyle=':', alpha=0.5, label='No GCN effect')

labels = [
    ('Beijing\nhomogeneous (n=12)',     0.497, -0.0172, 0.50, 0.04, 'left'),
    ('London\nheterogeneous (n=8)',     0.656, -0.3754, 0.610, -0.30, 'right'),
    ('Madrid\nmost heterogeneous (n=7)', 0.728, -0.3213, 0.745, -0.27, 'left'),
]
for label, x, y, tx, ty, ha in labels:
    ax.annotate(label, (x, y), xytext=(tx, ty),
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.8),
                fontsize=10, ha=ha,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                         edgecolor='lightgray', alpha=0.95))

ax.set_xlabel('Spatial Heterogeneity Index h(D)', fontsize=12)
ax.set_ylabel(r'$\Delta R^2$ (GCN+Transformer $-$ Linear+Transformer)', fontsize=12)
ax.set_title('GCN-Transformer Underperformance Across Three Heterogeneity Regimes',
             fontsize=13, pad=12)
ax.legend(loc='lower left', fontsize=10, framealpha=0.95)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.45, 0.78)
ax.set_ylim(-0.5, 0.10)

ax.text(0.59, 0.06,
        r'All Wilcoxon $p \leq 0.016$ (Holm-Bonferroni); all $|d| > 1.0$ (large effect)',
        fontsize=9, style='italic', ha='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff8dc',
                 edgecolor='goldenrod', alpha=0.9))

plt.tight_layout()
plt.savefig('figures/fig1_heterogeneity_3cities_v5.pdf', dpi=200, bbox_inches='tight')
plt.savefig('figures/fig1_heterogeneity_3cities_v5.png', dpi=200, bbox_inches='tight')
print("Figure v5 saved")
