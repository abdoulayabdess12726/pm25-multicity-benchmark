"""Figure 1 v3 (FINAL): GCN underperformance vs spatial heterogeneity, 3 cities.
Shows monotonic relationship for distance topology (Spearman rho = -1.000)
and saturation for correlation topology.
"""
import matplotlib.pyplot as plt
import os

os.makedirs('figures', exist_ok=True)

# Final results from full nightly runs (3 seeds, 50 epochs)
data = {
    'Beijing':  {'h': 0.497, 'delta_dist': -0.0198, 'delta_corr': -0.0401, 'n_stations': 12},
    'London':   {'h': 0.656, 'delta_dist': -0.3813, 'delta_corr': -0.4129, 'n_stations': 8},
    'Madrid':   {'h': 0.728, 'delta_dist': -0.4014, 'delta_corr': -0.3336, 'n_stations': 7},
}

cities = list(data.keys())
h_vals = [data[c]['h'] for c in cities]
delta_dist = [data[c]['delta_dist'] for c in cities]
delta_corr = [data[c]['delta_corr'] for c in cities]
n_stations = [data[c]['n_stations'] for c in cities]

fig, ax = plt.subplots(figsize=(8.5, 5.8))

# Plot lines connecting points
ax.plot(h_vals, delta_dist, 'o-', label=r'Distance topology ($\rho = -1.00$, monotonic)',
        markersize=14, linewidth=2.5, color='#1f77b4')
ax.plot(h_vals, delta_corr, 's--', label=r'Correlation topology ($\rho = -0.50$, saturating)',
        markersize=12, linewidth=2, color='#ff7f0e', alpha=0.85)

# Zero reference
ax.axhline(0, color='grey', linestyle=':', alpha=0.5, label='No GCN effect (Linear baseline)')

# City annotations
labels = [
    ('Beijing\nhomogeneous (n=12)',     0.497, -0.0198, 0.50, 0.04, 'left'),
    ('London\nheterogeneous (n=8)',     0.656, -0.3813, 0.605, -0.30, 'right'),
    ('Madrid\nmost heterogeneous (n=7)', 0.728, -0.4014, 0.745, -0.32, 'left'),
]
for label, x, y, tx, ty, ha in labels:
    ax.annotate(label, (x, y), xytext=(tx, ty),
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.8),
                fontsize=10, ha=ha,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                         edgecolor='lightgray', alpha=0.95))

# Format axes
ax.set_xlabel('Spatial Heterogeneity Index h(D)', fontsize=12)
ax.set_ylabel('Delta R-squared (GCN+Transformer - Linear+Transformer)', fontsize=12)
ax.set_title('GCN Underperformance Across Three Heterogeneity Regimes',
             fontsize=13, pad=12)
ax.legend(loc='lower left', fontsize=10, framealpha=0.95)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.45, 0.78)
ax.set_ylim(-0.5, 0.10)

# Key annotation
ax.text(0.59, 0.06,
        r'All Wilcoxon $p < 0.05$ (Holm-Bonferroni); all $|d| > 1.0$ (large)',
        fontsize=9, style='italic', ha='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff8dc',
                 edgecolor='goldenrod', alpha=0.9))

plt.tight_layout()
plt.savefig('figures/fig1_heterogeneity_3cities_FINAL.pdf', dpi=200, bbox_inches='tight')
plt.savefig('figures/fig1_heterogeneity_3cities_FINAL.png', dpi=200, bbox_inches='tight')
print("Figure 1 v3 saved to figures/fig1_heterogeneity_3cities_FINAL.{pdf,png}")
