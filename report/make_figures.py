#!/usr/bin/env python3
"""Generate figure exports for final report."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SCRIPT_DIR)
FIG_DIR = os.path.join(SCRIPT_DIR, 'figures'); os.makedirs(FIG_DIR, exist_ok=True)

# Set clean style
plt.rcParams.update({
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'font.size': 10,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
    'axes.spines.top': True,
    'axes.spines.right': True,
})

# ===========================================================================
# Figure 1: public_progression.png
# ===========================================================================

csv_path = os.path.join(REPO, 'evidence', 'kaggle_submissions.csv')
df = pd.read_csv(csv_path)
df = df.dropna(subset=['public_roc_auc'])
df = df.sort_values('submission_number').reset_index(drop=True)

# Remove spaces from submission_name for matching
df['submission_name_nospace'] = df['submission_name'].str.replace(' ', '')

# Milestone annotations: (name_pattern, label, (dx, dy) in offset points)
milestones = [
    ('0912-1', 'graph baseline', (8, 12)),
    ('0929-3', '3-status (uses unlabelled)', (8, -18)),
    ('0929-1', 'risk ensemble + alignment (final #2)', (-20, -48)),
    ('1008-4', '181-input teacher', (-40, -62)),
    ('1008-5', 'graph residual', (-90, 18)),
    ('1019-3', 'train on steps 22-35', (10, -30)),
    ('1019-5', 'train on steps 18-35', (-70, 12)),
    ('1019-6', 'soft window (final #1)', (-55, 22)),
]

# Compute running best
df['running_best'] = df['public_roc_auc'].cummax()

# Create figure
fig1, ax1 = plt.subplots(figsize=(7, 4.2))

# Plot all submissions as light grey dots
ax1.scatter(df['submission_number'], df['public_roc_auc'],
            color='lightgrey', s=20, edgecolors='none', alpha=0.7, zorder=1)

# Plot running best as a step line
ax1.step(df['submission_number'], df['running_best'],
         where='post', color='darkblue', zorder=2, linewidth=2)

# Annotate milestones
for name_pattern, label, offset in milestones:
    mask = df['submission_name_nospace'] == name_pattern
    if not mask.any():
        continue
    idx = df[mask].index[0]
    x_val = df.loc[idx, 'submission_number']
    y_val = df.loc[idx, 'public_roc_auc']
    ax1.scatter([x_val], [y_val], color='red', s=40, zorder=4,
                edgecolors='white', linewidths=0.5)
    ax1.annotate(
        label,
        xy=(x_val, y_val),
        xytext=offset,
        textcoords='offset points',
        fontsize=7,
        color='darkred',
        ha='left',
        va='center',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                  edgecolor='darkred', linewidth=0.5, alpha=0.85),
        arrowprops=dict(arrowstyle='-', lw=0.5, color='darkred'),
        zorder=5,
    )

ax1.set_xlabel('Submission order')
ax1.set_ylabel('Public ROC-AUC')
ax1.set_title('Public ROC-AUC by submission', fontsize=10, pad=10)
ax1.set_ylim(0.935, 0.962)
ax1.grid(True, axis='y', linestyle='--', alpha=0.3)

fig1.tight_layout()
output1 = os.path.join(FIG_DIR, 'fig1_public_progression.png')
fig1.savefig(output1, dpi=200)
plt.close(fig1)

# ===========================================================================
# Figure 2: training_window.png
# ===========================================================================

fig2, ax2 = plt.subplots(figsize=(7, 4.2))

# Hard cut points
hard_x = [1, 18, 22, 25]
hard_y = [0.95480, 0.95772, 0.95718, 0.95258]
ax2.plot(hard_x, hard_y, color='darkblue', marker='o',
         label='hard window, 5-seed ensembles', zorder=3)
for xv, yv in zip(hard_x, hard_y):
    ax2.annotate(f'{yv:.5f}', xy=(xv, yv), xytext=(0, 6),
                 textcoords='offset points', ha='center', va='bottom',
                 fontsize=7, color='darkblue')

# Soft window point
ax2.scatter([18], [0.95967], marker='*', s=12 * 12, color='orange',
            edgecolors='darkorange', linewidths=0.5, zorder=4,
            label='soft window from 18 (1019-6)')

# Light-orange band
ax2.fill_between([14, 23], 0.95713, 0.95862, alpha=0.25, color='orange',
                 label='seven one-change variants of 1019-6', zorder=1)

ax2.set_xlabel('First training time step (steps before it dropped or faded)')
ax2.set_ylabel('Public ROC-AUC')
ax2.set_xticks([1, 10, 14, 18, 22, 25])
ax2.set_ylim(0.951, 0.961)
ax2.set_title('Training on recent history (test = steps 36-49)', fontsize=10, pad=10)
ax2.legend(loc='lower left')
ax2.grid(True, axis='y', linestyle='--', alpha=0.3)

fig2.tight_layout()
output2 = os.path.join(FIG_DIR, 'fig2_training_window.png')
fig2.savefig(output2, dpi=200)
plt.close(fig2)

# ===========================================================================
# Print paths
# ===========================================================================
print(output1)
print(output2)

# ===========================================================================
# Figure 3: fig3_pair_selection.png - public AUC vs rank correlation with 1019-6
# ===========================================================================
aud = pd.read_csv(os.path.join(REPO, 'analysis', 'all_submissions_audit.csv'))
aud = aud.dropna(subset=['public_auc', 'spearman_vs_1019_6'])
fig3, ax3 = plt.subplots(figsize=(7, 3.6))
ax3.scatter(aud.spearman_vs_1019_6, aud.public_auc, s=14, color='lightgrey', zorder=1, label='all submissions')
short = {'1019-6': ('1019-6 (final #1)', (-95, -4)), '0929-1': ('0929-1 (final #2)', (-30, 9)),
         '1008-4': ('1008-4', (8, 4)), '0914-1': ('0914-1', (-42, -12)), '1023-2': ('1023-2 TabM', (8, -4))}
for name, (lab, off) in short.items():
    r = aud[aud.submission == name]
    if r.empty: continue
    x, y = float(r.spearman_vs_1019_6.iloc[0]), float(r.public_auc.iloc[0])
    final = name in ('1019-6', '0929-1')
    ax3.scatter([x], [y], s=46 if final else 28, color='darkred' if final else 'steelblue', zorder=3,
                edgecolors='white', linewidths=0.5)
    ax3.annotate(lab, xy=(x, y), xytext=off, textcoords='offset points', fontsize=7,
                 color='darkred' if final else 'steelblue')
ax3.set_xlabel('Spearman rank correlation of test predictions with 1019-6')
ax3.set_ylabel('Public ROC-AUC')
ax3.set_title('Choosing the second final: performance vs. similarity to 1019-6', fontsize=10, pad=8)
ax3.grid(True, linestyle='--', alpha=0.3)
fig3.tight_layout()
output3 = os.path.join(FIG_DIR, 'fig3_pair_selection.png')
fig3.savefig(output3, dpi=200)
plt.close(fig3)
print(output3)
