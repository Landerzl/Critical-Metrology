"""
QFI heatmap in the superradiant phase, displaced frame only (no rotation, no squeezing).

CORRECTED VERSION (see CORRECTIONS.md).  The previous version used a displacement alpha missing the factor
sqrt(Delta/omega) and differentiated the displaced ground state at fixed displacement.  The physical QFI
(Physical-QFI/physical_qfi.py, DisplacedModel) includes the frame term  alpha_s' (a^dag - a)  and is validated
against exact diagonalization.  Median over three Fock cutoffs; flagged points are interpolated.
Runtime: roughly 10 minutes.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Physical-QFI'))
from physical_qfi import DisplacedModel, qfi_map, gamma_c, Delta  # noqa: E402

# ====================================================================
# 1. PARAMETERS & ARRAYS (50 log-spaced omega values)
# ====================================================================
num_omega = 50
# Range [5e-5, 0.5]; flipped so that the largest omega is at the top of the heatmap
omega_list = np.flip(np.logspace(np.log10(5e-5), np.log10(0.5), num_omega))

gamma_points = 150
# Superradiant phase [gamma_c, 2 gamma_c] (start slightly above gamma_c, where alpha_s ~ sqrt(gamma - gamma_c))
gamma_vals = np.linspace(gamma_c * 1.001, 2 * gamma_c, gamma_points)

CUTOFFS = (60, 80, 100)

# ====================================================================
# 2. PHYSICAL QFI
# ====================================================================
print(f"Starting calculation for {num_omega} bands (cutoffs {CUTOFFS})...")
log_qfi_matrix, flagged = qfi_map(DisplacedModel, omega_list, gamma_vals, cutoffs=CUTOFFS)
print(f"Flagged and interpolated: {flagged.mean() * 100:.1f}% of the points.")

# ====================================================================
# 3. HEATMAP
# ====================================================================
plt.figure(figsize=(8, 6))

sns.set(font_scale=1.2)
ax = sns.heatmap(
    log_qfi_matrix,
    cmap="magma",
    xticklabels=False,
    yticklabels=True,
    cbar_kws={'label': r'$\log_{10}(F_Q)$'}
)

# --- Y-axis Ticks/Labels ---
y_ticks = [0.5, num_omega - 0.5]
y_labels_latex = [r'$5 \cdot 10^{-1}$', r'$5 \cdot 10^{-5}$']

ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels_latex, fontsize=16)

# X-axis Ticks
plt.xticks([0, gamma_points - 1], [r"$\gamma_c$", r"$2\gamma_c$"], fontsize=16)

plt.xticks(rotation=0)
plt.yticks(rotation=0)

plt.xlabel(r'$\gamma$', fontsize=16)
plt.ylabel(r'$\omega/\Delta$ (Log Scale)', fontsize=16, labelpad=-20)

plt.tight_layout()
plt.show()
