"""
QFI heatmap in the superradiant phase, displaced + rotated + squeezed frame (b, b^dagger).

CORRECTED VERSION (see CORRECTIONS.md in the repository root).  The previous version had
  (i)   the wrong sign of the tau_z coupling in H_+,
  (ii)  a displacement alpha_s missing the factor sqrt(Delta/omega),
  (iii) a QFI obtained by differentiating the transformed ground state only, ignoring that the
        displacement / rotation / squeezing themselves depend on g.
All three are handled in Physical-QFI/physical_qfi.py, which is validated against exact diagonalization
(Physical-QFI/validate_against_exact.py).

The map is computed with three Fock cutoffs; points where the cutoffs disagree (a thin strip next to the
critical point, where both symmetry-broken branches enter the truncated space) are interpolated.
Runtime: roughly 10 minutes.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Physical-QFI'))
from physical_qfi import SqueezedModel, qfi_map, gamma_c, Delta  # noqa: E402

# ====================================================================
# 1. PARAMETERS & ARRAYS
# ====================================================================
num_omega = 50
# 50 logarithmically spaced values of omega/Delta in [5e-5, 0.5]; largest omega at the top of the heatmap
omega_list = np.flip(np.logspace(np.log10(5e-5), np.log10(0.5), num_omega))

gamma_points = 150
# Superradiant phase [gamma_c, 2 gamma_c]; start slightly above gamma_c, where the frame (alpha_s ~ sqrt(gamma-gamma_c)) is singular
gamma_vals = np.linspace(gamma_c * 1.001, 2 * gamma_c, gamma_points)

CUTOFFS = (60, 80, 100)      # Fock cutoffs used for the robustness check

# ====================================================================
# 2. PHYSICAL QFI (log10), median over cutoffs, flagged points interpolated
# ====================================================================
print(f"Starting calculation for {num_omega} bands (cutoffs {CUTOFFS})...")
log_qfi_matrix, flagged = qfi_map(SqueezedModel, omega_list, gamma_vals, cutoffs=CUTOFFS)
print(f"Flagged and interpolated: {flagged.mean() * 100:.1f}% of the points. "
      f"log10(F_Q) range: {log_qfi_matrix.min():.2f} .. {log_qfi_matrix.max():.2f}")

# ====================================================================
# 3. HEATMAP
# ====================================================================
plt.figure(figsize=(8, 6))

sns.set(font_scale=1.2)
ax = sns.heatmap(
    log_qfi_matrix,
    cmap="magma_r",
    xticklabels=False,
    yticklabels=True,
    cbar_kws={'label': r'$\log_{10}(F_Q)$'}
)

# --- Y-axis Ticks/Labels ---
y_ticks = [0.5, num_omega - 0.5]
y_labels_latex = [r'$5 \cdot 10^{-1}$', r'$5 \cdot 10^{-5}$']

ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels_latex, fontsize=16)

# --- X-axis Ticks ---
plt.xticks([0, gamma_points - 1], [r"$\gamma_c$", r"$2\gamma_c$"], fontsize=16)

plt.xticks(rotation=0)
plt.yticks(rotation=0)

plt.xlabel(r'$\gamma$', fontsize=16)
plt.ylabel(r'$\omega/\Delta$ (Log Scale)', fontsize=16, labelpad=-20)

plt.tight_layout()
plt.show()
