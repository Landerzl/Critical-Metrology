"""
3D waterfall: physical QFI of the superradiant phase vs gamma for several omega/Delta.

CORRECTED VERSION (see CORRECTIONS.md): correct tau_z sign, physical displacement alpha_s, and the
frame derivative included in the QFI (Physical-QFI/physical_qfi.py, validated against exact diagonalization).
Median over three Fock cutoffs; points where the cutoffs disagree are interpolated.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Physical-QFI'))
from physical_qfi import SqueezedModel, qfi_map, gamma_c, Delta  # noqa: E402

# ====================================================================
# 1. PARAMETERS
# ====================================================================
gamma_points = 150
gamma_vals = np.linspace(gamma_c * 1.001, 2 * gamma_c, gamma_points)
target_omegas = [5e-1, 5e-2, 5e-3, 5e-4, 5e-5]
CUTOFFS = (60, 80, 100)

# ====================================================================
# 2. PHYSICAL QFI FOR EACH OMEGA
# ====================================================================
print("Calculating lines...")
log_qfi, flagged = qfi_map(SqueezedModel, target_omegas, gamma_vals, cutoffs=CUTOFFS)
print(f"Flagged and interpolated: {flagged.mean() * 100:.1f}% of the points.")

# ====================================================================
# 3. 3D PLOT
# ====================================================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

colors = ['#000080', '#4169E1', '#2E8B57', '#FFA500', '#DC143C']

for i, omega_exact in enumerate(target_omegas):
    xs = gamma_vals
    ys = np.full_like(xs, np.log10(omega_exact))
    zs = log_qfi[i]
    ax.plot(xs, ys, zs, color=colors[i], linewidth=2, label=f'$\\omega = {omega_exact:.0e}$')

ax.set_xlabel(r'$\gamma$', fontsize=14, labelpad=10)
ax.set_ylabel(r'$\log_{10}(\omega)$', fontsize=14, labelpad=10)
ax.set_zlabel(r'$\log_{10}(F_Q)$', fontsize=14, labelpad=10)

yticks_vals = np.log10(target_omegas)
yticks_labels = [r'$5 \cdot 10^{-1}$', r'$5 \cdot 10^{-2}$', r'$5 \cdot 10^{-3}$', r'$5 \cdot 10^{-4}$', r'$5 \cdot 10^{-5}$']
ax.set_yticks(yticks_vals)
ax.set_yticklabels(yticks_labels, fontsize=10, rotation=-15)

ax.set_xticks([gamma_c, 2 * gamma_c])
ax.set_xticklabels([r'$\gamma_c$', r'$2\gamma_c$'], fontsize=12)

ax.view_init(elev=30, azim=-60)

plt.legend(loc='upper right', bbox_to_anchor=(1.1, 0.9))
plt.tight_layout()
plt.show()
