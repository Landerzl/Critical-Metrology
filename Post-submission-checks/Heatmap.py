"""
Post-submission check: QFI heatmap in the superradiant phase (squeezed frame), SINGLE Fock cutoff.

CORRECTED VERSION (see CORRECTIONS.md).  Uses the validated physical QFI of Physical-QFI/physical_qfi.py
(correct tau_z sign, physical displacement alpha_s, frame derivative included).

Unlike 'Optimal Base superradiant/heatmap-b,bdagger.py' this does NOT compare several cutoffs, so the thin
strip near gamma_c where the result depends on the cutoff is NOT flagged: change CUTOFFS to see that dependence.
Runtime: a few minutes.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Physical-QFI'))
from physical_qfi import SqueezedModel, qfi_map, gamma_c, Delta  # noqa: E402

num_omega = 50
omega_list = np.flip(np.logspace(np.log10(5e-5), np.log10(0.5), num_omega))   # largest omega at the top

N = 60
gamma_points = 150
gamma_vals = np.linspace(gamma_c * 1.001, 2 * gamma_c, gamma_points)           # superradiant phase

CUTOFFS = (N,)     # single cutoff

print(f"Starting calculation for {num_omega} bands using the squeezed H_+ (cutoffs {CUTOFFS})...")
log_qfi_matrix, flagged = qfi_map(SqueezedModel, omega_list, gamma_vals, cutoffs=CUTOFFS)
print(f"log10(F_Q) range: {log_qfi_matrix.min():.2f} .. {log_qfi_matrix.max():.2f}")

plt.figure(figsize=(8, 6))

sns.set(font_scale=1.2)
ax = sns.heatmap(
    log_qfi_matrix,
    cmap="magma_r",
    xticklabels=False,
    yticklabels=True,
    cbar_kws={'label': r'$\log_{10}(F_Q)$'}
)

y_ticks = [0.5, num_omega - 0.5]
y_labels_latex = [r'$5 \cdot 10^{-1}$', r'$5 \cdot 10^{-5}$']
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels_latex, fontsize=16)

plt.xticks([0, gamma_points - 1], [r"$\gamma_c$", r"$2\gamma_c$"], fontsize=16)
plt.xticks(rotation=0)
plt.yticks(rotation=0)

plt.xlabel(r'$\gamma$', fontsize=16)
plt.ylabel(r'$\omega/\Delta$ (Log Scale)', fontsize=16, labelpad=-20)

plt.tight_layout()
plt.show()
