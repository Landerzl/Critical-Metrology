"""
QFI of the superradiant phase at fixed omega/Delta, displaced frame.

CORRECTED VERSION (see CORRECTIONS.md): physical displacement alpha_s = sqrt(gamma^2 - 1/(4 gamma^2)) sqrt(Delta/omega)
and the physical QFI, which includes the g-dependence of the displacement (frame term alpha_s' (a^dag - a)).
Uses Physical-QFI/physical_qfi.py (validated against exact diagonalization); median over three cutoffs.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Physical-QFI'))
from physical_qfi import DisplacedModel, qfi_map, gamma_c, Delta  # noqa: E402

# Parameters
N = 60            # Bosonic truncation (the map is computed for N, N+20, N+40)
omega = 5e-5      # Oscillator frequency (w/Delta = 5e-5)

# Superradiant phase [gamma_c, 2 gamma_c]; start slightly above gamma_c where alpha_s ~ sqrt(gamma - gamma_c)
gamma_vals = np.linspace(gamma_c * 1.001, gamma_c * 2, 100)

# Physical QFI (log10), median over cutoffs
log_qfi, flagged = qfi_map(DisplacedModel, [omega], gamma_vals, cutoffs=(N, N + 20, N + 40))
qfi_vals = 10.0 ** log_qfi[0]

# Plot
plt.figure(figsize=(7, 5))
plt.plot(gamma_vals, qfi_vals, label=rf"$F_Q \text{{ vs }} \gamma \quad (N={N})$")

plt.axvline(x=gamma_c, linestyle='--', linewidth=1,
            label=r"$\gamma_c = 1/\sqrt{2}$")
plt.yscale("log")

plt.xticks([0, gamma_c], [r"$0$", r"$\gamma_c$"], fontsize=13)

plt.xlabel(r"$\gamma$", fontsize=13)
plt.ylabel(r"$F_Q(g)$", fontsize=13)
plt.yticks(fontsize=13)

plt.legend(fontsize=13)
plt.grid(True, which='major', ls='--', alpha=0.5)
plt.tight_layout()
plt.show()
