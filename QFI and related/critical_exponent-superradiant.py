"""
Critical behaviour of the physical QFI on the SUPERRADIANT side of the transition.

The previous version of this file was a copy of critical_exponent.py: it used the normal-phase Hamiltonian
and gamma_c - gamma, so it did not compute a superradiant QFI.  This version computes the physical QFI
(Physical-QFI/physical_qfi.py: squeezed frame, frame derivative included, validated against exact
diagonalization) as a function of  delta = gamma/gamma_c - 1  at omega/Delta = 5e-5.

Result (printed): over 3e-3 < delta < 0.1 the QFI follows a power law with exponent of about -0.6,
i.e. NOT the -2 of the normal phase.  The closest points to gamma_c (flagged, open markers) depend on the
Fock cutoff and are not reliable.  The slopes -1 and -2 are shown as guides only.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Physical-QFI'))
from physical_qfi import SqueezedModel, qfi_map, gamma_c, Delta  # noqa: E402

# --- Parameters ---
omega = 5e-5                       # Oscillator frequency
num_points = 40
min_delta, max_delta = 1e-4, 0.5   # delta = gamma/gamma_c - 1
delta_vals = np.geomspace(min_delta, max_delta, num_points)
gamma_vals = gamma_c * (1.0 + delta_vals)

# --- Physical QFI (median over three Fock cutoffs) ---
log_qfi, flagged = qfi_map(SqueezedModel, [omega], gamma_vals, cutoffs=(60, 80, 100))
log_qfi, flagged = log_qfi[0], flagged[0]
qfi_vals = 10.0 ** log_qfi

# --- Power-law fit on reliable points ---
sel = (~flagged) & (delta_vals > 3e-3) & (delta_vals < 0.1)
slope, intercept = np.polyfit(np.log10(delta_vals[sel]), log_qfi[sel], 1)
print(f"Fitted exponent on 3e-3 < delta < 0.1: {slope:.2f}   ({sel.sum()} reliable points)")

# --- Plotting ---
plt.figure(figsize=(7, 5))

ok = ~flagged
plt.loglog(delta_vals[ok], qfi_vals[ok], 'o-', ms=3, label=r"$F_Q$ (physical), $\omega/\Delta=5\cdot10^{-5}$")
plt.loglog(delta_vals[flagged], qfi_vals[flagged], 'o', mfc='none', color='tab:blue',
           label="cutoff-dependent (unreliable)")

# guides: slopes -1 and -2 through the fitted value at delta = 1e-2
x_ref = np.geomspace(2e-3, 1.5e-1, 10)
y0 = 10.0 ** (slope * np.log10(1e-2) + intercept)
plt.loglog(x_ref, y0 * (x_ref / 1e-2) ** (-1.0), color='green', linestyle=':', linewidth=2, label="slope $-1$ (guide)")
plt.loglog(x_ref, y0 * (x_ref / 1e-2) ** (-2.0), color='red', linestyle=':', linewidth=2, label="slope $-2$ (guide)")
plt.loglog(x_ref, y0 * (x_ref / 1e-2) ** slope, color='k', linestyle='--', linewidth=1,
           label=rf"fit: slope ${slope:.2f}$")

plt.xlabel(r"$\gamma/\gamma_c - 1$", fontsize=13)
plt.ylabel(r"$F_Q(g)$", fontsize=13)
plt.tick_params(labelsize=13)

plt.grid(True, which="major", axis='x', ls="-", alpha=0.4)
plt.grid(True, which="minor", axis='x', ls="--", alpha=0.2)

plt.legend(fontsize=11)
plt.tight_layout()
plt.show()
