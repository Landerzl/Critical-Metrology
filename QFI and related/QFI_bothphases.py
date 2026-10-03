"""
QFI across the transition at fixed omega/Delta: normal phase (full Rabi H) and superradiant phase.

CORRECTED VERSION (see CORRECTIONS.md): the superradiant half uses the physical displacement
alpha_s = sqrt(gamma^2 - 1/(4 gamma^2)) sqrt(Delta/omega) and the physical QFI, which includes the g-dependence
of the displacement (Physical-QFI/physical_qfi.py, validated against exact diagonalization).
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

# ======================
# Global parameters
# ======================
N = 60          # Bosonic Fock truncation
Delta = 1.0     # Qubit splitting
omega = 5e-5    # Oscillator frequency
domega = Delta * omega

gamma_c = 1.0 / np.sqrt(2)

# Parameter grids for each phase
gamma_vals_norm = np.linspace(0, gamma_c, 100)          # Normal phase
gamma_vals_disp = np.linspace(gamma_c * 1.001, 2 * gamma_c, 100)  # Displaced phase (start slightly above gamma_c)

g_vals_norm = gamma_vals_norm * np.sqrt(domega)
g_vals_disp = gamma_vals_disp * np.sqrt(domega)

dg = 1e-6 * np.sqrt(domega)   # Finite difference step for derivative

# ============================================================
# 1) NORMAL PHASE: Full Rabi Hamiltonian (from QFI_fullH.py)
# ============================================================

# Operators (boson ⊗ qubit), consistent with your QFI_fullH.py file
a_n    = qt.destroy(N)
adag_n = a_n.dag()
I_q    = qt.qeye(2)
I_b    = qt.qeye(N)
sx_n   = qt.sigmax()
sz_n   = qt.sigmaz()

def H_rabi(g):
    """ Full (non-displaced) Rabi Hamiltonian. """
    H0   = omega * qt.tensor(adag_n * a_n, I_q)
    H1   = Delta * qt.tensor(I_b, sz_n)
    Hint = g * qt.tensor(a_n + adag_n, sx_n)
    return H0 + H1 + Hint

def groundstate_norm(g):
    """ Ground state in the normal phase. """
    H = H_rabi(g)
    evals, evecs = H.eigenstates()
    return evecs[0]

def dpsi_dg_norm(g):
    """ Derivative of the ground state wrt g (normal phase). """
    psi_p = groundstate_norm(g + dg)
    psi_m = groundstate_norm(g - dg)
    psi_0 = groundstate_norm(g)

    # Phase alignment
    phase_p = (psi_0.dag() * psi_p)
    phase_m = (psi_0.dag() * psi_m)

    psi_p = psi_p * np.exp(-1j * np.angle(phase_p))
    psi_m = psi_m * np.exp(-1j * np.angle(phase_m))

    return (psi_p - psi_m) / (2 * dg), psi_0

def QFI_norm(g):
    """ QFI in the normal phase. """
    dpsi, psi = dpsi_dg_norm(g)
    overlap   = (psi.dag() * dpsi)
    norm_dpsi2 = (dpsi.dag() * dpsi)
    return 4 * (norm_dpsi2 - np.abs(overlap)**2)

qfi_norm = np.array([QFI_norm(g) for g in g_vals_norm], dtype=complex).real


# ============================================================
# 2) SUPERRADIANT PHASE: physical QFI in the displaced frame
# ============================================================
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Physical-QFI'))
from physical_qfi import DisplacedModel, qfi_map  # noqa: E402

log_qfi_disp, _ = qfi_map(DisplacedModel, [omega], gamma_vals_disp, cutoffs=(N, N + 20, N + 40))
qfi_disp = 10.0 ** log_qfi_disp[0]


# ======================
# 3) Combined plot
# ======================
plt.figure(figsize=(7, 5))

# Normal phase
plt.plot(gamma_vals_norm, qfi_norm, 
         label=rf"Normal phase: $0 \leq \gamma \leq \gamma_c$   (N={N})")

# Displaced phase
plt.plot(gamma_vals_disp, qfi_disp, 
         label=rf"Superradiant phase: $\gamma_c \leq \gamma$   (N={N})")

# Critical line
plt.axvline(x=gamma_c, linestyle='--', linewidth=1,
            label=r"$\gamma_c = 1/\sqrt{2}$")

plt.yscale("log")
plt.xlabel(r"$\gamma$", fontsize=13)
plt.ylabel(r"$F_Q(g)$", fontsize=13)

plt.xticks([0, gamma_c, 2*gamma_c],
           [r"$0$", r"$\gamma_c$", r"$2\gamma_c$"], fontsize=13)
plt.yticks(fontsize=13)

plt.grid(True, which='major', ls='--', alpha=0.5)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()
