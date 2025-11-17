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
gamma_vals_disp = np.linspace(gamma_c, 2 * gamma_c, 100)  # Displaced phase

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
# 2) DISPLACED PHASE: Displaced Rabi Hamiltonian H'_+
# ============================================================

# Operators (qubit ⊗ boson), consistent with your displaced Hamiltonian
a_d    = qt.tensor(qt.qeye(2), qt.destroy(N))
adag_d = a_d.dag()
num_d  = adag_d * a_d
sx_d   = qt.tensor(qt.sigmax(), qt.qeye(N))
sz_d   = qt.tensor(qt.sigmaz(), qt.qeye(N))
Id_d   = qt.tensor(qt.qeye(2), qt.qeye(N))

def displaced_rabi_ground_state(N, omega, Delta, gamma):
    """ 
    Ground state of the displaced Hamiltonian:
    H'_+ = ω a†a + Δ σ_z + ω α (a + a†)
           + g σ_x (a + a†) + 2 g α σ_x + ω α²
    with α = sqrt(γ² - 1/(4γ²))
    and g = gamma * sqrt(ω Δ).
    """
    g = gamma * np.sqrt(omega * Delta)
    radicand = gamma**2 - 1.0 / (4.0 * gamma**2)
    alpha = np.sqrt(max(radicand, 0.0))

    H = (omega * num_d
         + Delta * sz_d
         + omega * alpha * (a_d + adag_d)
         + g * sx_d * (a_d + adag_d)
         + 2.0 * g * alpha * sx_d
         + omega * alpha**2 * Id_d)

    E0, psi0 = H.groundstate()
    return float(E0), psi0

def groundstate_disp(g):
    """ Ground state in the displaced phase. """
    gamma = g / np.sqrt(omega * Delta)
    _, psi0 = displaced_rabi_ground_state(N, omega, Delta, gamma)
    return psi0

def dpsi_dg_disp(g):
    """ Derivative of the ground state wrt g (displaced phase). """
    psi_p = groundstate_disp(g + dg)
    psi_m = groundstate_disp(g - dg)
    psi_0 = groundstate_disp(g)

    phase_p = (psi_0.dag() * psi_p)
    phase_m = (psi_0.dag() * psi_m)

    psi_p = psi_p * np.exp(-1j * np.angle(phase_p))
    psi_m = psi_m * np.exp(-1j * np.angle(phase_m))

    return (psi_p - psi_m) / (2 * dg), psi_0

def QFI_disp(g):
    """ QFI in the displaced phase. """
    dpsi, psi = dpsi_dg_disp(g)
    overlap   = (psi.dag() * dpsi)
    norm_dpsi2 = (dpsi.dag() * dpsi)
    return 4 * (norm_dpsi2 - np.abs(overlap)**2)

qfi_disp = np.array([QFI_disp(g) for g in g_vals_disp], dtype=complex).real


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
