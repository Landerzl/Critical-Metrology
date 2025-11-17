import numpy as np
import matplotlib.pyplot as plt
import qutip as qt  # nuevo: usamos alias qt

# Parameters
N = 60  # Bosonic truncation (max number of photons)
Delta = 1.0  # Qubit energy splitting
omega = 5e-5  # Oscillator frequency
# We fixed w/Delta = 5e-5
domega = Delta * omega

# Critical parameter (e.g., critical coupling in some limits)
gamma_c = 1.0 / np.sqrt(2)  # Defines the upper limit for gamma
gamma_vals = np.linspace(gamma_c, gamma_c*2, 100)  # gamma is defined in the normal phase only

# Relation g = gamma * sqrt(Delta * omega)
g_vals = gamma_vals * np.sqrt(domega)
dg = 1e-6 * np.sqrt(domega)  # Infinitesimal step for finite difference

# -------------------------------------------------------------------
# Operators for the displaced Rabi Hamiltonian
# -------------------------------------------------------------------
a = qt.tensor(qt.qeye(2), qt.destroy(N))
adag = a.dag()
num = adag * a
sx = qt.tensor(qt.sigmax(), qt.qeye(N))
sz = qt.tensor(qt.sigmaz(), qt.qeye(N))
Id = qt.tensor(qt.qeye(2), qt.qeye(N))

def displaced_rabi_ground_state(N, omega, Delta, gamma):
    """
    Ground state and energy of:
    H'_+ = ω a†a + Δ σ_z + ω α (a + a†)
           + g σ_x (a + a†) + 2 g α σ_x + ω α²
    with α = +sqrt(γ² - 1/(4γ²)) and g = γ sqrt(ω Δ).
    """
    # Coupling constants
    g = gamma * np.sqrt(omega * Delta)
    radicand = gamma**2 - 1.0 / (4.0 * gamma**2)
    alpha = np.sqrt(max(radicand, 0.0))  # + root only, 0 if radicand < 0

    # Hamiltonian H'_+
    H = (omega * num
         + Delta * sz
         + omega * alpha * (a + adag)
         + g * sx * (a + adag)
         + 2.0 * g * alpha * sx
         + omega * alpha**2 * Id)

    E0, psi0 = H.groundstate()
    return float(E0), psi0

# -------------------------------------------------------------------
# Ground state as a function of g 
# -------------------------------------------------------------------
def groundstate(g):
    # g = gamma * sqrt(omega * Delta)  ⇒  gamma = g / sqrt(omega * Delta)
    gamma = g / np.sqrt(omega * Delta)
    _, psi0 = displaced_rabi_ground_state(N, omega, Delta, gamma)
    return psi0  

# Centered finite difference of the state |ψ⟩ with respect to g
def dpsi_dg(g):
    psi_p = groundstate(g + dg)  # State at g + dg
    psi_m = groundstate(g - dg)  # State at g - dg
    psi_0 = groundstate(g)       # State at g

    # Phase Alignment: Essential for a correct numerical derivative.
    # We force the overlap <psi_0 | psi_p> and <psi_0 | psi_m> to be real and positive.
    phase_p = (psi_0.dag() * psi_p)
    phase_m = (psi_0.dag() * psi_m)

    psi_p = psi_p * np.exp(-1j * np.angle(phase_p))
    psi_m = psi_m * np.exp(-1j * np.angle(phase_m))

    # Centered finite difference (derivative approximation)
    return (psi_p - psi_m) / (2 * dg), psi_0

# Quantum Fisher Information (QFI) for each g
def QFI(g):
    dpsi, psi = dpsi_dg(g)
    overlap = (psi.dag() * dpsi)
    norm_dpsi2 = (dpsi.dag() * dpsi)

    # F_Q = 4 * (||dpsi||^2 - |<psi | dpsi>|^2)
    return 4 * (norm_dpsi2 - np.abs(overlap)**2)

# Calculation
qfi_vals = np.array([QFI(g) for g in g_vals], dtype=complex)  
qfi_vals = qfi_vals.real

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
