import numpy as np
import matplotlib.pyplot as plt
from qutip import destroy, sigmax, sigmaz, qeye, tensor

# Parameters
N = 60 # Bosonic truncation (max number of photons); tweak this parameter if you wish
Delta = 1.0  # Qubit energy splitting
omega = 5e-5  # Oscillator frequency
#We fixed w/Delta = 5e-5
domega = Delta * omega

# Critical parameter (e.g., critical coupling in some limits)
gamma_c = 1.0 / np.sqrt(2) # Defines the upper limit for gamma
gamma_vals = np.linspace(0, gamma_c, 100) #gamma is defined in the normal phase only
# g = gamma * sqrt(Delta * omega)
g_vals = gamma_vals * np.sqrt(domega)
dg = 1e-6 * np.sqrt(domega) # Infinitesimal step for finite difference. Minimize truncation and round-off error

# Operators
a = destroy(N)
adag = a.dag()
I_q = qeye(2) # Qubit identity
I_b = qeye(N) # Boson identity
sx = sigmax()
sz = sigmaz()

# General Rabi Hamiltonian as a function of g
def H_rabi(g):
    # H0: Free boson term
    H0 = omega * tensor(adag * a, I_q)
    # H1: Qubit term
    H1 = Delta * tensor(I_b, sz)
    # Hint: Interaction term (coupling)
    Hint = g * tensor(a + adag, sx)
    return H0 + H1 + Hint

# Ground state
def groundstate(g):
    H = H_rabi(g)
    # Calculate all eigenstates and eigenvalues
    evals, evecs = H.eigenstates()
    return evecs[0]  # Return the lowest energy eigenstate (ground state). Alternative to H.groundstate

# Centered finite difference of the state |ψ⟩ with respect to g
def dpsi_dg(g):
    psi_p = groundstate(g + dg) # State at g + dg
    psi_m = groundstate(g - dg) # State at g - dg
    psi_0 = groundstate(g)      # State at g

    # Phase Alignment: Essential for a correct numerical derivative.
    # We force the overlap <psi_0 | psi_p> and <psi_0 | psi_m> to be real and positive.
    
    # Calculate overlap (complex number)
    phase_p = (psi_0.dag() * psi_p)
    phase_m = (psi_0.dag() * psi_m)
    
    # Correct phases
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
# NOTE: This calculation can be computationally intensive.
qfi_vals = np.array([QFI(g) for g in g_vals])

# Plot
plt.figure(figsize=(7, 5))
plt.plot(gamma_vals, qfi_vals, label=r"$F_Q \text{ vs } \gamma \quad (N=" + str(N) + r")$")


plt.axvline(x=gamma_c, color='r', linestyle='--', linewidth=1, label=r"$\gamma_c = 1/\sqrt{2}$")
plt.yscale("log")


plt.xticks([0, gamma_c], [r"$0$", r"$\gamma_c$"], fontsize=13)


plt.xlabel(r"$\gamma $", fontsize=13)
plt.ylabel(r"$F_Q(g)$", fontsize=13)
plt.yticks(fontsize=13) 


plt.legend(fontsize=13)


plt.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()
plt.legend()
plt.show()
