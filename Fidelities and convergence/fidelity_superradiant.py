import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

# ----- Fixed parameters -----
Delta = 1.0  # Qubit splitting (Δ)
GAMMA_FIXED = 0.8 

# ----- ω values to compare (approaching classical oscillator limit) -----
omega_list = np.array([0.5, 0.05, 0.005, 0.0005, 0.00005])

# ----- g values from fixed γ -----
g_list = GAMMA_FIXED * np.sqrt(omega_list * Delta)

# Parameter pairs (ω, g)
param_list = list(zip(omega_list, g_list))

# ----- Fock cutoffs -----
N_list = list(range(5, 80, 1))

# -----------------------------------------------
# Helper: truncate |ψ_N> → |ψ_{N-1}> (for fidelity)
# -----------------------------------------------
def truncate_spin_fock_state(psi_big, N_minus_1):
    vec = psi_big.full().flatten()
    N = N_minus_1 + 1
    remove = [N - 1, 2 * N - 1]  # indices of |↓,N-1> and |↑,N-1>
    keep = [i for i in range(len(vec)) if i not in remove]
    truncated_vec = vec[keep]
    psi_small = qt.Qobj(truncated_vec.reshape((2 * N_minus_1, 1)), dims=[[2, N_minus_1], [1, 1]])
    return psi_small.unit()

# -----------------------------------------------
# Displaced Hamiltonian H'_+
# -----------------------------------------------
def displaced_rabi_ground_state(N, omega, Delta, gamma):
    """
    Ground state and energy of:
    H'_+ = ω a†a + Δ σ_z + ω α (a + a†)
           + g σ_x (a + a†) + 2 g α σ_x + ω α²
    with α = +sqrt(γ² - 1/(4γ²)) and g = γ sqrt(ω Δ).
    """
    a = qt.tensor(qt.qeye(2), qt.destroy(N))
    num = a.dag() * a
    sx = qt.tensor(qt.sigmax(), qt.qeye(N))
    sz = qt.tensor(qt.sigmaz(), qt.qeye(N))
    Id = qt.tensor(qt.qeye(2), qt.qeye(N))

    # Coupling constants
    g = gamma * np.sqrt(omega * Delta)
    radicand = gamma**2 - 1.0/(4.0 * gamma**2)
    alpha = np.sqrt(max(radicand, 0.0))  # + root only

    # Hamiltonian H'_+
    H = (omega * num
         + Delta * sz
         + omega * alpha * (a + a.dag())
         + g * sx * (a + a.dag())
         + 2.0 * g * alpha * sx
         + omega * alpha**2 * Id)

    E0, psi0 = H.groundstate()
    return float(E0), psi0

# -----------------------------------------------
# Compute fidelity vs N
# -----------------------------------------------
E0_dict = {}
Fidelity_dict = {}

for om, g_val in param_list:
    key = (om, g_val)
    E0_vals = []
    psi_prev = None
    F_vals = [np.nan]

    for N in N_list:
        E0, psi_N = displaced_rabi_ground_state(N, om, Delta, GAMMA_FIXED)
        E0_vals.append(E0)

        if psi_prev is not None:
            psi_N_red = truncate_spin_fock_state(psi_N, N_minus_1=N - 1)
            fidelity = abs(psi_prev.overlap(psi_N_red))**2
            F_vals.append(fidelity)

        psi_prev = psi_N

    E0_dict[key] = E0_vals
    Fidelity_dict[key] = F_vals

# -----------------------------------------------
# Plot configuration
# -----------------------------------------------
plt.rcParams.update({
    "figure.figsize": (6.5, 4),
    "font.size": 12,
    "font.family": "serif",
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.2,
    "lines.markersize": 5,
})
linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1, 1, 1))]


# -----------------------------------------------
# Figure 1: Fidelity vs N
# -----------------------------------------------
fig2, ax2 = plt.subplots()
for i, (om, g_val) in enumerate(param_list):
    label = fr'$\omega = {om}$'
    ax2.plot(N_list[1:], Fidelity_dict[(om, g_val)][1:],
             marker='o', linestyle=linestyles[i % len(linestyles)],
             label=label)


ax2.set_ylim(0.9, 1.01)
ax2.set_xlabel(r'Fock cutoff $N$')
ax2.set_ylabel(r'Fidelity $\mathcal{F}(|\psi_N\rangle, |\psi_{N-1}\rangle)$')
ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=1)
ax2.legend(title=fr'$\gamma={GAMMA_FIXED}$', frameon=False)
fig2.tight_layout()

plt.show()

