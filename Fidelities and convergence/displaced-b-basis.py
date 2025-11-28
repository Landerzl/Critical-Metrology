import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

# ----- Fixed parameters -----
Delta = 1.0  # Qubit splitting (Δ)
GAMMA_FIXED = 0.8  # Choose one γ > 1/√2 ≈ 0.707 (e.g., 1.1)

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

# -----------------------------------------------------------
# NEW HAMILTONIAN:  Squeezed–Displaced Rabi H'_+ (from images)
# -----------------------------------------------------------
def displaced_squeezed_rabi_ground_state(N, omega, Delta, gamma):
    """
    Ground state of the squeezed–displaced Hamiltonian H'_+:
    H'_+ = ω cosh(2r) b†b 
           + (ω/2) sinh(2r)(b^2 + b†2)
           + e^r (b+b†)[ ω α_s + g cos(2θ) τ_x − g sin(2θ) τ_z ]
           + Δ̃ τ_z
    """

    # Bosonic operators
    b = qt.tensor(qt.qeye(2), qt.destroy(N))
    Id = qt.tensor(qt.qeye(2), qt.qeye(N))
    num = b.dag() * b

    # Spin operators
    tx = qt.tensor(qt.sigmax(), qt.qeye(N))
    tz = qt.tensor(qt.sigmaz(), qt.qeye(N))

    # Coupling
    g = gamma * np.sqrt(omega * Delta)

    # α_s  (displacement)
    alpha_s = np.sqrt(max(gamma**2 - 1/(4*gamma**2), 0))

    # Δ̃  (effective qubit gap)
    Delta_tilde = np.sqrt(Delta**2 + (2*g*alpha_s)**2)

    # Squeezing parameter r
    arg = 1 + (4 * g**2 * Delta**2) / (omega * Delta_tilde**3)
    r = -0.25 * np.log(arg)

    # Rotation angles
    cos2θ = Delta / Delta_tilde
    sin2θ = -(2*g*alpha_s) / Delta_tilde 

    # Build H'_+
    H = (
        omega * np.cosh(2*r) * num
        + (omega/2) * np.sinh(2*r) * (b**2 + b.dag()**2)
        + np.exp(r) * (b + b.dag()) * (
            omega*alpha_s + g*cos2θ*tx - g*sin2θ*tz
        )
        + Delta_tilde * tz
    )

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
        E0, psi_N = displaced_squeezed_rabi_ground_state(N, om, Delta, GAMMA_FIXED)
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

ax2.set_xlabel(r'Fock cutoff $N$')
ax2.set_ylabel(r'Fidelity $\mathcal{F}(|\psi_N\rangle, |\psi_{N-1}\rangle)$')
ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=1)
ax2.set_ylim(0.5, 1.05)
ax2.legend(title=fr'$\gamma={GAMMA_FIXED}$', frameon=False)
fig2.tight_layout()

plt.show()
