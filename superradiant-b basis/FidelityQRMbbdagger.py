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
# Squeezed & Displaced Hamiltonian H_+
# -----------------------------------------------
def squeezed_hamiltonian_ground_state(N, omega, Delta, gamma):
    """
    Ground state of H_+ expressed in terms of b and b† (squeezed frame).
    Includes the parameter r defined by the user.
    """
    # 1. Calculate Coupling and Alpha 
    g = gamma * np.sqrt(omega * Delta)
    radicand = gamma**2 - 1.0/(4.0 * gamma**2)
    alpha_s = np.sqrt(max(radicand, 0.0)) 

    # 2. Calculate Tilde_Delta and Angles (Diagonalizing the qubit part)
    # The qubit part in previous step was: Delta*sz + 2*g*alpha*sx
    # Tilde_Delta is the eigenvalue splitting of that vector.
    Tilde_Delta = np.sqrt(Delta**2 + (2.0 * g * alpha_s)**2)
    
    # Avoiding division by zero if Tilde_Delta is 0 
    if Tilde_Delta == 0:
        cos2theta = 1.0
        sin2theta = 0.0
    else:
        cos2theta = Delta / Tilde_Delta
        sin2theta = (2.0 * g * alpha_s) / Tilde_Delta

    # 3. Calculate Squeezing Parameter r (Formula provided in image)
    # r = 1/4 * ln( 1 + (4 g^2 Delta^2) / (omega * Tilde_Delta^3) )
    numerator_r = 4.0 * (g**2) * (Delta**2)
    denominator_r = omega * (Tilde_Delta**3)
    
    # Safety check for log
    val_inside_log = 1.0 + numerator_r / denominator_r
    r = 0.25 * np.log(val_inside_log)

    # Pre-calculate hyperbolic functions and exponentials
    cosh_2r = np.cosh(2 * r)
    sinh_2r = np.sinh(2 * r)
    exp_minus_r = np.exp(-r)

    # 4. Construct Operators in the 'b' basis
    b = qt.tensor(qt.qeye(2), qt.destroy(N))
    bdag = b.dag()
    # tau matrices correspond to sigma matrices in the rotated frame
    tau_x = qt.tensor(qt.sigmax(), qt.qeye(N))
    tau_z = qt.tensor(qt.sigmaz(), qt.qeye(N))
    
    # 5. Build Hamiltonian H_+
    # H_+ = ω cosh(2r) b†b 
    #       - (ω/2) sinh(2r) (b² + b†²)
    #       + e^{-r} (b + b†) [ ω α_s + g( cos(2θ)τ_x - sin(2θ)τ_z ) ]
    #       + Tilde_Delta τ_z
    
    H = (omega * cosh_2r * bdag * b
         - 0.5 * omega * sinh_2r * (b*b + bdag*bdag)
         + exp_minus_r * (b + bdag) * (omega * alpha_s + g * (cos2theta * tau_x - sin2theta * tau_z))
         + Tilde_Delta * tau_z)

    # 6. Find Ground State
    E0, psi0 = H.groundstate()
    return float(E0), psi0

# -----------------------------------------------
# Compute fidelity vs N
# -----------------------------------------------
E0_dict = {}
Fidelity_dict = {}

print("Calculando fidelidades con el nuevo Hamiltoniano H_+...")

for om, g_val in param_list:
    key = (om, g_val)
    E0_vals = []
    psi_prev = None
    F_vals = [np.nan]
    
    print(f"Procesando omega = {om}...")

    for N in N_list:
        E0, psi_N = squeezed_hamiltonian_ground_state(N, om, Delta, GAMMA_FIXED)
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
    "figure.figsize": (7, 5),
    "font.size": 12,
    "font.family": "serif",
    "axes.labelsize": 14,
    "legend.fontsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
})
linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1, 1, 1))]

# -----------------------------------------------
# Figure: Fidelity vs N
# -----------------------------------------------
fig, ax = plt.subplots()
for i, (om, g_val) in enumerate(param_list):
    label = fr'$\omega = {om}$'
    ax.plot(N_list[1:], Fidelity_dict[(om, g_val)][1:],
            marker='o', linestyle=linestyles[i % len(linestyles)],
            label=label)

ax.set_xlabel(r'Fock cutoff $N$')
ax.set_ylabel(r'Fidelity $\mathcal{F}(|\psi_N\rangle, |\psi_{N-1}\rangle)$')
ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1)
ax.legend(title=fr'$\gamma={GAMMA_FIXED}$', frameon=True)
fig.tight_layout()

plt.show()