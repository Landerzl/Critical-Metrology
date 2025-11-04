import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

# ----- Fixed parameters -----
Delta = 1.0 # Fixed qubit frequency
GAMMA_1_FIXED = 1/np.sqrt(2) # Fixed normalized coupling parameter (γ₁)

# ----- w values to compare (approachig the class oscillator limit) -----
omega_list = np.array([0.5, 0.05, 0.005, 0.0005,0.00005])

# ----- Calculate g values based on fixed gamma_1 -----
# g = gamma_1 * sqrt(omega * Delta)
g_list = GAMMA_1_FIXED * np.sqrt(omega_list * Delta)

# Parameter pairing (ω, g)
param_list = list(zip(omega_list, g_list))

# Fock cutoffs N. Start at 5. Note that in principle N starts at 0. So N = 5 means that the maximum will be 4.
N_list = list(range(5, 80, 1)) 


# -----------------------------------------------
# Helper function: truncate |ψ_N> → |ψ_{N-1}> by removing |↓,N-1> and |↑,N-1> (This is based on how Qutip orders the components)
# -----------------------------------------------
def truncate_spin_fock_state(psi_big, N_minus_1):
    """Truncate a spin+Fock state (dim 2*(N_minus_1+1)) to dim 2*N_minus_1."""
    vec = psi_big.full().flatten()
    N = N_minus_1 + 1  # original Fock truncation
    remove = [N - 1, 2 * N - 1]  # indices of |↓,N-1> and |↑,N-1>
    keep = [i for i in range(len(vec)) if i not in remove]
    truncated_vec = vec[keep]
    psi_small = qt.Qobj(truncated_vec.reshape((2 * N_minus_1, 1)), dims=[[2, N_minus_1], [1, 1]])
    return psi_small.unit()  # renormalize, important

def rabi_ground_state(N, omega, Delta, g):
    """Calculates and returns the ground state and energy of the Rabi Hamiltonian."""
    a = qt.tensor(qt.qeye(2), qt.destroy(N))
    num = a.dag() * a
    sx= qt.tensor(qt.sigmax(), qt.qeye(N))
    sz= qt.tensor(qt.sigmaz(), qt.qeye(N))
    H = omega * num + Delta * sz + g * sx * (a + a.dag())
    E0, psi0 = H.groundstate()
    return float(E0), psi0 

# Store results
E0_dict = {}
dE_dict = {}
Fidelity_dict = {}

# -----------------------------------------------
# LOOP TO CALCULATE ENERGY AND FIDELITY
# -----------------------------------------------

for om, g_val in param_list:
    key = (om, g_val)
    
    E0_vals = []
    psi_prev = None 
    F_vals = [np.nan] 

    for N in N_list:
        # Calculate E0 and the state for cutoff N
        E0, psi_N = rabi_ground_state(N, om, Delta, g_val)
        E0_vals.append(E0)
        
        if psi_prev is not None:
            # Truncate ψ_N to match ψ_{N-1}
            psi_N_red = truncate_spin_fock_state(psi_N, N_minus_1=N - 1)
            # Fidelity
            fidelity = abs(psi_prev.overlap(psi_N_red))**2
            F_vals.append(fidelity)
        
        # Update psi_prev for the next iteration
        psi_prev = psi_N 
    
    # Energy convergence calculation    
    E0_dict[key] = E0_vals   
    Fidelity_dict[key] = F_vals


# ----- Plot config -----
plt.rcParams.update({
    "figure.figsize": (6.0, 3.5),
    "font.size": 12,
    "font.family": "serif",
    "axes.labelsize": 14,
 "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.2,
    "lines.markersize": 5,
})
linestyles = ['-', '--', '-.', ':']


# ----- Figure 1: Ground Energy -----
fig1, ax1 = plt.subplots()
for i, (om, g_val) in enumerate(param_list):
    label = fr'$\omega = {om}$, $\gamma_1 = \gamma_c$'
    ax1.plot(N_list, E0_dict[(om, g_val)],
             marker='x', linestyle=linestyles[i % len(linestyles)],
             label=label)
ax1.set_xlabel(r'Fock cutoff $N$')
ax1.set_ylabel(r'Ground energy $E_0(N)$')
ax1.legend(frameon=False, title=r'$\omega$')
fig1.tight_layout()

# ----- Figure 2: Fidelity F(|psi_N>, |psi_{N-1}>) vs Fock Cutoff N ----- Very good indicator of convergence.
fig2, ax2 = plt.subplots()
for i, (om, g_val) in enumerate(param_list):
    label = fr'$\omega = {om}$, $\gamma_1 = \gamma_c$'
    ax2.plot(N_list[1:], Fidelity_dict[(om, g_val)][1:],
             marker='o', linestyle=linestyles[i % len(linestyles)],
             label=label)

ax2.set_xlabel(r'Fock cutoff $N$')
ax2.set_ylabel(r'Fidelity $\mathcal{F}(|\psi_N\rangle, |\psi_{N-1}\rangle)$')
ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=1)
ax2.set_yscale('linear')
ax2.set_ylim(-0.2, 1.3)
ax2.legend(frameon=False, title=r'$\omega$')
fig2.tight_layout()

plt.show()