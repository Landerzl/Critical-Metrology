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

# -----------------------------------------------
# NEW HELPER FUNCTION FOR QRM HAMILTONIAN
# -----------------------------------------------
def qrm_ground_state(N, omega, Delta, gamma_1_fixed):
    """Calculates and returns the ground state and energy of the QRM Hamiltonian."""
    
    # 1. Calculate r_1
    # r_1 = -1/4 ln(1-2*gamma_1^2)
    # Check for valid argument for log (1 - 2*gamma_1^2 > 0)
    log_arg = 1 - 2 * gamma_1_fixed**2
    if log_arg <= 0:
        raise ValueError(f"Invalid γ₁ = {gamma_1_fixed}. Must satisfy 1 - 2*γ₁^2 > 0.")
    
    r1 = -0.25 * np.log(log_arg)
    
    # 2. Define operators
    a = qt.tensor(qt.qeye(2), qt.destroy(N))
    
    # Qubit operators (in the combined space)
    sz = qt.tensor(qt.sigmaz(), qt.qeye(N))
    sx = qt.tensor(qt.sigmax(), qt.qeye(N))
    
    # Constants from r_1
    c2r = np.cosh(2 * r1)
    s2r = np.sinh(2 * r1)
    s2r_sq = np.sinh(r1)**2
    
    # 3. Calculate Hamiltonian components (H_QRM = H_osc + H_qubit + H_int)
    
    # Oscillator part of the QRM
    # $\Delta \left[ \frac{\omega}{\Delta} \right]_1 (\hat{b}^\dagger \hat{b} \cosh(2r_1) - \frac{1}{2} (\hat{b}^\dagger \hat{b}^\dagger + \hat{b} \hat{b}) \sinh(2r_1))$
    H_osc = Delta * (omega / Delta) * (
        a.dag() * a * c2r 
        - 0.5 * (a.dag() * a.dag() + a * a) * s2r
    )
    
    # Qubit part of the QRM (pure $\sigma_z$)
    # $\Delta \sigma_z$
    H_qubit = Delta * sz
    
    # Interaction part of the QRM
    # $\Delta \gamma_1 \sqrt{\frac{\omega}{\Delta}} e^{-r_1} (\hat{b} + \hat{b}^\dagger) \sigma_x$
    g_eff = Delta * gamma_1_fixed * np.sqrt(omega / Delta) * np.exp(-r1)
    H_int = g_eff * sx * (a + a.dag())
    
    # Constant shift (we include it for E0 calculation consistency, although it won't affect state or relative energy)
    H_const = Delta * (omega / Delta) * s2r_sq * qt.qeye([2, N])
    
    # 4. Total Hamiltonian
    H_QRM = H_osc + H_qubit + H_int + H_const
    
    # 5. Calculate ground state
    E0, psi0 = H_QRM.groundstate()
    return float(E0), psi0  

# Store results
E0_dict = {}
dE_dict = {}
Fidelity_dict = {}

# -----------------------------------------------
# LOOP TO CALCULATE ENERGY AND FIDELITY
# -----------------------------------------------

# NOTE: Since the new Hamiltonian depends on GAMMA_1_FIXED directly, we pass it. 
# The g_val is only used in the label for historical consistency.
for om, g_val in param_list:
    key = (om, g_val)
    
    E0_vals = []
    psi_prev = None  
    F_vals = [np.nan]  

    for N in N_list:
        # Calculate E0 and the state for cutoff N
        E0, psi_N = qrm_ground_state(N, om, Delta, GAMMA_1_FIXED) # *** NEW FUNCTION CALL ***
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
    label = fr'QRM: $\omega = {om}$, $\gamma_1 = \gamma_c$'
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
    label = fr'QRM: $\omega = {om}$, $\gamma_1 = \gamma_c$'
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