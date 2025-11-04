import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

# ----- Parameters -----
Delta = 1.0 # Fixed qubit frequency
N_FIXED = 100 # Fixed Fock cutoff 

# ----- w value to use  -----
omega_list = np.array([5e-5]) 

# ----- Gamma_1 sweep parameters -----
GAMMA_1_CRITICAL = 1/np.sqrt(2)
# We must stop just short of the critical point to avoid log(0)
epsilon = 1e-6 
gamma_1_sweep = np.linspace(0, GAMMA_1_CRITICAL - epsilon, 50) # 50 points for the sweep

# -----------------------------------------------
# H_QRM Hamiltonian in the b and b dagger base
# -----------------------------------------------
def qrm_ground_state_numerical(N, omega, Delta, gamma_1_fixed):
    """Calculates the NUMERICAL GS of H_QRM by diagonalizing."""
    
    # 1. Calculate r_1
    log_arg = 1 - 2 * gamma_1_fixed**2
    if log_arg <= 0:
        raise ValueError(f"Invalid γ₁ = {gamma_1_fixed}. Must satisfy 1 - 2*γ₁^2 > 0.")
    r1 = -0.25 * np.log(log_arg)
    
    # 2. Define operators
    a = qt.tensor(qt.qeye(2), qt.destroy(N))
    sz = qt.tensor(qt.sigmaz(), qt.qeye(N))
    sx = qt.tensor(qt.sigmax(), qt.qeye(N))
    
    # 3. Constants from r_1
    c2r = np.cosh(2 * r1)
    s2r = np.sinh(2 * r1)
    s2r_sq = np.sinh(r1)**2
    
    # 4. Hamiltonian components
    H_osc = Delta * (omega / Delta) * (
        a.dag() * a * c2r 
        - 0.5 * (a.dag() * a.dag() + a * a) * s2r
    )
    H_qubit = Delta * sz
    g_eff = Delta * gamma_1_fixed * np.sqrt(omega / Delta) * np.exp(-r1)
    H_int = g_eff * sx * (a + a.dag()) # Interaction with sigma_x
    H_const = Delta * (omega / Delta) * s2r_sq * qt.qeye([2, N])
    
    H_QRM = H_osc + H_qubit + H_int + H_const
    
    # 5. Calculate numerical ground state
    E0, psi0_num = H_QRM.groundstate()
    return float(E0), psi0_num  

# -----------------------------------------------
# Analytical State, obtained ground state ket from SW1
# -----------------------------------------------
def build_analytical_ket(N, omega, Delta, gamma_1):
    """Builds the ANALYTICAL KET from the provided formula."""

    # 1. Calculate r_1
    log_arg = 1 - 2 * gamma_1**2
    if log_arg <= 0:
        raise ValueError(f"Invalid γ₁ = {gamma_1}. Must satisfy 1 - 2*γ₁^2 > 0.")
    r1 = -0.25 * np.log(log_arg)

    # 2. Define base operators 
    a = qt.tensor(qt.qeye(2), qt.destroy(N))
    sy = qt.tensor(qt.sigmay(), qt.qeye(N)) 
    
    # 3. Initial state |↓⟩ ⊗ |0⟩
    # In QuTiP: |↑⟩ is basis(2,0), |↓⟩ is basis(2,1)
    spin_down = qt.basis(2, 1) 
    fock_0 = qt.basis(N, 0)
    psi_initial = qt.tensor(spin_down, fock_0)

    # 4. Build the exponent operator: A
    # A = -i/2 * C * (b+b†) * σ_y
    C = gamma_1 * np.sqrt(omega / Delta) * np.exp(-r1)
    A = -0.5j * C * (a.dag() + a) * sy

    # 5. Calculate the unitary U = exp(A)
    U = A.expm()

    # 6. Apply the unitary to the initial state
    psi_ana = U * psi_initial
    return psi_ana.unit() # Renormalize just in case

# Store results
Fidelity_dict = {}

# -----------------------------------------------
# LOOP TO CALCULATE FIDELITY
# -----------------------------------------------

print(f"Starting calculation with fixed N = {N_FIXED}")
print(f"Sweeping γ₁ from 0 to {GAMMA_1_CRITICAL - epsilon:.5f}")

for om in omega_list:
    key = om
    fidelities_for_this_omega = []
    print(f"  Calculating for ω = {om}...")
    
    for gamma_1 in gamma_1_sweep:
        
        try:
            # 1. Calculate Numerical GS of H_QRM 
            E_num, psi_numerical = qrm_ground_state_numerical(N_FIXED, om, Delta, gamma_1)
            
            # 2. Build Analytical GS 
            psi_analytical = build_analytical_ket(N_FIXED, om, Delta, gamma_1)
            
            # 3. Calculate Fidelity
            fidelity = abs(psi_numerical.overlap(psi_analytical))**2
            fidelities_for_this_omega.append(fidelity)
            
        except Exception as e:
            print(f"    Error at ω={om}, γ₁={gamma_1}: {e}")
            fidelities_for_this_omega.append(np.nan) 
            
    Fidelity_dict[key] = fidelities_for_this_omega

print("Calculation complete.")

# ----- Plot config -----
plt.rcParams.update({
    "figure.figsize": (7.0, 4.5),
    "font.size": 12,
    "font.family": "serif",
    "axes.labelsize": 14,
   "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
})

# ----- Figure: Fidelity(Numerical, Analytical) vs. gamma_1 -----
fig, ax = plt.subplots()

for i, om in enumerate(omega_list):
    if om in Fidelity_dict:
        label = fr'$\omega = {om}$'
        ax.plot(gamma_1_sweep, Fidelity_dict[om], 
                linestyle='-', # Only one line, so one style is fine
                marker='o', ms=4, 
                label=label)

ax.set_xlabel(r'Coupling $\gamma_1$')
ax.set_ylabel(r'Fidelity $\mathcal{F}(|\psi_{GS, Num}\rangle, |\psi_{GS, Ana}\rangle)$')


ax.text(0.95, 0.05, fr'$N = {N_FIXED}$', 
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes, 
        fontsize=12, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
# --------------------------------------------------

# Add vertical line for the critical point
ax.axvline(x=GAMMA_1_CRITICAL, color='red', linestyle='--', linewidth=1.5, label=r'$\gamma_c = 1/\sqrt{2}$')

ax.legend(frameon=False, title=r'$\omega$')
ax.set_ylim(-0.1, 1.1) # Full fidelity range
ax.set_xlim(0, GAMMA_1_CRITICAL + 0.02)
fig.tight_layout()
plt.show()