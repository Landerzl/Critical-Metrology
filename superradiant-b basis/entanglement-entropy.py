import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# ====================================================================
# 0. CONFIGURATION AND PHYSICAL PARAMETERS
# ====================================================================
omega = 5e-5     # Oscillator frequency
Delta = 1.0       # Qubit parameter (Coefficient of sigma_z)
N_normal = 50     # Fock cutoff for Normal Phase
N_squeezed = 50   # Fock cutoff for Superradiant Phase

# Theoretical Critical Point
g_c = np.sqrt(Delta * omega / 2.0)

# ====================================================================
# 1. NORMAL PHASE HAMILTONIAN (g < g_c)
# Tensor Order: [Boson, Spin]
# ====================================================================
def get_groundstate_normal(g, N, omega_val, Delta_val):
    a = destroy(N)
    adag = a.dag()
    I_q = qeye(2)
    I_b = qeye(N)
    sx = sigmax()
    sz = sigmaz()
    
    H0 = omega_val * tensor(adag * a, I_q)
    H1 = Delta_val * tensor(I_b, sz)  
    Hint = g * tensor(a + adag, sx)
    
    H = H0 + H1 + Hint
    
    E, states = H.eigenstates()
    return states[0]

# ====================================================================
# 2. SUPERRADIANT PHASE HAMILTONIAN (g > g_c)
# Tensor Order: [Spin, Boson]
# ====================================================================
def get_groundstate_squeezed(g, N, omega_val, Delta_val):
    b = tensor(qeye(2), destroy(N))
    bdag = b.dag()
    num = bdag * b
    sx = tensor(sigmax(), qeye(N))
    sz = tensor(sigmaz(), qeye(N))
    Id = tensor(qeye(2), qeye(N))
    
    gamma_param = g / np.sqrt(omega_val * Delta_val)
    
    radicand = gamma_param**2 - 1.0 / (4.0 * gamma_param**2)
    alpha_s = np.sqrt(max(radicand, 0.0))

    Tilde_Delta = np.sqrt(Delta_val**2 + (2.0 * g * alpha_s)**2)
    
    if Tilde_Delta == 0:
        cos2theta, sin2theta = 1.0, 0.0
    else:
        cos2theta = Delta_val / Tilde_Delta
        sin2theta = (2.0 * g * alpha_s) / Tilde_Delta

    numerator_r = 4.0 * (g**2) * (Delta_val**2)
    denominator_r = omega_val * (Tilde_Delta**3)
    
    if denominator_r == 0:
        r = 0.0
    else:
        val_inside_log = 1.0 + numerator_r / denominator_r
        r = 0.25 * np.log(val_inside_log)

    cosh_2r = np.cosh(2 * r)
    sinh_2r = np.sinh(2 * r)
    exp_minus_r = np.exp(-r)

    H_osc = omega_val * cosh_2r * num - 0.5 * omega_val * sinh_2r * (b*b + bdag*bdag)
    term_bracket = omega_val * alpha_s * Id + g * (cos2theta * sx - sin2theta * sz)
    H_int = exp_minus_r * (b + bdag) * term_bracket
    H_qubit = Tilde_Delta * sz 
    
    H_total = H_osc + H_int + H_qubit
    
    E, states = H_total.eigenstates()
    return states[0]

# ====================================================================
# 3. MAIN LOOP (MODIFIED)
# ====================================================================

# Sweep normalized coupling lambda = g / g_c
gamma_values = np.linspace(0.0, 2, 100) 

# Split arrays to avoid connecting lines across the critical point
gamma_normal = gamma_values[gamma_values < 1.0]
gamma_super  = gamma_values[gamma_values >= 1.0]

entropy_normal = []
entropy_super = []

# --- Loop 1: Normal Phase ---
for val in gamma_normal:
    g_val = val * g_c
    psi = get_groundstate_normal(g_val, N_normal, omega, Delta)
    rho_spin = psi.ptrace(1) 
    entropy_normal.append(entropy_vn(rho_spin, base=2))

# --- Loop 2: Superradiant Phase ---
for val in gamma_super:
    g_val = val * g_c
    psi = get_groundstate_squeezed(g_val, N_squeezed, omega, Delta)
    rho_spin = psi.ptrace(0) 
    entropy_super.append(entropy_vn(rho_spin, base=2))

# ====================================================================
# 4. VISUALIZATION (MODIFIED)
# ====================================================================
plt.figure(figsize=(9, 6))

# Normal Phase: Solid Line + Markers
plt.plot(gamma_normal, entropy_normal, '-', linewidth=2, markersize=4, 
         label='Normal Phase (Solid)')

# Superradiant Phase: Dashed Line + Markers
plt.plot(gamma_super, entropy_super, '--', linewidth=2, markersize=4, 
         label='Superradiant Phase (Dashed)')

# Critical Point
plt.axvline(x=1.0, color='r', linestyle=':', linewidth=1.5, 
            label=r'Critical Point ($\gamma = \gamma_c$)')

plt.xlabel(r'$\gamma/\gamma_c$', fontsize=12)
plt.ylabel(r'Von Neumann Entropy $S_{VN}$', fontsize=12)

plt.legend(fontsize=11)
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()

plt.show()