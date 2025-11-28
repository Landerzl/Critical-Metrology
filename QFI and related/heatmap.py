import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from qutip import destroy, sigmax, sigmaz, qeye, tensor

# ====================================================================
# 1. PARAMETERS & ARRAYS
# ====================================================================

num_omega = 50
# Generating 50 logarithmically spaced values for omega/Delta
# Range: [5e-5, 0.5]
omega_list = np.logspace(np.log10(5e-5), np.log10(0.5), num_omega)
omega_list = np.flip(omega_list) # Invert to ensure largest omega is at the top of the heatmap

N = 60
Delta = 1.0

gamma_c = 1.0 / np.sqrt(2)
gamma_points = 150
gamma_vals = np.linspace(0, gamma_c, gamma_points)

# The matrix now has 50 rows
qfi_matrix = np.zeros((num_omega, gamma_points))

# ====================================================================
# 2. OPERATORS AND HELPERS
# ====================================================================

# Bosonic operators
a = destroy(N)
adag = a.dag()

# Identities
I_q = qeye(2)
I_b = qeye(N)

# Qubit operators
sx = sigmax()
sz = sigmaz()

def H_rabi(g, omega_val):
    """Rabi Hamiltonian with the given oscillator frequency."""
    # H0: Boson (tensor(Boson, Qubit))
    H0 = omega_val * tensor(adag * a, I_q)
    # H1: Qubit (tensor(Boson, Qubit))
    H1 = Delta * tensor(I_b, sz)
    # Hint: Interaction (tensor(Boson, Qubit))
    Hint = g * tensor(a + adag, sx)
    return H0 + H1 + Hint

def groundstate(g, omega_val):
    """Calculates the ground state for given g and omega."""
    H = H_rabi(g, omega_val)
    # Use .groundstate() as a faster alternative to .eigenstates()[0]
    _, psi0 = H.groundstate() 
    return psi0

def dpsi_dg(g, omega_val, dg_val):
    """Calculates the centered derivative of the ground state with respect to g."""
    psi_p = groundstate(g + dg_val, omega_val)
    psi_m = groundstate(g - dg_val, omega_val)
    psi_0 = groundstate(g, omega_val)

    # Phase correction: align phase with psi_0
    phase_p = (psi_0.dag() * psi_p)
    phase_m = (psi_0.dag() * psi_m)
    psi_p = psi_p * np.exp(-1j * np.angle(phase_p))
    psi_m = psi_m * np.exp(-1j * np.angle(phase_m))

    return (psi_p - psi_m) / (2 * dg_val), psi_0

def QFI(g, omega_val, dg_val):
    """Calculates the Quantum Fisher Information (QFI)."""
    dpsi, psi = dpsi_dg(g, omega_val, dg_val)
    overlap = (psi.dag() * dpsi)
    norm_dpsi2 = (dpsi.dag() * dpsi)
    # F_Q = 4 * (||dpsi||^2 - |<psi | dpsi>|^2)
    return 4 * (norm_dpsi2 - np.abs(overlap)**2)

# ====================================================================
# 3. FILL MATRIX (Calculation)
# ====================================================================

print(f"Starting calculation for {num_omega} bands...")
for i, omega in enumerate(omega_list):
    domega = Delta * omega
    g_vals = gamma_vals * np.sqrt(domega)
    # dg depends on omega
    dg = 1e-6 * np.sqrt(domega)

    # Vectorize the QFI calculation over g_vals for speed
    QFI_func_vec = np.vectorize(lambda g: QFI(g, omega, dg))
    qfi_matrix[i, :] = np.real(QFI_func_vec(g_vals))
    print(f"Band {i+1}/{num_omega} (omega={omega:.2e}) calculated.")

# Logarithmic transformation for the heatmap
log_qfi_matrix = np.log10(qfi_matrix + 1e-10) 
# Ensure no infinities (if a value is 0)
finite_max = np.max(log_qfi_matrix[np.isfinite(log_qfi_matrix)])
log_qfi_matrix[np.isinf(log_qfi_matrix)] = finite_max

# ====================================================================
# 4. HEATMAP
# ====================================================================

plt.figure(figsize=(8, 6)) 

sns.set(font_scale=1.2)
ax = sns.heatmap(
    log_qfi_matrix,
    cmap="magma",
    xticklabels=False,        
    yticklabels=True, 
    cbar_kws={'label': r'$\log_{10}(F_Q)$'}
)

# --- Y-axis Ticks/Labels---
# Calculate the y-coordinates for the center of the first and last rows
y_ticks = [0.5, num_omega - 0.5]

y_labels_latex = [r'$5 \cdot 10^{-1}$', r'$5 \cdot 10^{-5}$'] 

ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels_latex, fontsize=16) # Font size increased here
# ---------------------------------

# X-axis Ticks
cols_for_ticks = np.linspace(0, gamma_points - 1, 2).astype(int)
plt.xticks([0, gamma_points - 1], [r"$0$", r"$\gamma_c$"], fontsize=16)

plt.xticks(rotation=0)
plt.yticks(rotation=0)

plt.xlabel(r'$\gamma$',fontsize=16)
plt.ylabel(r'$\omega/\Delta$ (Log Scale)',fontsize=16, labelpad=-20)
plt.tight_layout()
plt.show()
