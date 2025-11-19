import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from qutip import destroy, sigmax, sigmaz, qeye, tensor

# ====================================================================
# 1. PARAMETERS & ARRAYS (Updated for 50 log-spaced omega values)
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
# Gamma range: Superradiant phase [gamma_c, 2*gamma_c]
gamma_vals = np.linspace(gamma_c, 2*gamma_c, gamma_points)

# The matrix now has 50 rows
qfi_matrix = np.zeros((num_omega, gamma_points))

# ====================================================================
# 2. OPERATORS AND HELPERS (displaced Rabi, qubit ⊗ boson)
# ====================================================================

# Qubit ⊗ boson operators
a = tensor(qeye(2), destroy(N))
adag = a.dag()
num = adag * a
sx = tensor(sigmax(), qeye(N))
sz = tensor(sigmaz(), qeye(N))
Id = tensor(qeye(2), qeye(N))

def displaced_H(g, omega_val):
    """
    Displaced Rabi Hamiltonian H'_+ for given g and omega_val.

    Relation between g and gamma:
        g = gamma * sqrt(omega_val * Delta)  =>  gamma = g / sqrt(omega_val * Delta)

    H'_+ = ω a†a + Δ σ_z + ω α (a + a†)
           + g σ_x (a + a†) + 2 g α σ_x + ω α²
    with α = sqrt(gamma^2 - 1/(4 gamma^2)).
    """
    # Safety check for potential division by zero when omega_val is extremely small
    if omega_val * Delta == 0:
        gamma = 0
    else:
        gamma = g / np.sqrt(omega_val * Delta)
        
    radicand = gamma**2 - 1.0 / (4.0 * gamma**2)
    # Since gamma >= gamma_c, alpha should be real and positive (Superradiant phase)
    alpha = np.sqrt(max(radicand, 0.0))

    H = (omega_val * num
         + Delta * sz
         + omega_val * alpha * (a + adag)
         + g * sx * (a + adag)
         + 2.0 * g * alpha * sx
         + omega_val * alpha**2 * Id)

    return H

def groundstate(g, omega_val):
    """Calculates the ground state for given g and omega."""
    H = displaced_H(g, omega_val)
    # Use .groundstate() for efficient calculation
    _, psi0 = H.groundstate()
    return psi0

def dpsi_dg(g, omega_val, dg_val):
    """Calculates the centered derivative of the ground state with respect to g."""
    psi_p = groundstate(g + dg_val, omega_val)
    psi_m = groundstate(g - dg_val, omega_val)
    psi_0 = groundstate(g, omega_val)

    # Phase correction (important for numerical derivative)
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
    # g_vals corresponds to gamma_vals in the range [gamma_c, 2*gamma_c]
    g_vals = gamma_vals * np.sqrt(domega)
    # dg depends on omega
    dg = 1e-6 * np.sqrt(domega)

    # Vectorize the QFI calculation over g_vals for speed
    QFI_func_vec = np.vectorize(lambda g: QFI(g, omega, dg))
    qfi_matrix[i, :] = np.real(QFI_func_vec(g_vals))
    print(f"Band {i+1}/{num_omega} (omega={omega:.2e}) calculated.")

# Log10 of QFI (avoid log(0))
log_qfi_matrix = np.log10(qfi_matrix + 1e-10)
# Ensure no infinities (if a value is 0)
finite_max = np.max(log_qfi_matrix[np.isfinite(log_qfi_matrix)])
log_qfi_matrix[np.isinf(log_qfi_matrix)] = finite_max

# ====================================================================
# 4. HEATMAP (Aesthetics improved)
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

# --- Y-axis Ticks/Labels (Modified for 50 bands, LaTeX, and increased size) ---
# Calculate the y-coordinates for the center of the first and last rows
y_ticks = [0.5, num_omega - 0.5]

# Custom labels using LaTeX math notation: 
# omega_list[0] = 0.5 (or 5 * 10^-1)
# omega_list[-1] = 5e-5 (or 5 * 10^-5)
y_labels_latex = [r'$5 \cdot 10^{-1}$', r'$5 \cdot 10^{-5}$'] 

ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels_latex, fontsize=16) 
# -----------------------------------------------------------------------------

# X-axis Ticks (Modified to show gamma_c and 2*gamma_c)
plt.xticks([0, gamma_points - 1], [r"$\gamma_c$", r"$2\gamma_c$"], fontsize=16)

plt.xticks(rotation=0)
plt.yticks(rotation=0)

plt.xlabel(r'$\gamma$',fontsize=16)
# Use negative labelpad to bring the y-axis label closer to the tick marks
plt.ylabel(r'$\omega/\Delta$ (Log Scale)', fontsize=16, labelpad=-20) 

plt.tight_layout()
plt.show()