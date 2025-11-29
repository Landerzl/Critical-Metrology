import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from qutip import destroy, sigmax, sigmaz, qeye, tensor

# ====================================================================
# 1. PARAMETERS & ARRAYS
# ====================================================================

num_omega = 50
# Generating 50 logarithmically spaced values for omega
# Range: [5e-5, 0.5]
omega_list = np.logspace(np.log10(5e-5), np.log10(0.5), num_omega)
omega_list = np.flip(omega_list)  # Largest omega at the top

N = 60
Delta = 1.0

gamma_c = 1.0 / np.sqrt(2)
gamma_points = 150
# Gamma range: Superradiant phase [gamma_c, 2*gamma_c]
gamma_vals = np.linspace(gamma_c, 2*gamma_c, gamma_points)

# The matrix to store QFI data
qfi_matrix = np.zeros((num_omega, gamma_points))

# ====================================================================
# 2. OPERATORS (Squeezed Frame b, b^dag)
# ====================================================================
# Define 'b' and derived operators.
# Note: sx and sz act here as tau_x and tau_z in the rotated frame.
b = tensor(qeye(2), destroy(N))
bdag = b.dag()
num = bdag * b
sx = tensor(sigmax(), qeye(N))
sz = tensor(sigmaz(), qeye(N))
Id = tensor(qeye(2), qeye(N))

def squeezed_H(g, omega_val):
    """
    Hamiltonian H_+ in the b, b^dag basis including squeezing r.
    """
    # 1. Recover gamma
    if omega_val * Delta == 0:
        gamma = 0
    else:
        gamma = g / np.sqrt(omega_val * Delta)
        
    # 2. Calculate alpha_s (Superradiant displacement)
    radicand = gamma**2 - 1.0 / (4.0 * gamma**2)
    alpha_s = np.sqrt(max(radicand, 0.0))

    # 3. Calculate Tilde_Delta and angles (Qubit diagonalization)
    # Original qubit term: Delta*sigma_z + 2*g*alpha*sigma_x
    Tilde_Delta = np.sqrt(Delta**2 + (2.0 * g * alpha_s)**2)
    
    if Tilde_Delta == 0:
        cos2theta = 1.0
        sin2theta = 0.0
    else:
        cos2theta = Delta / Tilde_Delta
        sin2theta = (2.0 * g * alpha_s) / Tilde_Delta

    # 4. Calculate Squeezing Parameter r
    # Formula: r = 1/4 * ln( 1 + (4 g^2 Delta^2) / (omega * Tilde_Delta^3) )
    numerator_r = 4.0 * (g**2) * (Delta**2)
    denominator_r = omega_val * (Tilde_Delta**3)
    
    if denominator_r == 0:
        r = 0.0
    else:
        val_inside_log = 1.0 + numerator_r / denominator_r
        r = 0.25 * np.log(val_inside_log)

    # Hyperbolic functions
    cosh_2r = np.cosh(2 * r)
    sinh_2r = np.sinh(2 * r)
    exp_minus_r = np.exp(-r)

    # 5. Construct Hamiltonian H_+
    # H_+ = w * cosh(2r) b^dag b - (w/2) sinh(2r) (b^2 + b^dag^2)
    #       + e^{-r} (b + b^dag) [ w * alpha_s + g( cos(2theta)tau_x - sin(2theta)tau_z ) ]
    #       + Tilde_Delta * tau_z
    
    # Oscillator terms (Squeezed)
    H_osc = omega_val * cosh_2r * num - 0.5 * omega_val * sinh_2r * (b*b + bdag*bdag)
    
    # Linear interaction term (Displaced)
    # Note: Using sx for tau_x and sz for tau_z
    term_bracket = omega_val * alpha_s * Id + g * (cos2theta * sx - sin2theta * sz)
    H_int = exp_minus_r * (b + bdag) * term_bracket
    
    # Qubit energy term
    H_qubit = Tilde_Delta * sz
    
    return H_osc + H_int + H_qubit

def groundstate(g, omega_val):
    """Calculates the ground state using the new H_+."""
    H = squeezed_H(g, omega_val)
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

print(f"Starting calculation for {num_omega} bands using Squeezed H_+...")
for i, omega in enumerate(omega_list):
    domega = Delta * omega
    g_vals = gamma_vals * np.sqrt(domega)
    dg = 1e-6 * np.sqrt(domega)

    # Vectorize the calculation
    QFI_func_vec = np.vectorize(lambda g: np.real(QFI(g, omega, dg)))
    qfi_matrix[i, :] = QFI_func_vec(g_vals)
    
    # Optional print to track progress
    if (i+1) % 5 == 0:
        print(f"Band {i+1}/{num_omega} (omega={omega:.2e}) calculated.")

# Log10 of QFI (avoid log(0))
log_qfi_matrix = np.log10(qfi_matrix + 1e-10)
# Handle infinite values if any
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
    cbar_kws={'label': r'$\log_{10}(F_Q)$'},
    vmax=8
)

# --- Y-axis Ticks/Labels ---
y_ticks = [0.5, num_omega - 0.5]
y_labels_latex = [r'$5 \cdot 10^{-1}$', r'$5 \cdot 10^{-5}$'] 

ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels_latex, fontsize=16) 

# --- X-axis Ticks ---
plt.xticks([0, gamma_points - 1], [r"$\gamma_c$", r"$2\gamma_c$"], fontsize=16)

plt.xticks(rotation=0)
plt.yticks(rotation=0)

plt.xlabel(r'$\gamma$',fontsize=16)
plt.ylabel(r'$\omega/\Delta$ (Log Scale)', fontsize=16, labelpad=-20) 

plt.tight_layout()
plt.show()