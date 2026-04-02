import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from qutip import destroy, sigmax, sigmaz, qeye, tensor
# ====================================================================
# 1. SHARED PARAMETERS
# ====================================================================
num_omega = 50
omega_list = np.logspace(np.log10(5e-5), np.log10(0.5), num_omega)
omega_list = np.flip(omega_list)  # Largest omega at the top
N = 60
Delta = 1.0
gamma_c = 1.0 / np.sqrt(2)
gamma_points = 150
# Gamma ranges for each phase
gamma_vals_normal = np.linspace(0, gamma_c, gamma_points)           # Normal phase
gamma_vals_super  = np.linspace(gamma_c, 2*gamma_c, gamma_points)   # Superradiant phase
# QFI matrices
qfi_matrix_normal = np.zeros((num_omega, gamma_points))
qfi_matrix_super  = np.zeros((num_omega, gamma_points))
# ====================================================================
# 2A. OPERATORS — NORMAL PHASE (a, a†)
# ====================================================================
a = destroy(N)
adag = a.dag()
I_q = qeye(2)
I_b = qeye(N)
sx_q = sigmax()
sz_q = sigmaz()
def H_rabi(g, omega_val):
    H0   = omega_val * tensor(adag * a, I_q)
    H1   = Delta * tensor(I_b, sz_q)
    Hint = g * tensor(a + adag, sx_q)
    return H0 + H1 + Hint
def groundstate_normal(g, omega_val):
    H = H_rabi(g, omega_val)
    _, psi0 = H.groundstate()
    return psi0
def dpsi_dg_normal(g, omega_val, dg_val):
    psi_p = groundstate_normal(g + dg_val, omega_val)
    psi_m = groundstate_normal(g - dg_val, omega_val)
    psi_0 = groundstate_normal(g, omega_val)
    phase_p = (psi_0.dag() * psi_p)
    phase_m = (psi_0.dag() * psi_m)
    psi_p = psi_p * np.exp(-1j * np.angle(phase_p))
    psi_m = psi_m * np.exp(-1j * np.angle(phase_m))
    return (psi_p - psi_m) / (2 * dg_val), psi_0
def QFI_normal(g, omega_val, dg_val):
    dpsi, psi = dpsi_dg_normal(g, omega_val, dg_val)
    overlap    = (psi.dag() * dpsi)
    norm_dpsi2 = (dpsi.dag() * dpsi)
    return 4 * (norm_dpsi2 - np.abs(overlap)**2)
# ====================================================================
# 2B. OPERATORS — SUPERRADIANT PHASE (b, b† squeezed frame)
# ====================================================================
b    = tensor(qeye(2), destroy(N))
bdag = b.dag()
num_op = bdag * b
sx_s = tensor(sigmax(), qeye(N))
sz_s = tensor(sigmaz(), qeye(N))
Id_s = tensor(qeye(2), qeye(N))
def squeezed_H(g, omega_val):
    if omega_val * Delta == 0:
        gamma = 0
    else:
        gamma = g / np.sqrt(omega_val * Delta)
    radicand = gamma**2 - 1.0 / (4.0 * gamma**2)
    alpha_s  = np.sqrt(max(radicand, 0.0))
    Tilde_Delta = np.sqrt(Delta**2 + (2.0 * g * alpha_s)**2)
    if Tilde_Delta == 0:
        cos2theta = 1.0
        sin2theta = 0.0
    else:
        cos2theta = Delta / Tilde_Delta
        sin2theta = (2.0 * g * alpha_s) / Tilde_Delta
    numerator_r   = 4.0 * (g**2) * (Delta**2)
    denominator_r = omega_val * (Tilde_Delta**3)
    if denominator_r == 0:
        r = 0.0
    else:
        val_inside_log = 1.0 + numerator_r / denominator_r
        r = 0.25 * np.log(val_inside_log)
    cosh_2r     = np.cosh(2 * r)
    sinh_2r     = np.sinh(2 * r)
    exp_minus_r = np.exp(-r)
    H_osc   = omega_val * cosh_2r * num_op - 0.5 * omega_val * sinh_2r * (b*b + bdag*bdag)
    term_bracket = omega_val * alpha_s * Id_s + g * (cos2theta * sx_s - sin2theta * sz_s)
    H_int   = exp_minus_r * (b + bdag) * term_bracket
    H_qubit = Tilde_Delta * sz_s
    return H_osc + H_int + H_qubit
def groundstate_super(g, omega_val):
    H = squeezed_H(g, omega_val)
    _, psi0 = H.groundstate()
    return psi0
def dpsi_dg_super(g, omega_val, dg_val):
    psi_p = groundstate_super(g + dg_val, omega_val)
    psi_m = groundstate_super(g - dg_val, omega_val)
    psi_0 = groundstate_super(g, omega_val)
    phase_p = (psi_0.dag() * psi_p)
    phase_m = (psi_0.dag() * psi_m)
    psi_p = psi_p * np.exp(-1j * np.angle(phase_p))
    psi_m = psi_m * np.exp(-1j * np.angle(phase_m))
    return (psi_p - psi_m) / (2 * dg_val), psi_0
def QFI_super(g, omega_val, dg_val):
    dpsi, psi = dpsi_dg_super(g, omega_val, dg_val)
    overlap    = (psi.dag() * dpsi)
    norm_dpsi2 = (dpsi.dag() * dpsi)
    return 4 * (norm_dpsi2 - np.abs(overlap)**2)
# ====================================================================
# 3. FILL MATRICES
# ====================================================================
# --- Normal Phase ---
print(f"Starting NORMAL PHASE calculation ({num_omega} bands)...")
for i, omega in enumerate(omega_list):
    domega = Delta * omega
    g_vals = gamma_vals_normal * np.sqrt(domega)
    dg = 1e-6 * np.sqrt(domega)
    QFI_vec = np.vectorize(lambda g: np.real(QFI_normal(g, omega, dg)))
    qfi_matrix_normal[i, :] = QFI_vec(g_vals)
    if (i+1) % 5 == 0:
        print(f"  Normal band {i+1}/{num_omega} (omega={omega:.2e}) done.")
# --- Superradiant Phase ---
print(f"\nStarting SUPERRADIANT PHASE calculation ({num_omega} bands)...")
for i, omega in enumerate(omega_list):
    domega = Delta * omega
    g_vals = gamma_vals_super * np.sqrt(domega)
    dg = 1e-6 * np.sqrt(domega)
    QFI_vec = np.vectorize(lambda g: np.real(QFI_super(g, omega, dg)))
    qfi_matrix_super[i, :] = QFI_vec(g_vals)
    if (i+1) % 5 == 0:
        print(f"  Super band {i+1}/{num_omega} (omega={omega:.2e}) done.")
# Log10 transform
log_qfi_normal = np.log10(qfi_matrix_normal + 1e-10)
log_qfi_super  = np.log10(qfi_matrix_super  + 1e-10)
# Cap infinities
for mat in [log_qfi_normal, log_qfi_super]:
    finite_max = np.max(mat[np.isfinite(mat)])
    mat[np.isinf(mat)] = finite_max
# ====================================================================
# 4. COMBINED HEATMAP — SINGLE CONTINUOUS FIGURE
# ====================================================================
# Concatenate both matrices horizontally: Normal | Superradiant
log_qfi_combined = np.hstack([log_qfi_normal, log_qfi_super])
sns.set(font_scale=1.2)
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(
    log_qfi_combined,
    cmap="magma_r",
    xticklabels=False,
    yticklabels=True,
    cbar_kws={'label': r'$\log_{10}(F_Q)$'},
    vmax=8,
    ax=ax
)
# --- Y-axis ---
y_ticks = [0.5, num_omega - 0.5]
y_labels = [r'$5 \cdot 10^{-1}$', r'$5 \cdot 10^{-5}$']
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels, fontsize=16)
plt.yticks(rotation=0)
# --- X-axis: 0, gamma_c, 2*gamma_c ---
total_cols = 2 * gamma_points
ax.set_xticks([0, gamma_points, total_cols - 1])
ax.set_xticklabels([r"$0$", r"$\gamma_c$", r"$2\gamma_c$"], fontsize=16, rotation=0)
ax.set_xlabel(r'$\gamma$', fontsize=16)
ax.set_ylabel(r'$\omega/\Delta$ (Log Scale)', fontsize=16, labelpad=-20)
plt.tight_layout()
plt.show()
