import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from qutip import destroy, sigmax, sigmaz, qeye, tensor

# ====================================================================
# 1. PARAMETERS
# ====================================================================
N = 60
Delta = 1.0
gamma_c = 1.0 / np.sqrt(2)
gamma_points = 150
gamma_vals = np.linspace(gamma_c, 2*gamma_c, gamma_points)

# ====================================================================
# 2. OPERATORS & FUNCTIONS
# ====================================================================
b = tensor(qeye(2), destroy(N))
bdag = b.dag()
num = bdag * b
sx = tensor(sigmax(), qeye(N))
sz = tensor(sigmaz(), qeye(N))
Id = tensor(qeye(2), qeye(N))

def squeezed_H(g, omega_val):
    if omega_val * Delta == 0: gamma = 0
    else: gamma = g / np.sqrt(omega_val * Delta)
    
    radicand = gamma**2 - 1.0 / (4.0 * gamma**2)
    alpha_s = np.sqrt(max(radicand, 0.0))

    Tilde_Delta = np.sqrt(Delta**2 + (2.0 * g * alpha_s)**2)
    
    if Tilde_Delta == 0: cos2theta = 1.0; sin2theta = 0.0
    else: cos2theta = Delta / Tilde_Delta; sin2theta = (2.0 * g * alpha_s) / Tilde_Delta

    numerator_r = 4.0 * (g**2) * (Delta**2)
    denominator_r = omega_val * (Tilde_Delta**3)
    
    if denominator_r == 0: r = 0.0
    else: r = 0.25 * np.log(1.0 + numerator_r / denominator_r)

    cosh_2r = np.cosh(2 * r)
    sinh_2r = np.sinh(2 * r)
    exp_minus_r = np.exp(-r)

    H_osc = omega_val * cosh_2r * num - 0.5 * omega_val * sinh_2r * (b*b + bdag*bdag)
    term_bracket = omega_val * alpha_s * Id + g * (cos2theta * sx - sin2theta * sz)
    H_int = exp_minus_r * (b + bdag) * term_bracket
    H_qubit = Tilde_Delta * sz
    return H_osc + H_int + H_qubit

def groundstate(g, omega_val):
    _, psi0 = squeezed_H(g, omega_val).groundstate()
    return psi0

def QFI(g, omega_val, dg_val):
    psi_0 = groundstate(g, omega_val)
    psi_p = groundstate(g + dg_val, omega_val)
    psi_m = groundstate(g - dg_val, omega_val)

    phase_p = (psi_0.dag() * psi_p)
    phase_m = (psi_0.dag() * psi_m)

    
    psi_p = psi_p * np.exp(-1j * np.angle(phase_p))
    psi_m = psi_m * np.exp(-1j * np.angle(phase_m))

    dpsi = (psi_p - psi_m) / (2 * dg_val)
    overlap = (psi_0.dag() * dpsi)
    norm_dpsi2 = (dpsi.dag() * dpsi)
    

    return np.real(4 * (norm_dpsi2 - np.abs(overlap)**2))

# ====================================================================
# 3. EXACT 3D PLOT GENERATION
# ====================================================================

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# --- Define EXACT target values ---
target_omegas = [5e-1, 5e-2, 5e-3, 5e-4, 5e-5]

colors = ['#000080', '#4169E1', '#2E8B57', '#FFA500', '#DC143C']

print("Calculating Exact Lines...")

for i, omega_exact in enumerate(target_omegas):
    print(f"Processing omega = {omega_exact:.1e} ...")
    
    # 1. Recalculate parameters specifically for this exact omega
    domega = Delta * omega_exact
    g_vals = gamma_vals * np.sqrt(domega)
    dg = 1e-6 * np.sqrt(domega)
    
    # 2. Compute QFI vector for this specific omega
    QFI_func_vec = np.vectorize(lambda g: np.real(QFI(g, omega_exact, dg)))
    qfi_line = QFI_func_vec(g_vals)
    
    # 3. Logarithms for plotting
    log_qfi_line = np.log10(qfi_line + 1e-10)
    
    finite_vals = log_qfi_line[np.isfinite(log_qfi_line)]
    if len(finite_vals) > 0:
        max_val = np.max(finite_vals)
        log_qfi_line[np.isinf(log_qfi_line)] = max_val
    
    # 4. Prepare 3D coordinates
    xs = gamma_vals
    y_val = np.log10(omega_exact) # Position on Y axis
    ys = np.full_like(xs, y_val)
    zs = log_qfi_line
    
    # 5. Plot
    ax.plot(xs, ys, zs, color=colors[i], linewidth=2, label=f'$\\omega = {omega_exact:.0e}$')


# --- Plot Adjustments ---
ax.set_xlabel(r'$\gamma$', fontsize=14, labelpad=10)
ax.set_ylabel(r'$\log_{10}(\omega)$', fontsize=14, labelpad=10)
ax.set_zlabel(r'$\log_{10}(F_Q)$', fontsize=14, labelpad=10)

# Set Y-ticks exactly at the target locations
yticks_vals = np.log10(target_omegas)
yticks_labels = [r'$5 \cdot 10^{-1}$', r'$5 \cdot 10^{-2}$', r'$5 \cdot 10^{-3}$', r'$5 \cdot 10^{-4}$', r'$5 \cdot 10^{-5}$']

ax.set_yticks(yticks_vals)
ax.set_yticklabels(yticks_labels, fontsize=10, rotation=-15)

ax.set_xticks([gamma_c, 2*gamma_c])
ax.set_xticklabels([r'$\gamma_c$', r'$2\gamma_c$'], fontsize=12)

ax.view_init(elev=30, azim=-60)

# Legend adjustments
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 0.9))
plt.tight_layout()
plt.show()