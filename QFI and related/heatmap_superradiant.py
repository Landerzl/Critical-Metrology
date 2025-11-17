import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from qutip import destroy, sigmax, sigmaz, qeye, tensor

# ====================================================================
# 1. PARAMETERS & ARRAYS
# ====================================================================

omega_list = np.array([0.5, 0.05, 0.005, 0.0005, 0.00005])

N = 60
Delta = 1.0

gamma_c = 1.0 / np.sqrt(2)
gamma_points = 150
gamma_vals = np.linspace(gamma_c, 2*gamma_c, gamma_points)

qfi_matrix = np.zeros((len(omega_list), gamma_points))

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
        g = gamma * sqrt(omega_val * Delta)  ⇒  gamma = g / sqrt(omega_val * Delta)

    H'_+ = ω a†a + Δ σ_z + ω α (a + a†)
           + g σ_x (a + a†) + 2 g α σ_x + ω α²
    with α = sqrt(gamma^2 - 1/(4 gamma^2)).
    """
    gamma = g / np.sqrt(omega_val * Delta)
    radicand = gamma**2 - 1.0 / (4.0 * gamma**2)
    alpha = np.sqrt(max(radicand, 0.0))

    H = (omega_val * num
         + Delta * sz
         + omega_val * alpha * (a + adag)
         + g * sx * (a + adag)
         + 2.0 * g * alpha * sx
         + omega_val * alpha**2 * Id)

    return H

def groundstate(g, omega_val):
    H = displaced_H(g, omega_val)
    evals, evecs = H.eigenstates()
    return evecs[0]

def dpsi_dg(g, omega_val, dg_val):
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
    dpsi, psi = dpsi_dg(g, omega_val, dg_val)
    overlap = (psi.dag() * dpsi)
    norm_dpsi2 = (dpsi.dag() * dpsi)
    return 4 * (norm_dpsi2 - np.abs(overlap)**2)

# ====================================================================
# 3. FILL MATRIX
# ====================================================================

for i, omega in enumerate(omega_list):
    domega = Delta * omega
    g_vals = gamma_vals * np.sqrt(domega)
    dg = 1e-6 * np.sqrt(domega)

    QFI_func_vec = np.vectorize(lambda g: QFI(g, omega, dg))
    qfi_matrix[i, :] = np.real(QFI_func_vec(g_vals))

# Log10 of QFI (avoid log(0))
log_qfi_matrix = np.log10(qfi_matrix + 1e-10)
log_qfi_matrix[np.isinf(log_qfi_matrix)] = np.max(log_qfi_matrix[np.isfinite(log_qfi_matrix)])

# ====================================================================
# 4. HEATMAP
# ====================================================================

plt.figure(figsize=(8, 4))

sns.set(font_scale=1.4)
ax = sns.heatmap(
    log_qfi_matrix,
    cmap="magma",
    xticklabels=False,
    yticklabels=[f'{o:.5f}' for o in omega_list],
    cbar_kws={'label': r'$\log_{10}(F_Q)$'}
)

# Custom x-ticks at gamma = 0 and gamma = gamma_c
plt.xticks([0, len(gamma_vals) - 1], [r"$\gamma_c$", r"2$\gamma_c$"], fontsize=16)

plt.xticks(rotation=0)
plt.yticks(rotation=0)

plt.xlabel(r'$\gamma$', fontsize=16)
plt.ylabel(r'$\omega/\Delta$', fontsize=16)

plt.tight_layout()
plt.show()
