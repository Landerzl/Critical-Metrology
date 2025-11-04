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
gamma_vals = np.linspace(0, gamma_c, gamma_points)

qfi_matrix = np.zeros((len(omega_list), gamma_points))

# ====================================================================
# 2. OPERATORS AND HELPERS
# ====================================================================

a = destroy(N)
adag = a.dag()
I_q = qeye(2)
I_b = qeye(N)
sx = sigmax()
sz = sigmaz()

def H_rabi(g, omega_val):
    H0 = omega_val * tensor(adag * a, I_q)
    H1 = Delta * tensor(I_b, sz)
    Hint = g * tensor(a + adag, sx)
    return H0 + H1 + Hint

def groundstate(g, omega_val):
    H = H_rabi(g, omega_val)
    evals, evecs = H.eigenstates()
    return evecs[0]

def dpsi_dg(g, omega_val, dg_val):
    psi_p = groundstate(g + dg_val, omega_val)
    psi_m = groundstate(g - dg_val, omega_val)
    psi_0 = groundstate(g, omega_val)

    # phase correction, as was done in the normal QFI.
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
# 3. FILL MATRIX (we build a matrix to get data more accessible)
# ====================================================================

for i, omega in enumerate(omega_list):
    domega = Delta * omega
    g_vals = gamma_vals * np.sqrt(domega)
    dg = 1e-6 * np.sqrt(domega)

    QFI_func_vec = np.vectorize(lambda g: QFI(g, omega, dg))
    qfi_matrix[i, :] = np.real(QFI_func_vec(g_vals))

log_qfi_matrix = np.log10(qfi_matrix + 1e-10) #just in case there's a 0 value.
log_qfi_matrix[np.isinf(log_qfi_matrix)] = np.max(log_qfi_matrix[np.isfinite(log_qfi_matrix)])

# ====================================================================
# 4. HEATMAP
# ====================================================================

plt.figure(figsize=(8,4))

sns.set(font_scale=1.4)
ax = sns.heatmap(
    log_qfi_matrix,
    cmap="magma",
    xticklabels=False,       
    yticklabels=[f'{o:.5f}' for o in omega_list],
    cbar_kws={'label': r'$\log_{10}(F_Q)$'}
)


num_xticks = 2 #Had some problems with the x axis using seaborn. This choice solved it.
cols_for_ticks = np.linspace(0, gamma_points-1, num_xticks).astype(int)

plt.xticks([0, len(gamma_vals)-1], [r"$0$", r"$\gamma_c$"], fontsize=16)


plt.xticks(rotation=0)
plt.yticks(rotation=0)

plt.xlabel(r'$\gamma$',fontsize=16)
plt.ylabel(r'$\omega/\Delta$',fontsize=16)


plt.tight_layout()
plt.show()
