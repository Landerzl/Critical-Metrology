import numpy as np
import matplotlib.pyplot as plt
from qutip import destroy, sigmax, sigmaz, qeye, tensor

# --- Parameters ---
N = 60       # Bosonic truncation
Delta = 1.0  # Qubit splitting
omega = 5e-5 # Oscillator frequency
domega = Delta * omega

# Critical parameter
gamma_c = 1.0 / np.sqrt(2) 

# --- User defined points ---
num_points = 100
min_delta_gamma = 1e-4 # Closest approach to the critical point
delta_gamma_vals = np.geomspace(min_delta_gamma, gamma_c, num_points)
gamma_vals = gamma_c - delta_gamma_vals 

g_vals = gamma_vals * np.sqrt(domega)
dg = 1e-6 * np.sqrt(domega) 

# --- Operators ---
a = destroy(N)
adag = a.dag()
I_q = qeye(2)
I_b = qeye(N)
sx = sigmax()
sz = sigmaz()

# General Rabi Hamiltonian
def H_rabi(g):
    H0 = omega * tensor(adag * a, I_q)
    H1 = Delta * tensor(I_b, sz)
    Hint = g * tensor(a + adag, sx)
    return H0 + H1 + Hint

# Ground state
def groundstate(g):
    H = H_rabi(g)
    evals, evecs = H.eigenstates()
    return evecs[0]

# Centered finite difference
def dpsi_dg(g):
    psi_p = groundstate(g + dg)
    psi_m = groundstate(g - dg)
    psi_0 = groundstate(g)
    
    # Phase correction
    phase_p = (psi_0.dag() * psi_p)
    phase_m = (psi_0.dag() * psi_m)
    
    psi_p = psi_p * np.exp(-1j * np.angle(phase_p))
    psi_m = psi_m * np.exp(-1j * np.angle(phase_m))

    return (psi_p - psi_m) / (2 * dg), psi_0

# Quantum Fisher Information
def QFI(g):
    dpsi, psi = dpsi_dg(g)
    overlap = (psi.dag() * dpsi)
    norm_dpsi2 = (dpsi.dag() * dpsi)
    return 4 * (np.real(norm_dpsi2) - np.abs(overlap)**2)

# --- Calculation ---
qfi_vals = np.array([np.real(QFI(g)) for g in g_vals])

# --- Plotting ---
plt.figure(figsize=(7, 5))

# 1. Define X-axis
# Since delta_gamma_vals IS ALREADY (gamma_c - gamma), we use it directly as x
x_axis = delta_gamma_vals

# 2. Reference line with slope -2
# We use the same range as the data for the reference line to match nicely
x_ref = np.geomspace(min_delta_gamma, 1e-1, 10) # Visual range for the line
C_ref = 0.5 
y_ref = C_ref * x_ref**(-2)

# 3. Plot reference
plt.loglog(x_ref, y_ref, color='green', linestyle=':', linewidth=2, label="Slope -2 reference")

# 4. Plot main data
plt.loglog(x_axis, qfi_vals, label=r"$F_Q$ vs $\gamma_c - \gamma \quad (N=" + str(N) + r")$")

# 5. Vertical line for gamma = 0
# At gamma=0, the distance (x-axis) is exactly gamma_c
plt.axvline(x=gamma_c, color='r', linestyle='--', label=r"$\gamma = 0$")

# --- Styles ---
plt.xlabel(r"$\gamma_c - \gamma$", fontsize=13)
plt.ylabel(r"$F_Q(g)$", fontsize=13)
plt.tick_params(labelsize=13)

# Grid (x-axis only)
plt.grid(True, which="major", axis='x', ls="-", alpha=0.4)
plt.grid(True, which="minor", axis='x', ls="--", alpha=0.2)

plt.legend(fontsize=12)
plt.tight_layout()
plt.show()