import numpy as np
import matplotlib.pyplot as plt
from qutip import destroy, sigmax, sigmaz, qeye, tensor

# --- PARAMETERS ---
N = 60  # Bosonic truncation
Delta = 1.0   # Qubit energy splitting
omega = 5e-5   # Oscillator frequency
domega = Delta * omega

# Critical parameter
gamma_c = 1.0 / np.sqrt(2) # Defines the transition point



num_points = 100
min_delta_gamma = 1e-4
delta_gamma_vals = np.geomspace(min_delta_gamma, gamma_c, num_points) #Equispaced point in the plot
gamma_vals = gamma_c + delta_gamma_vals



# g = gamma * sqrt(Delta * omega)
g_vals = gamma_vals * np.sqrt(domega)
dg = 1e-6 * np.sqrt(domega) # Infinitesimal step for finite difference

# --- OPERATORS ---
a = tensor(qeye(2), destroy(N))
adag = a.dag()
num = adag * a
sx = tensor(sigmax(), qeye(N))
sz = tensor(sigmaz(), qeye(N))
Id = tensor(qeye(2), qeye(N))

# --- HAMILTONIAN FUNCTIONS ---

def displaced_rabi_ground_state(N, omega, Delta, gamma):
    """
    Ground state and energy of the displaced Rabi Hamiltonian.
    H'_+ = ω a†a + Δ σ_z + ω α (a + a†) + g σ_x (a + a†) + 2 g α σ_x + ω α²
    """
    g = gamma * np.sqrt(omega * Delta)
    
    # Safety check for gamma=0, although now gamma is always > gamma_c
    if gamma <= 1e-10:
        radicand = -1.0
    else:
        radicand = gamma**2 - 1.0 / (4.0 * gamma**2)
        
    alpha = np.sqrt(max(radicand, 0.0))  # alpha > 0 in this range (gamma > gamma_c)

    # Hamiltonian H'_+
    H = (omega * num
         + Delta * sz
         + omega * alpha * (a + adag)
         + g * sx * (a + adag)
         + 2.0 * g * alpha * sx
         + omega * alpha**2 * Id)

    E0, psi0 = H.groundstate()
    return float(E0), psi0

def groundstate(g):
    gamma = g / np.sqrt(omega * Delta)
    _, psi0 = displaced_rabi_ground_state(N, omega, Delta, gamma)
    return psi0

# --- QFI ---
def dpsi_dg(g):
    psi_p = groundstate(g + dg)
    psi_m = groundstate(g - dg)
    psi_0 = groundstate(g)

    phase_p = (psi_0.dag() * psi_p)
    phase_m = (psi_0.dag() * psi_m)
    
    psi_p = psi_p * np.exp(-1j * np.angle(phase_p))
    psi_m = psi_m * np.exp(-1j * np.angle(phase_m))

    return (psi_p - psi_m) / (2 * dg), psi_0

def QFI(g):
    dpsi, psi = dpsi_dg(g)
    overlap = (psi.dag() * dpsi)
    norm_dpsi2 = (dpsi.dag() * dpsi)
    
    return 4 * (norm_dpsi2 - np.abs(overlap)**2).real

# --- CALCULATION ---
print("Initiating QFI computation in the Superradiant phase...")
qfi_vals = np.array([QFI(g) for g in g_vals])
print("Computation completed.")

# --- PLOTTING SECTION ---
plt.figure(figsize=(8, 5))

# Reference line logic: Now reference_x is the smallest value (closest to critical point)
reference_x = delta_gamma_vals[0]
reference_y = qfi_vals[0]

# Use an array covering the critical region for the reference line
ref_line_x = np.geomspace(min_delta_gamma, gamma_c, 50) 

# Calculate y-values for the reference line with slope -2
ref_line_y = reference_y * (ref_line_x / reference_x)**(-2)

# Plot reference line
plt.plot(ref_line_x, ref_line_y, color='green', linestyle=':', linewidth=2, 
         label=r"Slope -2 reference")

# Plot QFI (y) vs (gamma - gamma_c) (x)
plt.plot(delta_gamma_vals, qfi_vals, label=r"$F_Q \text{ vs } \gamma - \gamma_c \quad (N=" + str(N) + r")$")


# Straight vertical line for gamma = 2*gamma_c 
plt.axvline(x=gamma_c, color='purple', linestyle='--', label=r'$\gamma = 2\gamma_c$')


# Set both scales to logarithmic
plt.yscale("log")
plt.xscale("log")


plt.xlim(left=min_delta_gamma / 2, right=gamma_c * 1.1)


plt.xlabel(r"$\gamma - \gamma_c$", fontsize=14)
plt.ylabel(r"$F_Q(g)$", fontsize=14)
plt.xticks(fontsize=12) 
plt.yticks(fontsize=12) 

plt.legend(fontsize=12)
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()
plt.show()