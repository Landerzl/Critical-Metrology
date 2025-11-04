import numpy as np
import matplotlib.pyplot as plt
from qutip import destroy, sigmax, sigmaz, qeye, tensor

# Parameters
N = 60 # Bosonic truncation
Delta = 1.0  # Qubit energy splitting
omega = 5e-5  # Oscillator frequency
#We fixed w/Delta = 5e-5
domega = Delta * omega

# Critical parameter
gamma_c = 1.0 / np.sqrt(2) # Defines the upper limit for gamma


# Create an exponentially spaced array for the x-axis (gamma_c - gamma)
# We want to plot from a very small (gamma_c - gamma) up to gamma_c
num_points = 100
# We define the minimum distance from the critical point, as log scale cannot plot 0
min_delta_gamma = 1e-3

# We use np.geomspace to create exponentially spaced points, this is important as we will use a log-log plot.
# We generate them in descending order: from gamma_c (corresponds to gamma=0)
# down to min_delta_gamma (corresponds to gamma -> gamma_c)
# This creates points that will be equispaced on a log plot
delta_gamma_vals = np.geomspace(gamma_c, min_delta_gamma, num_points)

# We then calculate the corresponding gamma_vals for the Hamiltonian
# This array will be ascending: from 0 up to (gamma_c - min_delta_gamma)
gamma_vals = gamma_c - delta_gamma_vals

# g = gamma * sqrt(Delta * omega)
g_vals = gamma_vals * np.sqrt(domega)
dg = 1e-6 * np.sqrt(domega) # Infinitesimal step for finite difference. (same as in other fi


# Operators
a = destroy(N)
adag = a.dag()
I_q = qeye(2) # Qubit identity
I_b = qeye(N) # Boson identity
sx = sigmax()
sz = sigmaz()

# General Rabi Hamiltonian as a function of g
def H_rabi(g):
    # H0: Free boson term
    H0 = omega * tensor(adag * a, I_q)
    # H1: Qubit term
    H1 = Delta * tensor(I_b, sz)
    # Hint: Interaction term (coupling)
    Hint = g * tensor(a + adag, sx)
    return H0 + H1 + Hint

# Ground state
def groundstate(g):
    H = H_rabi(g)
    # Calculate all eigenstates and eigenvalues
    evals, evecs = H.eigenstates()
    return evecs[0]  # Return the lowest energy eigenstate (ground state). Alternative to H.groundstate

# Centered finite difference of the state |ψ⟩ with respect to g
def dpsi_dg(g):
    psi_p = groundstate(g + dg) # State at g + dg
    psi_m = groundstate(g - dg) # State at g - dg
    psi_0 = groundstate(g)      # State at g

    # Phase Alignment: Essential for a correct numerical derivative.
    # We force the overlap <psi_0 | psi_p> and <psi_0 | psi_m> to be real and positive.
    
    # Calculate overlap (complex number)
    phase_p = (psi_0.dag() * psi_p)
    phase_m = (psi_0.dag() * psi_m)
    
    # Correct phases
    psi_p = psi_p * np.exp(-1j * np.angle(phase_p))
    psi_m = psi_m * np.exp(-1j * np.angle(phase_m))

    # Centered finite difference (derivative approximation)
    return (psi_p - psi_m) / (2 * dg), psi_0

# Quantum Fisher Information (QFI) for each g
def QFI(g):
    dpsi, psi = dpsi_dg(g)
    overlap = (psi.dag() * dpsi)
    norm_dpsi2 = (dpsi.dag() * dpsi)
    
    # F_Q = 4 * (||dpsi||^2 - |<psi | dpsi>|^2)
    # .real is used because QFI must be a real number
    return 4 * (norm_dpsi2 - np.abs(overlap)**2).real

# Calculation
# NOTE: This calculation can be computationally intensive.
print("Initiating QFI computation...")
qfi_vals = np.array([QFI(g) for g in g_vals])
print("Computation completed.")

# --- PLOTTING SECTION ---
plt.figure(figsize=(7, 5))

#We want to add a straight line of slope -2 for comparison.
reference_x = delta_gamma_vals[0]
reference_y = qfi_vals[0]

# Create an array of x-values for the reference line, covering the critical region
# Let's make it span from the smallest delta_gamma to a bit further out (the exact limit is arbitrary)
ref_line_x = np.geomspace(min_delta_gamma, 0.1, 50) 

# Calculate y-values for the reference line with slope -2
# Equation: Y_ref = Y_anchor * (X_ref / X_anchor)^(-2)
ref_line_y = reference_y * (ref_line_x / reference_x)**(-2)

plt.plot(ref_line_x, ref_line_y, color='green', linestyle=':', linewidth=2, 
         label=r"Slope -2 reference")



# Plot QFI (y) vs (gamma_c - gamma) (x)
# Matplotlib sorts the x-axis, so the plot will correctly go
# from min_delta_gamma (left) to gamma_c (right)
plt.plot(delta_gamma_vals, qfi_vals, label=r"$F_Q \text{ vs } \gamma_c - \gamma \quad (N=" + str(N) + r")$")

# Set both scales to logarithmic
plt.yscale("log")
plt.xscale("log")

plt.xlabel(r"$\gamma_c - \gamma$", fontsize=14)
plt.ylabel(r"$F_Q(g)$", fontsize=14)
plt.xticks(fontsize=14) 
plt.yticks(fontsize=14) 

plt.legend(fontsize=14)
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()
plt.show() 