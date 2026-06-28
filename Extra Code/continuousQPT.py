import numpy as np
import matplotlib.pyplot as plt
from qutip import destroy, tensor, sigmaz, sigmax, qeye

# --- 1. System Parameters ---
Delta = 1.0
omega = 0.05  # Small frequency to approach the classical oscillator limit
gamma_c = 1 / np.sqrt(2) # Analytical critical point
gamma_vec = np.linspace(0.1, 1.2, 400) # Normalized coupling range
N = 80 # Fock cutoff

# --- 2. Operators ---
a = tensor(destroy(N), qeye(2))
sz = tensor(qeye(N), sigmaz())
sx = tensor(qeye(N), sigmax())

E0 = np.zeros(len(gamma_vec))

# --- 3. Diagonalization and E_0 computation ---
print("Calculating ground state energies...")
for i, gamma in enumerate(gamma_vec):
    g = gamma * np.sqrt(omega * Delta)
    # Rabi Hamiltonian
    H = omega * a.dag() * a + Delta * sz + g * sx * (a + a.dag())
    
    # Extract only the lowest eigenvalue (Ground State)
    E0[i] = H.eigenenergies(eigvals=1)[0]

# --- 4. Numerical Derivatives ---
# We use np.gradient to compute the first and second derivatives with respect to gamma
dE = np.gradient(E0, gamma_vec)
d2E = np.gradient(dE, gamma_vec)

# --- 5. Visualization ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

# Energy Plot (Continuous)
ax1.plot(gamma_vec, E0, 'b-', lw=2)
ax1.axvline(gamma_c, color='r', linestyle='--', label=r'$\gamma_c = 1/\sqrt{2}$')
ax1.set_ylabel(r'Energy $E_0$')
ax1.set_title('Ground State Energy and its Second Derivative')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Second Derivative Plot (Discontinuous / Divergent)
ax2.plot(gamma_vec, d2E, 'g-', lw=2)
ax2.axvline(gamma_c, color='r', linestyle='--', label=r'$\gamma_c$')
ax2.set_xlabel(r'Normalized coupling $\gamma$')
ax2.set_ylabel(r'$\partial^2 E_0 / \partial \gamma^2$')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()