import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------
# Parameters
# -----------------------------------------
Delta = 1.0
omega = 5e-5
gamma_c = 1 / np.sqrt(2)

# Gamma range in the superradiant phase
gamma_vals = np.linspace(gamma_c, 2*gamma_c, 300)

# -----------------------------------------
# Functions
# -----------------------------------------
def alpha_s(gamma):
    return np.sqrt(gamma**2 - 1/(4 * gamma**2))

def g(gamma):
    return gamma * np.sqrt(omega * Delta)

def tilde_Delta(gamma):
    return np.sqrt(Delta**2 + (2 * g(gamma) * alpha_s(gamma))**2)

A_vals = omega - 2 * (g(gamma_vals)**2 * Delta**2) / (tilde_Delta(gamma_vals)**3)

# -----------------------------------------
# Plot
# -----------------------------------------
plt.figure(figsize=(7,5))

# Main curve
plt.plot(gamma_vals, A_vals, label=r"$\omega - 2g^2\Delta^2/\tilde\Delta^3$")

# y = 0 line with legend
plt.axhline(0, linestyle="--", color="black", label=r"$A(\gamma)=0$")

# Only x ticks: γ_c and 2γ_c
plt.xticks(
    [gamma_c, 2*gamma_c],
    [r"$\gamma_c$", r"$2\gamma_c$"]
)

plt.yticks([])
plt.gca().spines['left'].set_visible(True)

plt.xlabel(r"$\gamma$")
plt.ylabel(r"$A(\gamma)$")
plt.legend()
plt.grid(False)
plt.show()
