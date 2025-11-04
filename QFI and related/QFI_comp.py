import numpy as np
import matplotlib.pyplot as plt

# Constants
Delta = 1
omega = 0.01

# Gamma range
gamma2 = np.linspace(0, 0.5, 500)
gamma = np.sqrt(gamma2)

# Expressions
T_SW1 = np.sqrt(1 - 2 * gamma2) / Delta**2
T_SW2 = (4 * omega * gamma**2 / Delta**3) * (1 - 2 * gamma2) \
        + np.sqrt(1 - 2 * gamma2) / Delta**2 \
        + (20 * omega**2 * gamma**4 / (3 * Delta**4)) * (1 - 2 * gamma2)**(3/2)

# Plot
plt.figure(figsize=(7,5))
plt.plot(gamma2, T_SW1, label=r"$\mathcal{I}_{\mathrm{regular}}^{\mathrm{SW1}}$", color='orange', linewidth=2)
plt.plot(gamma2, T_SW2, label=r"$\mathcal{I}_{\mathrm{regular}}^{\mathrm{SW2}}$", color='deepskyblue', linewidth=2)
plt.xlabel(r"$\gamma^2$", fontsize=13)
plt.ylabel(r"$\mathcal{I}$", fontsize=13)
plt.legend()
plt.text(0.02, 0.95, rf"$\omega = {omega}$", transform=plt.gca().transAxes, fontsize=12, 
         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
plt.xlim(0, 0.5)
plt.ylim(0,1.5)
plt.tight_layout()
plt.show()
