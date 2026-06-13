import numpy as np
import matplotlib.pyplot as plt

Delta = 1
omegas = [1, 0.1, 0.01, 0.001]

gamma2 = np.linspace(0, 0.5, 500)
gamma = np.sqrt(gamma2)

fig, axs = plt.subplots(2, 2, figsize=(24, 16))
axs = axs.flatten()

for i, omega in enumerate(omegas):
    ax = axs[i]
    
    T_SW1 = np.sqrt(1 - 2 * gamma2) / Delta**2
    T_SW2 = (4 * omega * gamma**2 / Delta**3) * (1 - 2 * gamma2) \
            + np.sqrt(1 - 2 * gamma2) / Delta**2 \
            + (20 * omega**2 * gamma**4 / (3 * Delta**4)) * (1 - 2 * gamma2)**(3/2)

    ax.plot(gamma2, T_SW1, label=r"$\mathcal{I}_{\mathrm{regular}}^{\mathrm{SW1}}$", color='orange', linewidth=2)
    ax.plot(gamma2, T_SW2, label=r"$\mathcal{I}_{\mathrm{regular}}^{\mathrm{SW2}}$", color='deepskyblue', linewidth=2)
    

    ax.set_xlabel(r"$\gamma^2$", fontsize=30, labelpad=15)
    ax.set_ylabel(r"$\mathcal{I}$", fontsize=30, labelpad=15)
    

    ax.tick_params(axis='both', which='major', labelsize=24, pad=12)
    
    ax.legend(fontsize=22)
    ax.text(0.02, 0.95, rf"$\omega = {omega}$", transform=ax.transAxes, fontsize=28, 
            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1.5)


plt.tight_layout(pad=5.0, w_pad=5.0, h_pad=5.0)


plt.savefig("mi_grafica.png", bbox_inches='tight', dpi=300)

plt.show()