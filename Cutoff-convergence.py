# Rabi-model ground energy & convergence vs Fock cutoff (multiple omegas)
# ----------------------------------------------------------------------
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

# ----- parameters -----
omega_list = [0.5,0.05, 0.005, 0.0005]   # values of ω to compare
Delta = 0.7                    # qubit frequency
g     = 0.5                    # coupling strength
N_list = list(range(4, 81, 1)) # Fock cutoffs 

def rabi_ground_energy(N, omega, Delta, g):
    a   = qt.tensor(qt.qeye(2), qt.destroy(N))
    num = a.dag() * a
    sx  = qt.tensor(qt.sigmax(), qt.qeye(N))
    sz  = qt.tensor(qt.sigmaz(), qt.qeye(N))
    H = omega * num + Delta * sz + g * sx * (a + a.dag())
    E0, _ = H.groundstate()
    return float(E0)

# Store results for all ω
E0_dict = {}
dE_dict = {}

for om in omega_list:
    E0_vals = [rabi_ground_energy(N, om, Delta, g) for N in N_list]
    # Convergence to previous N in the list
    dE_vals = [np.nan] + [abs(E0_vals[i]-E0_vals[i-1]) / max(1e-12, abs(E0_vals[i])) for i in range(1, len(E0_vals))] #Risk of dividing by zero
    E0_dict[om] = E0_vals
    dE_dict[om] = dE_vals


# ----- Plot -----
plt.rcParams.update({
    "figure.figsize": (6.0, 3.5),
    "font.size": 12,
    "font.family": "serif",
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.2,
    "lines.markersize": 5,
})


linestyles = ['-', '--', '-.', ':']

# ----- Figure 1: ground energy -----
fig1, ax1 = plt.subplots()
for i, om in enumerate(omega_list):
    ax1.plot(N_list, E0_dict[om],
             marker='x', linestyle=linestyles[i % len(linestyles)],
             label=fr'$\omega = {om}$')
ax1.set_xlabel(r'Fock cutoff $N$')
ax1.set_ylabel(r'Ground energy $E_0(N)$')
ax1.legend(frameon=False, title=r'$\omega$')
fig1.tight_layout()

# ----- Figure 2: convergence -----
fig2, ax2 = plt.subplots()
for i, om in enumerate(omega_list):
    ax2.plot(N_list[1:], dE_dict[om][1:],
             marker='x', linestyle=linestyles[i % len(linestyles)],
             label=fr'$\omega = {om}$')
ax2.set_xlabel(r'Fock cutoff $N$')
ax2.set_ylabel(r'$|E_0(N) - E_0(N_{\mathrm{prev}})| / |/E_0(N)|$')
ax2.set_yscale('log')  # log for visualization
ax2.legend(frameon=False, title=r'$\omega$')
fig2.tight_layout()

plt.show()
plt.show()
