# Ground energy of Rabi H and effective H' vs Fock cutoff (QuTiP)
# ---------------------------------------------------------------
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

# --------- parameters (edit) ---------
omega   = 1.0     # oscillator frequency
Delta   = 0.7     # qubit frquency 
g       = 0.001    # coupling
N_list  = list(range(4, 81, 2))  # Fock cutoffs

# --------- useful functions ---------
def ops(N):
    """Tensor operators for qubit ⊗ oscillator(N)."""
    a  = qt.tensor(qt.qeye(2), qt.destroy(N))
    ad = a.dag()
    I2 = qt.qeye(2)
    IN = qt.qeye(N)
    sx = qt.tensor(qt.sigmax(), IN)
    sz = qt.tensor(qt.sigmaz(), IN)
    num = ad * a
    I  = qt.tensor(I2, IN)
    return a, ad, num, sx, sz, I

def ground_energy_rabi(N, omega, Delta, g):
    a, ad, num, sx, sz, I = ops(N)
    H = omega*num + Delta*sz + g*sx*(a+ad)
    E0, _ = H.groundstate()
    return float(E0)

def ground_energy_eff(N, omega, Delta, g):
    a, ad, num, sx, sz, I = ops(N)
    denom = (4*Delta*Delta - omega*omega)
    if abs(denom) < 1e-10:
        raise ValueError("Denominator 4Δ^2 - ω^2 ≈ 0 (breakdown of effective model).")
    H = (omega*num
         + Delta*sz
         - (2*g*g*Delta/denom) * sz * (a+ad)*(a+ad)
         + (g*g*omega/denom) * I)  # constant energy shift
    E0, _ = H.groundstate()
    return float(E0)

# --------- compute ---------
E0_rabi = []
E0_eff  = []
for N in N_list:
    E0_rabi.append(ground_energy_rabi(N, omega, Delta, g))
    E0_eff.append(ground_energy_eff (N, omega, Delta, g))



# ---------  plotting ---------
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

# (1) Energies vs cutoff
fig1, ax1 = plt.subplots()
ax1.plot(N_list, E0_rabi, marker='x', linestyle='-', label=r'Rabi $H$')
ax1.plot(N_list, E0_eff,  marker='x', linestyle='--', label=r'Effective $H^\prime$')
ax1.set_xlabel(r'Fock cutoff $N$')
ax1.set_ylabel(r'Ground energy $E_0(N)$')
ax1.legend(frameon=False)
fig1.tight_layout()

plt.show()
