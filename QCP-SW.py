import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
Delta = 10.0
omega = 1.0
g_c = np.sqrt(Delta * omega / 2.0)

# Choose g values (prefactors don't matter)
g_less    = 0.6 * g_c
g_equal   = g_c
g_greater = 1.4 * g_c

# Range for |alpha|^2
alpha2 = np.linspace(0.0, 3.0, 300)

def E_I(alpha2, g, Delta, omega):
    return Delta + (omega - 2.0 * g**2 / Delta) * alpha2

# --- Plot ---
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(alpha2, E_I(alpha2, g_less, Delta, omega),   label=rf"$g<g_c$")
ax.plot(alpha2, E_I(alpha2, g_equal, Delta, omega),  label=rf"$g=g_c$")
ax.plot(alpha2, E_I(alpha2, g_greater, Delta, omega),label=rf"$g>g_c$")

ax.set_xlabel(r"$|\alpha|^2$")
ax.set_ylabel(r"$E_I(|\alpha|^2)$")
ax.legend(loc = "center")

ax.grid(False)


ax.axhline(0, color="black", linewidth=1)
ax.axvline(0, color="black", linewidth=1)

plt.tight_layout()
plt.show()
