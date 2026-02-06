import numpy as np
import matplotlib.pyplot as plt

def f(x, gamma):
    return x**2 - np.sqrt(1 + 4 * gamma**2 * x**2)

x = np.linspace(-3, 3, 500)

# Gamma values
gamma_sub = 0.6 / np.sqrt(2)   # gamma^2 < 1/2
gamma_crit = 1.0 / np.sqrt(2)  # gamma^2 = 1/2 
gamma_super = 2.0 / np.sqrt(2) # gamma^2 > 1/2

# Calculate functions
f_sub = f(x, gamma_sub)
f_crit = f(x, gamma_crit)
f_super = f(x, gamma_super)

plt.figure(figsize=(3, 5)) 


# 1. Sub-critical 
plt.plot(x, f_sub, linestyle='-', linewidth=2, label=r'$\gamma^2 < 1/2$')

# 2. Critical 
plt.plot(x, f_crit, linestyle='-.', linewidth=2, label=r'$\gamma^2 = 1/2$')

# 3. Super-critical 
plt.plot(x, f_super, linestyle='--', linewidth=2, label=r'$\gamma^2 > 1/2$')

# Reference lines 
plt.axhline(0, color='gray', linestyle=':', linewidth=1)
plt.axvline(0, color='gray', linestyle=':', linewidth=1)

plt.xlabel(r'$x$', fontsize=18)
plt.ylabel(r'$f(x)$', fontsize=18)
plt.ylim(-3, 1)


plt.xticks([])   
plt.yticks([])   

plt.legend(fontsize=10)
plt.grid(False) 
plt.tight_layout()
plt.show()