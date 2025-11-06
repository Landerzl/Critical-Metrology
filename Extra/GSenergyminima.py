import numpy as np
import matplotlib.pyplot as plt

def f(x, gamma):
    return x**2 - np.sqrt(1 + 4 * gamma**2 * x**2)


x = np.linspace(-3, 3, 500)


gamma1 = 0.6 / np.sqrt(2)   # gamma^2 < 1/2
gamma2 = 2 / np.sqrt(2)   # gamma^2 > 1/2

f1 = f(x, gamma1)
f2 = f(x, gamma2)


plt.figure(figsize=(5,5))
plt.plot(x, f1, label=r'$\gamma^2 < 1/2$', linewidth=2)
plt.plot(x, f2, label=r'$\gamma^2 > 1/2$', linewidth=2)

plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.axvline(0, color='gray', linestyle='--', linewidth=1)
plt.xlabel(r'$x$', fontsize=14)
plt.ylabel(r'$f(x)$', fontsize=14)
plt.ylim(-3,1)

plt.xticks([])   
plt.yticks([])   

plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
