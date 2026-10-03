"""
Validate the transformed-frame Hamiltonians and the physical QFI against the exact QRM.

 (1) Ground energy: H_+ (squeezed) and H'_+ (displaced) must reproduce the exact QRM ground energy.
 (2) QFI: physical QFI (frame term included) vs the QFI of the exact QRM ground state computed in a very
     large Fock basis inside its parity sector (no displacement, no squeezing, no SW at all).
     Also shown: the 'fixed-frame' QFI of the old scripts (frame term omitted).
Takes about a minute.
"""
import numpy as np
from physical_qfi import *

print("(1) ground-state energy vs exact QRM (|dE|)")
print("    omega   gamma/gamma_c |  squeezed (N=150)   displaced (N=150)")
for om in (0.5, 0.1, 0.02):
    ex = ExactModel(ExactModel.required_cutoff(om))
    sq, di = SqueezedModel(150), DisplacedModel(150)
    for gr in (1.2, 1.5, 2.0):
        g = gr * gamma_c * np.sqrt(om * Delta)
        alpha, theta, r, Dt = frame(g, om)
        E_ex = ex.energy(g, om)
        E_sq = sq.energy(g, om) + om * alpha**2 + om * np.sinh(r) ** 2      # constants dropped in H_+
        E_di = di.energy(g, om)
        print("    %.3f   %.2f          |  %.2e           %.2e" % (om, gr, abs(E_sq - E_ex), abs(E_di - E_ex)))

print("\n(2) log10 F_Q: physical (frame term) vs exact; and fixed-frame (old scripts) vs exact")
print("    omega   gamma/gamma_c | exact    squeezed  displaced | fixed-frame squeezed")
for om in (0.2, 0.05, 0.02):
    ex = ExactModel(ExactModel.required_cutoff(om))
    sq, di = SqueezedModel(100), DisplacedModel(100)
    for gr in (1.3, 1.6, 2.0):
        g = gr * gamma_c * np.sqrt(om * Delta)
        f_ex = np.log10(ex.qfi(g, om))
        f_sq = np.log10(sq.qfi(g, om))
        f_di = np.log10(di.qfi(g, om))
        f_old = np.log10(sq.qfi(g, om, include_frame=False))
        print("    %.3f   %.2f          | %6.3f   %6.3f    %6.3f   | %6.3f" % (om, gr, f_ex, f_sq, f_di, f_old))
