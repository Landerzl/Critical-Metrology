"""
Physical QFI of the Quantum Rabi Model in the superradiant phase.

    H_QRM = omega a^dag a + Delta sigma_z + g sigma_x (a + a^dag),      gamma = g / sqrt(omega Delta)

The superradiant ground state is computed in a transformed frame in which the Fock truncation
converges quickly.  The physical ground state (symmetry-broken branch) is

    |Psi(g)> = U(g) |phi(g)>,        U = D(alpha_s)  e^{i theta sigma_y}  S_z(z = -r)        (squeezed frame)
    |Psi(g)> = D(alpha_s) |phi(g)>                                                              (displaced frame)

with
    alpha_s = sqrt(gamma^2 - 1/(4 gamma^2)) * sqrt(Delta/omega)        displacement
    tan(2 theta) = -2 g alpha_s / Delta                                  spin rotation
    r = 1/4 ln(1 + 4 g^2 Delta^2 / (omega Delta_t^3)),  Delta_t = sqrt(Delta^2 + (2 g alpha_s)^2)

IMPORTANT: U depends on g, so d|Psi>/dg = U ( d|phi>/dg + A |phi> ) with A = U^dag dU/dg:

    A_squeezed  = alpha_s' e^{r} (b^dag - b) + i theta' sigma_y + (r'/2) (b^2 - b^dag^2)
    A_displaced = alpha_s' (a^dag - a)

and  F_Q = 4 ( <chi|chi> - |<phi|chi>|^2 ),  |chi> = d|phi>/dg + A |phi>.
Differentiating |phi> alone (the old scripts) is NOT the QFI of the physical state.

Hamiltonian in the squeezed frame (b, b^dag):
    H_+ = omega cosh(2r) b^dag b - (omega/2) sinh(2r) (b^2 + b^dag^2)
          + e^{-r} (b + b^dag) [ omega alpha_s + g ( cos(2 theta) tau_x - sin(2 theta) tau_z ) ] + Delta_t tau_z
(with sin(2 theta) = -2 g alpha_s / Delta_t < 0).  It is unitarily equivalent to H_QRM up to the constant
omega alpha_s^2 + omega sinh^2(r), which has been dropped.

Everything here is dense linear algebra (numpy only).  Validated against exact diagonalization of the
full QRM in a very large Fock basis (see validate_against_exact.py).
"""
import numpy as np

Delta = 1.0
gamma_c = 1.0 / np.sqrt(2.0)

SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def _destroy(N):
    return np.diag(np.sqrt(np.arange(1, N)), 1).astype(complex)


def frame(g, om, Delta=Delta):
    """(alpha_s, theta, r, Delta_tilde) as functions of g."""
    gam = g / np.sqrt(om * Delta)
    x = np.sqrt(max(gam**2 - 1.0 / (4.0 * gam**2), 0.0))
    alpha = x * np.sqrt(Delta / om)
    Dt = np.sqrt(Delta**2 + (2.0 * g * alpha) ** 2)
    theta = -0.5 * np.arctan2(2.0 * g * alpha, Delta)          # tan(2 theta) = -2 g alpha / Delta
    r = 0.25 * np.log1p(4.0 * g**2 * Delta**2 / (om * Dt**3))
    return alpha, theta, r, Dt


class _Model:
    """Base class: ground state + physical QFI by centred finite differences."""

    def ground_state(self, g, om):
        return np.linalg.eigh(self.H(g, om))[1][:, 0]

    def energy(self, g, om):
        return np.linalg.eigvalsh(self.H(g, om))[0]

    def qfi(self, g, om, dg=None, include_frame=True):
        if dg is None:
            dg = 1e-6 * np.sqrt(Delta * om)
        z = self.ground_state(g, om)
        p, m = self.ground_state(g + dg, om), self.ground_state(g - dg, om)
        p = p * np.exp(-1j * np.angle(z.conj() @ p))            # global-phase alignment
        m = m * np.exp(-1j * np.angle(z.conj() @ m))
        chi = (p - m) / (2 * dg)
        if include_frame:
            A = self.A(g, om, dg)
            if not np.isscalar(A):                               # the exact model has no frame
                chi = chi + A @ z
        return 4.0 * (np.real(chi.conj() @ chi) - np.abs(z.conj() @ chi) ** 2)


class SqueezedModel(_Model):
    """Displaced + rotated + squeezed frame (b, b^dag), tensor order [spin, boson]."""

    def __init__(self, N):
        self.N = N
        b = np.kron(np.eye(2), _destroy(N))
        bd = b.conj().T
        self.num = bd @ b
        self.bb = b @ b + bd @ bd
        self.X = b + bd
        self.Dk = bd - b
        self.Sq = b @ b - bd @ bd
        self.tx = np.kron(SX, np.eye(N))
        self.tz = np.kron(SZ, np.eye(N))
        self.sy = np.kron(SY, np.eye(N))

    def H(self, g, om):
        alpha, theta, r, Dt = frame(g, om)
        c2, s2 = np.cos(2 * theta), np.sin(2 * theta)             # = Delta/Dt, -2 g alpha/Dt
        bracket = om * alpha * np.eye(2 * self.N) + g * (c2 * self.tx - s2 * self.tz)
        return (om * np.cosh(2 * r) * self.num - 0.5 * om * np.sinh(2 * r) * self.bb
                + np.exp(-r) * (self.X @ bracket) + Dt * self.tz)

    def A(self, g, om, dg):
        fp, fm, f0 = frame(g + dg, om), frame(g - dg, om), frame(g, om)
        d = [(fp[k] - fm[k]) / (2 * dg) for k in range(3)]       # alpha', theta', r'
        return (d[0] * np.exp(f0[2]) * self.Dk + 1j * d[1] * self.sy + 0.5 * d[2] * self.Sq)


class DisplacedModel(_Model):
    """Displacement only (no rotation, no squeezing), tensor order [spin, boson]."""

    def __init__(self, N):
        self.N = N
        a = np.kron(np.eye(2), _destroy(N))
        ad = a.conj().T
        self.num = ad @ a
        self.X = a + ad
        self.Dk = ad - a
        self.sx = np.kron(SX, np.eye(N))
        self.sz = np.kron(SZ, np.eye(N))

    def H(self, g, om):
        alpha = frame(g, om)[0]
        return (om * self.num + Delta * self.sz + om * alpha * self.X + g * (self.sx @ self.X)
                + 2.0 * g * alpha * self.sx + om * alpha**2 * np.eye(2 * self.N))

    def A(self, g, om, dg):
        d_alpha = (frame(g + dg, om)[0] - frame(g - dg, om)[0]) / (2 * dg)
        return d_alpha * self.Dk


class ExactModel(_Model):
    """Full QRM in the Fock basis, restricted to the parity sector of the ground state.

    P = sigma_z (-1)^n commutes with H_QRM and the g = 0 ground state |0, down> has P = -1.
    Needs N >~ 2*alpha_s^2 + margin (alpha_s^2 ~ 1/omega), so it is only feasible for omega >~ 0.005.
    """

    def __init__(self, N):
        self.N = N
        a = np.kron(np.eye(2), _destroy(N))
        sx, sz = np.kron(SX, np.eye(N)), np.kron(SZ, np.eye(N))
        n = np.arange(N)
        parity = np.concatenate([(-1.0) ** n, -((-1.0) ** n)])
        idx = np.where(parity < 0)[0]
        self.nsec = (a.conj().T @ a)[np.ix_(idx, idx)]
        self.sz = sz[np.ix_(idx, idx)]
        self.v = (sx @ (a + a.conj().T))[np.ix_(idx, idx)]

    def H(self, g, om):
        return om * self.nsec + Delta * self.sz + g * self.v

    def A(self, g, om, dg):
        return 0.0

    @staticmethod
    def required_cutoff(om):
        return int(2.2 * 1.875 / om) + 80


def qfi_map(model_cls, omegas, gammas, cutoffs=(60, 80, 100), tau=0.05, verbose=True):
    """log10 of the physical QFI on an (omega, gamma) grid.

    Near the critical point the two symmetry-broken branches enter the truncated space and
    the result depends on the Fock cutoff.  The map is computed for several cutoffs; the median is
    used, points where the cutoffs disagree by >= tau (in log10 F_Q) are flagged and interpolated
    along gamma.  Returns (log10_map_filled, flagged_mask).
    """
    maps = []
    for N in cutoffs:
        model = model_cls(N)
        if verbose:
            print("  cutoff N =", N, flush=True)
        m = np.zeros((len(omegas), len(gammas)))
        for i, om in enumerate(omegas):
            dg = 1e-6 * np.sqrt(Delta * om)
            m[i] = [model.qfi(gm * np.sqrt(Delta * om), om, dg) for gm in gammas]
        maps.append(np.log10(m + 1e-10))
    stack = np.stack(maps)
    med = np.median(stack, axis=0)
    flag = (stack.max(axis=0) - stack.min(axis=0)) >= tau
    filled = med.copy()
    for i in range(len(omegas)):
        ok = ~flag[i]
        if ok.sum() >= 2 and flag[i].any():
            filled[i, flag[i]] = np.interp(np.asarray(gammas)[flag[i]], np.asarray(gammas)[ok], med[i, ok])
    return filled, flag
