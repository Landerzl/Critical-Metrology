# Corrections to the superradiant-phase numerics (October 2026)

While preparing an arXiv version of the thesis, the superradiant-phase code was checked against an
**exact** reference: the QRM ground state computed in a very large Fock basis, inside its parity sector,
with no displacement, no squeezing and no Schrieffer-Wolff transformation at all.
This revealed three problems in the scripts used for the thesis figures. They are fixed in this repository.

## The three problems

| # | Problem | Effect |
|---|---------|--------|
| 1 | **Sign of the `tau_z` coupling** in the squeezed Hamiltonian `H_+`. The scripts used `sin2θ = +2gα_s/Δ̃` together with `−sin2θ τ_z`; the correct convention is `tan 2θ = −2gα_s/Δ̃` (as in the thesis text), i.e. `+|sin 2θ| τ_z`. | `H_+` was **not** unitarily equivalent to the QRM: its ground energy was wrong by up to ~5 (in units of Δ). |
| 2 | **Displacement without `sqrt(Δ/ω)`**. The scripts used `α = sqrt(γ² − 1/(4γ²))` instead of the physical `α_s = sqrt(γ² − 1/(4γ²))·sqrt(Δ/ω)` of the thesis. | Any displacement is a unitary, so the Hamiltonians stayed exact, but at small ω the state was far from the origin of the truncated basis, so convergence was poor (e.g. ground-energy error 0.12 at ω/Δ = 0.01, N = 60). Consecutive-cutoff fidelities did not reveal it. |
| 3 | **QFI of the transformed state, not of the physical state.** The displacement, spin rotation and squeezing all depend on `g`. Differentiating only the ground state of the transformed Hamiltonian ignores this dependence. | Superradiant QFI underestimated by orders of magnitude at small ω (e.g. ≈ 10² instead of ≈ 5·10⁴ at ω/Δ = 0.01, γ/γ_c = 1.8). |

The physical state is `|Ψ(g)⟩ = U(g)|φ(g)⟩`, so `∂_g|Ψ⟩ = U(∂_g|φ⟩ + A|φ⟩)` with `A = U†∂_gU`:

```
squeezed frame :  A = α_s' e^{r} (b† − b) + i θ' σ_y + (r'/2)(b² − b†²)
displaced frame:  A = α_s' (a† − a)
F_Q = 4 ( <χ|χ> − |<φ|χ>|² ),   |χ> = ∂_g|φ> + A|φ>
```

Derivation and implementation: `Physical-QFI/physical_qfi.py`.

## Validation (`Physical-QFI/validate_against_exact.py`)

* Ground energy of the corrected `H_+` and of the displaced `H'_+` vs the exact QRM: agreement to ~1e-15 (N = 150).
* Physical QFI vs QFI of the exact ground state (ω/Δ ≥ 0.01, away from the critical point): agreement to ~1e-9 in
  log10(F_Q) at typical points; ≤ 6e-3 on all points accepted by the cutoff criterion below.
* The old "fixed-frame" QFI is wrong by orders of magnitude (e.g. 0.8 instead of 3.4 in log10 F_Q at ω/Δ = 0.05, γ/γ_c = 1.6).

## Remaining caveat: a thin strip next to the critical point

Close to γ_c, at moderate ω, the second symmetry-broken branch enters the truncated Fock space and the result depends on
the cutoff. The maps are therefore computed with three cutoffs (60, 80, 100); the median is used, points where the cutoffs
differ by ≥ 0.05 in log10(F_Q) are flagged (≈ 7% of the points of the heatmap) and interpolated along γ.
The exact reference is the parity-symmetric ground state, which is the symmetry-broken branch only where the branches do not overlap.

## What changed in the repository

| File | Change |
|------|--------|
| `Physical-QFI/` (new) | validated module `physical_qfi.py`, `validate_against_exact.py` |
| `Optimal Base superradiant/FidelityQRMbbdagger.py` | sign (1) and `α_s` (2) |
| `Optimal Base superradiant/entanglement-entropy.py` | sign (1) and `α_s` (2); defaults set to ω/Δ = 5·10⁻⁵, N = 50 (the values of the thesis figure; were 5·10⁻², 500) |
| `Optimal Base superradiant/heatmap-b,bdagger.py` | rewritten on top of `physical_qfi.py` (1, 2, 3); three-cutoff median |
| `Optimal Base superradiant/waterfall.py` | same |
| `QFI and related/heatmap_combined.py` | superradiant half on `physical_qfi.py`; colour cap `vmax=8` removed |
| `QFI and related/heatmap_superradiant.py`, `QFI_superradiant.py`, `QFI_bothphases.py` | displaced frame: `α_s` (2) and frame term (3) |
| `QFI and related/critical_exponent-superradiant.py` | was a copy of the normal-phase script; now computes the physical superradiant QFI |
| `Fidelities and convergence/fidelity_superradiant.py`, `displaced-b-basis.py` | `α_s` (2) (the sign convention of `displaced-b-basis.py` was already the correct one) |
| `Post-submission-checks/Fidelity.py` | sign (1) (`α_s` was already correct) |
| `Post-submission-checks/Heatmap.py` | on `physical_qfi.py`, single cutoff (the cutoff dependence is not flagged) |
| `Extra Code/Superradiant-signoscillator.py` | `α_s` (2) inside `Δ̃` |

Scripts not listed (normal phase, full-Fock-basis fidelity, SW analytics, …) were not affected.

## Consequences for the thesis results

* **Fidelity of the displaced and displaced-squeezed bases.** With the corrected scripts the fidelity between consecutive
  cutoffs is above 0.99 from the smallest cutoff for all ω/Δ. The displacement alone already converges; the squeezing does
  *not* noticeably improve the convergence (it puts the Hamiltonian in the form diagonalised by the Bogoliubov transformation).
* **Entanglement entropy.** The corrected entropy peaks at γ_c and decays on both sides (≈ 6.8·10⁻⁴ at γ/γ_c = 0.99,
  ≈ 4.9·10⁻⁴ at 1.01, → 0 for large γ). The previous "change of slope and plateau" in the superradiant phase came from the wrong Hamiltonian.
* **Superradiant QFI heatmap.** The corrected QFI is smooth, largest near γ_c and increasing as ω → 0 (≈ 10¹¹ at ω/Δ = 5·10⁻⁵).
  The "anomalous regime of enhanced susceptibility" of the previous heatmap is not present in the exact QFI and is not reproduced.
* **Critical exponent on the superradiant side.** The exponent −2 holds in the normal phase only. On the superradiant side the
  physical QFI follows a milder power law, ≈ −0.6 over 3·10⁻³ < γ/γ_c − 1 < 0.1 at ω/Δ = 5·10⁻⁵ (`critical_exponent-superradiant.py`).
* Normal-phase results (QFI, exponent −2, normal-phase heatmap, fidelities of the full Fock basis) are unaffected.
