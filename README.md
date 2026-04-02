# Critical Metrology in the Quantum Rabi Model

**Bachelor's Thesis — Physics**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![QuTiP](https://img.shields.io/badge/QuTiP-Quantum%20Toolbox-76B900)](https://qutip.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](https://github.com/Landerzl/Critical-Metrology/pulls)

*Numerical study of the Quantum Fisher Information and critical quantum metrology near the quantum phase transition of the Quantum Rabi Model.*

---

## About

This repository contains all the numerical code (and supplementary derivations) used in my Bachelor's thesis on **Critical Metrology in the Quantum Rabi Model (QRM)**.

The project investigates how quantum parameter estimation precision — quantified by the **Quantum Fisher Information (QFI)** — diverges near the **quantum phase transition** of the QRM. The analysis covers:

- **Normal phase** (γ < γ_c): Standard Fock-basis diagonalisation of the full Rabi Hamiltonian.
- **Superradiant phase** (γ > γ_c): Displaced-squeezed basis representation to accelerate convergence.
- **Critical exponents**: Log-log analysis showing F_Q ~ |γ - γ_c|^(-2).
- **Analytical validation**: Comparison of numerically obtained ground states with Schrieffer–Wolff (SW) perturbative results.

---

## Repository Structure

```
Critical-Metrology/
│
├── QFI and related/                              # Core QFI computations
│   ├── QFI_fullH.py                                # QFI from full Rabi H (normal phase)
│   ├── QFI_superradiant.py                         # QFI in the superradiant phase
│   ├── QFI_bothphases.py                           # Combined QFI plot across the QPT
│   ├── QFI_comp.py                                 # QFI comparison utilities
│   ├── critical_exponent.py                        # Critical exponent extraction (normal)
│   ├── critical_exponent-superradiant.py           # Critical exponent (superradiant)
│   ├── heatmap_normalPhase.py                      # QFI heatmap — normal phase (γ vs ω)
│   ├── heatmap_superradiant.py                     # QFI heatmap — superradiant phase
│   └── heatmap_combined.py                         # Combined heatmap across both phases
│
├── Fidelities and convergence/                   # Numerical convergence & basis validation
│   ├── Cutoff-convergence.py                       # Ground energy convergence vs Fock cutoff
│   ├── fidelity.py                                 # Fidelity F(ψ_N, ψ_{N-1}) — Fock basis
│   ├── fidelity_superradiant.py                    # Fidelity — displaced basis (superradiant)
│   ├── fidelity_numericalVSanalytical.py           # Fidelity: numerical vs SW analytical GS
│   └── displaced-b-basis.py                        # Fidelity in squeezed-displaced basis
│
├── Optimal Base superradiant/                    # Superradiant-phase analysis tools
│   ├── FidelityQRMbbdagger.py                      # Fidelity analysis with b, b† operators
│   ├── entanglement-entropy.py                     # Von Neumann entropy across the QPT
│   ├── heatmap-b,bdagger.py                        # QFI heatmap in the b, b† basis
│   └── waterfall.py                                # 3D waterfall plot of QFI vs γ and ω
│
├── Extra Code/                                   # Supplementary & exploratory scripts
│   ├── Classical_Rabi.ipynb                        # Classical Rabi model (Jupyter notebook)
│   ├── GSenergyminima.py                           # Ground-state energy landscape f(x, γ)
│   ├── GSenergy_vs_firstexcited                    # Energy gap E₁ − E₀ near the QPT
│   ├── QCP-SW.py                                   # Quantum critical point from SW expansion
│   ├── SW_vs_QRM.py                                # SW effective H vs full Rabi comparison
│   └── Superradiant-signoscillator.py              # Sign of the effective oscillator freq.
│
├── Further derivations/                          # Supplementary analytical PDF documents
│   ├── Finite_difference_step.pdf                  # Optimal finite-difference step analysis
│   ├── QFI_SW1__extended_calculation_.pdf          # QFI from 1st-order SW (extended)
│   └── QFI_SW2__extended_calculation_.pdf          # QFI from 2nd-order SW (extended)
│
└── README.md
```

---

## Getting Started

### Prerequisites

| Requirement    | Version | Purpose                                      |
|----------------|---------|----------------------------------------------|
| **Python**     | >= 3.8  | Runtime                                      |
| **NumPy**      | >= 1.20 | Linear algebra & numerical arrays            |
| **QuTiP**      | >= 4.7  | Quantum objects, operators & diagonalisation  |
| **Matplotlib** | >= 3.4  | Plotting & visualisation                     |
| **Seaborn**    | >= 0.11 | Heatmap visualisation (used in heatmap scripts)|

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Landerzl/Critical-Metrology.git
   cd Critical-Metrology
   ```

2. **Create a virtual environment** *(recommended)*

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install numpy qutip matplotlib seaborn
   ```

   > **Note:** QuTiP may require a C compiler. On Windows it is usually installed via pre-built wheels with no extra setup. If you encounter issues, see the [QuTiP installation guide](https://qutip.org/docs/latest/installation.html).

---

## Running the Scripts

Each script is **self-contained** — it defines its own parameters, builds the Hamiltonians, runs the computation, and displays the results. Simply run any script directly:

```bash
# QFI across both phases
python "QFI and related/QFI_bothphases.py"

# Fock-cutoff convergence study
python "Fidelities and convergence/Cutoff-convergence.py"

# Entanglement entropy across the QPT
python "Optimal Base superradiant/entanglement-entropy.py"

# 3D waterfall plot of QFI
python "Optimal Base superradiant/waterfall.py"

# Combined QFI heatmap (normal + superradiant)
python "QFI and related/heatmap_combined.py"
```

The Jupyter notebook can be opened with:

```bash
jupyter notebook "Extra Code/Classical_Rabi.ipynb"
```

### Adjustable Parameters

Most scripts expose tunable physical parameters at the top of the file:

| Parameter | Symbol   | Typical Value     | Description                        |
|-----------|----------|-------------------|------------------------------------|
| `N`       | N        | 50 – 100          | Fock-space truncation (bosonic cutoff) |
| `Delta`   | Δ        | 1.0               | Qubit energy splitting             |
| `omega`   | ω        | 5e-5              | Oscillator frequency               |
| `gamma_c` | γ_c      | 1/√2 ≈ 0.707     | Critical coupling (derived)        |
| `dg`      | δg       | 1e-6 × √(ωΔ)     | Finite-difference step for QFI     |

> **Performance note:** Some computations (especially the heatmaps and 3D waterfall) involve sweeping over many (γ, ω) pairs and can take **several minutes** depending on the grid resolution and Fock cutoff.

---

## Key Results You Can Reproduce

| Script | What It Shows |
|--------|---------------|
| `QFI_fullH.py` | QFI diverges as γ → γ_c from the normal phase |
| `QFI_bothphases.py` | Continuous QFI across normal → superradiant transition |
| `critical_exponent.py` | Power-law F_Q ∝ \|1 − γ/γ_c\|^(−2) |
| `heatmap_combined.py` | Full QFI landscape in the (γ, ω) plane |
| `fidelity_numericalVSanalytical.py` | SW analytical ground state matches numerics |
| `entanglement-entropy.py` | Von Neumann entropy peaks at the QPT |
| `waterfall.py` | 3D view of QFI divergence for multiple ω values |

---

## Method Overview

The **Quantum Fisher Information** is computed via centered finite differences of the ground state:

```
F_Q(g) = 4 ( ⟨∂_g ψ | ∂_g ψ⟩ − |⟨ψ | ∂_g ψ⟩|² )
```

where the derivative is approximated as:

```
|∂_g ψ⟩ ≈ ( |ψ(g + δg)⟩ − |ψ(g − δg)⟩ ) / (2 δg)
```

with global phase alignment at each step to ensure numerical stability.

In the **superradiant phase**, the Hamiltonian is expressed in a displaced-squeezed frame to improve Fock-space convergence dramatically for γ > γ_c.

---

## License

This project is open source. Feel free to use, modify, and distribute the code. If you use it in your own work, a citation or acknowledgement would be appreciated!

---

## Contributing

Contributions, bug reports, and suggestions are welcome! Feel free to:

- Open an [Issue](https://github.com/Landerzl/Critical-Metrology/issues)
- Submit a [Pull Request](https://github.com/Landerzl/Critical-Metrology/pulls)

---

**Made with QuTiP** — *If you found this useful, consider giving the repo a star!*
