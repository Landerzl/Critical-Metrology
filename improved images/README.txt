Improved figures
================

Final figures used in the arXiv version of the thesis, redrawn at the real column width of the paper
(exactly 3.40 in wide, 7-8 pt Computer Modern, colour-blind-safe palette, fonts embedded, 300 dpi raster in the heatmaps). *.pdf are vector (use these in LaTeX), *.png are previews.

Superradiant-phase figures (fidelity_displaced, fidelity_displaced_squeezed, qfi_heatmap_superradiant,
entanglement_entropy) were regenerated with the CORRECTED code (see Critical-Metrology/CORRECTIONS.md);
they differ from the thesis versions.

file                           script (D:\UNI\TFG ArXiv\scripts)       content
qfi_sw1_vs_sw2                 fig_qfi_sw1_vs_sw2.py                   regular part of the QFI, SW1 vs SW2
fidelity_qrm                   fig_fidelity_qrm.py                     full-QRM fidelity vs cutoff: (a) normal, (b) superradiant
qfi_normal_combined            fig_qfi_normal_combined.py              normal-phase QFI: numerics vs SW1, and slope -2
qfi_heatmap_normal             fig_qfi_heatmap_normal.py               normal-phase QFI heatmap
gs_energy_minima               fig_gs_energy_minima.py                 variational energy f(x)
fidelity_displaced             fig_fidelity_displaced.py               fidelity, displaced basis (corrected)
fidelity_displaced_squeezed    fig_fidelity_displaced_squeezed.py      fidelity, displaced+squeezed basis (corrected)
qfi_heatmap_superradiant       fig_qfi_heatmap_superradiant.py         physical QFI, superradiant phase (corrected)
entanglement_entropy           fig_entanglement_entropy.py             atom-field entanglement entropy (corrected)
