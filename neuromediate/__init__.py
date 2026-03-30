"""
NeuroMediate — Neuroimaging Mediation Analysis Toolkit
=====================================================

Test whether brain measures mediate exposure → behaviour relationships
at three spatial resolutions:

* **ROI-level** — CSV / DataFrame input, multi-mediator + FDR
* **Voxelwise** — 4-D NIfTI, mass-univariate, parallel bootstrap
* **Along-tract** — tract profiles (AFQ / TractSeg), node-wise FDR

Quick start
-----------
>>> from neuromediate import roi_mediation
>>> results = roi_mediation(
...     "subjects.csv",
...     exposure="lead_level",
...     mediators=["pfc_volume", "hippo_volume"],
...     outcome="psychopathy",
...     covariates=["age", "sex"],
... )
>>> for name, r in results.items():
...     print(f"{name}: ab={r.indirect_effect:.4f}  sig={r.significant}")
"""

__version__ = "0.1.0"
__author__ = "Travis Beckwith"

# Core engine
from .core import mediation_analysis, sensitivity_analysis, MediationResult

# ROI level
from .roi import roi_mediation, roi_sensitivity, results_to_dataframe

# Voxelwise
from .voxelwise import voxelwise_mediation, VoxelwiseResult

# Along-tract
from .tract import (
    tract_mediation,
    multi_tract_mediation,
    TractMediationResult,
    tract_results_to_dataframe,
)

# Reporting
from .reporting import (
    save_csv,
    generate_html_report,
    plot_mediation_diagram,
    plot_bootstrap_distribution,
    plot_sensitivity,
    plot_tract_mediation,
    plot_roi_forest,
)

__all__ = [
    # core
    "mediation_analysis", "sensitivity_analysis", "MediationResult",
    # roi
    "roi_mediation", "roi_sensitivity", "results_to_dataframe",
    # voxelwise
    "voxelwise_mediation", "VoxelwiseResult",
    # tract
    "tract_mediation", "multi_tract_mediation",
    "TractMediationResult", "tract_results_to_dataframe",
    # reporting
    "save_csv", "generate_html_report",
    "plot_mediation_diagram", "plot_bootstrap_distribution",
    "plot_sensitivity", "plot_tract_mediation", "plot_roi_forest",
]
