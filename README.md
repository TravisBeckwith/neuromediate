# NeuroMediate [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20088336.svg)](https://doi.org/10.5281/zenodo.20088336)

**Neuroimaging mediation analysis toolkit** — test whether brain measures mediate exposure–behaviour relationships.

NeuroMediate provides a clean Python API for running mediation analysis at three spatial resolutions:

| Level | Input | Use case |
|-------|-------|----------|
| **ROI** | CSV / DataFrame | Hypothesis-driven: mean FA in a tract, regional cortical thickness |
| **Voxelwise** | 4-D NIfTI | Exploratory: mass-univariate mediation across the whole brain |
| **Along-tract** | Tract profiles | Spatially resolved: mediation at each node along a white-matter pathway |

All levels use **bootstrap inference** with bias-corrected and accelerated (BCa) confidence intervals for the indirect effect.

---

## Installation

```bash
# Core only (ROI-level — numpy, scipy, pandas)
pip install neuromediate

# With NIfTI support (voxelwise)
pip install neuromediate[neuroimaging]

# With parallel processing
pip install neuromediate[parallel]

# Everything
pip install neuromediate[full]
```

From source:

```bash
git clone https://github.com/travbeckwith/neuromediate.git
cd neuromediate
pip install -e ".[full]"
```

---

## Quick Start

### ROI-Level

```python
from neuromediate import roi_mediation, generate_html_report

results = roi_mediation(
    data="subjects.csv",
    exposure="blood_lead_level",
    mediators=["pfc_volume", "hippocampus_volume", "amygdala_volume"],
    outcome="psychopathy_score",
    covariates=["age", "sex", "ses", "total_brain_volume"],
    n_boot=5000,
    seed=42,
)

for name, res in results.items():
    print(res.summary())

generate_html_report(results, "report.html",
    title="Lead → Brain → Psychopathy",
    labels={"x": "Blood Lead", "m": "Brain ROI", "y": "Psychopathy"})
```

### Sensitivity Analysis

```python
from neuromediate import roi_sensitivity, plot_sensitivity

sens = roi_sensitivity(
    "subjects.csv",
    exposure="blood_lead_level",
    mediator="pfc_volume",
    outcome="psychopathy_score",
    covariates=["age", "sex", "ses"],
)
print(f"Indirect effect crosses zero at ρ = {sens['rho_at_zero']:.3f}")
plot_sensitivity(sens, output_path="sensitivity.png")
```

### Voxelwise

```python
import numpy as np
from neuromediate import voxelwise_mediation

result = voxelwise_mediation(
    brain_4d="gray_matter_4d.nii.gz",
    X=np.loadtxt("exposure.txt"),
    Y=np.loadtxt("behavior.txt"),
    mask="brain_mask.nii.gz",
    covariates=np.loadtxt("covariates.txt"),
    n_boot=5000,
    n_jobs=-1,
    seed=42,
)

result.save_maps("output/voxelwise/", prefix="lead_psychopathy")
fdr_map = result.apply_fdr(q=0.05)
cluster_map = result.cluster_threshold(primary_p=0.001, min_cluster_size=20)
```

### Along-Tract

```python
import numpy as np
from neuromediate import tract_mediation, multi_tract_mediation, plot_tract_mediation

result = tract_mediation(
    tract_profiles=np.loadtxt("uncinate_fa.csv", delimiter=","),
    X=np.loadtxt("exposure.txt"),
    Y=np.loadtxt("behavior.txt"),
    covariates=np.loadtxt("covariates.txt"),
    tract_name="Uncinate Fasciculus (FA)",
    n_boot=5000,
    seed=42,
)
print(result.summary())
plot_tract_mediation(result, output_path="uncinate_mediation.png")

# Multiple tracts at once
all_results = multi_tract_mediation(
    {"Uncinate (FA)": "uncinate_fa.csv",
     "Cingulum (FA)": "cingulum_fa.csv"},
    X=X, Y=Y, covariates=covariates,
)
```

---

## The Model

```
        a           b
  X --------→ M --------→ Y
  |                        ↑
  |          c'            |
  └────────────────────────┘
```

| Path | Meaning |
|------|---------|
| **a** | Effect of exposure (X) on brain measure (M) |
| **b** | Effect of brain (M) on behaviour (Y), controlling for X |
| **c** | Total effect of X on Y |
| **c'** | Direct effect of X on Y, controlling for M |
| **a × b** | **Indirect (mediation) effect** |

The indirect effect is tested via nonparametric bootstrap with BCa confidence intervals (Efron, 1987).  Significant when CI excludes zero.

---

## Outputs

| Format | Function | Description |
|--------|----------|-------------|
| CSV | `save_csv()` | Summary statistics per mediator / node |
| HTML | `generate_html_report()` | Self-contained report with embedded figures |
| NIfTI | `VoxelwiseResult.save_maps()` | Statistical maps (FSLeyes / freeview) |
| PNG | `plot_mediation_diagram()` | Classic path diagram |
| PNG | `plot_bootstrap_distribution()` | Histogram of bootstrap indirect effects |
| PNG | `plot_sensitivity()` | Sensitivity-to-confounding curve |
| PNG | `plot_tract_mediation()` | Along-tract indirect-effect profile |
| PNG | `plot_roi_forest()` | Forest plot comparing multiple ROIs |

---

## API Reference

### Core

```python
mediation_analysis(X, M, Y, covariates=None, n_boot=5000, ci_level=0.95, seed=None)
    → MediationResult

sensitivity_analysis(X, M, Y, covariates=None, rho_range=None, n_boot=1000, seed=None)
    → dict
```

### ROI

```python
roi_mediation(data, exposure, mediators, outcome, covariates=None,
              n_boot=5000, ci_level=0.95, fdr_correct=True, seed=None)
    → dict[str, MediationResult]

roi_sensitivity(data, exposure, mediator, outcome, covariates=None, ...)
    → dict

results_to_dataframe(results)
    → pd.DataFrame
```

### Voxelwise

```python
voxelwise_mediation(brain_4d, X, Y, mask=None, covariates=None,
                    n_boot=5000, ci_level=0.95, n_jobs=1, seed=None)
    → VoxelwiseResult

VoxelwiseResult.save_maps(output_dir, prefix="mediation") → dict[str, str]
VoxelwiseResult.apply_fdr(q=0.05) → np.ndarray
VoxelwiseResult.cluster_threshold(primary_p=0.001, min_cluster_size=10) → np.ndarray
```

### Along-Tract

```python
tract_mediation(tract_profiles, X, Y, covariates=None,
                tract_name="tract", n_boot=5000, ci_level=0.95,
                fdr_q=0.05, seed=None)
    → TractMediationResult

multi_tract_mediation(tract_dict, X, Y, covariates=None, ...)
    → dict[str, TractMediationResult]

tract_results_to_dataframe(result) → pd.DataFrame
```

---

## Dependencies

| Group | Packages |
|-------|----------|
| Core | `numpy ≥ 1.20`, `scipy ≥ 1.7`, `pandas ≥ 1.3` |
| Neuroimaging | `nibabel ≥ 3.0`, `nilearn ≥ 0.9` |
| Parallel | `joblib ≥ 1.0` |
| Plotting | `matplotlib ≥ 3.4` |

---

## Methodological Notes

**Cross-sectional mediation** cannot establish temporal ordering.  The model assumes X → M → Y, but a single time-point cannot verify this.  Longitudinal designs are strongly preferred.

**Sensitivity analysis** (`sensitivity_analysis` / `roi_sensitivity`) quantifies how robust the indirect effect is to unmeasured confounding of the M–Y relationship.  The `rho_at_zero` value tells you what residual correlation between the M and Y error terms would be needed to eliminate the mediation effect.  Larger |ρ| at zero = more robust.

**Multiple comparisons** apply to voxelwise and multi-ROI analyses.  `VoxelwiseResult.apply_fdr()` and `roi_mediation(fdr_correct=True)` handle this via Benjamini–Hochberg.  Cluster-extent thresholding is also available for voxelwise maps.

---

## Citation

```bibtex
@software{neuromediate,
  author  = {Beckwith, Travis},
  title   = {NeuroMediate: Neuroimaging Mediation Analysis Toolkit},
  url     = {https://github.com/travisbeckwith/neuromediate},
  version = {0.1.0},
}
```

## License

MIT
