"""
Along-tract mediation analysis.

Runs the mediation model at each node of a white-matter tract profile,
testing whether exposure-related microstructural change at each
location mediates the exposure → behaviour link.

Input formats
-------------
* NumPy array  (n_subjects × n_nodes)
* pandas DataFrame
* CSV path  (subjects in rows, nodes in columns)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .core import MediationResult, mediation_analysis
from .roi import _fdr_bh


# =========================================================================== #
#  Result container                                                           #
# =========================================================================== #

@dataclass
class TractMediationResult:
    """Results from along-tract mediation."""

    tract_name: str = ""
    n_nodes: int = 0
    node_results: List[MediationResult] = field(default_factory=list)
    indirect_effects: np.ndarray = field(default_factory=lambda: np.array([]))
    ci_lower: np.ndarray = field(default_factory=lambda: np.array([]))
    ci_upper: np.ndarray = field(default_factory=lambda: np.array([]))
    p_values: np.ndarray = field(default_factory=lambda: np.array([]))
    a_paths: np.ndarray = field(default_factory=lambda: np.array([]))
    b_paths: np.ndarray = field(default_factory=lambda: np.array([]))
    significant_nodes: np.ndarray = field(default_factory=lambda: np.array([]))
    fdr_significant: np.ndarray = field(default_factory=lambda: np.array([]))

    def summary(self) -> str:
        n_sig = int(np.sum(self.significant_nodes))
        n_fdr = int(np.sum(self.fdr_significant))
        peak = int(np.argmax(np.abs(self.indirect_effects)))
        return "\n".join([
            "=" * 60,
            f"  Along-Tract Mediation: {self.tract_name}",
            "=" * 60,
            f"  Nodes analysed       : {self.n_nodes}",
            f"  Significant (uncorr) : {n_sig} / {self.n_nodes}",
            f"  Significant (FDR)    : {n_fdr} / {self.n_nodes}",
            f"  Peak indirect effect : node {peak}"
            f"  (ab = {self.indirect_effects[peak]:.4f})",
            "=" * 60,
        ])


# =========================================================================== #
#  Public API                                                                 #
# =========================================================================== #

def tract_mediation(
    tract_profiles: Union[np.ndarray, pd.DataFrame, str, Path],
    X: np.ndarray,
    Y: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    tract_name: str = "tract",
    n_boot: int = 5000,
    ci_level: float = 0.95,
    fdr_q: float = 0.05,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> TractMediationResult:
    """Run mediation at every node along a tract profile.

    Parameters
    ----------
    tract_profiles : (n_subjects, n_nodes) array, DataFrame, or CSV path.
    X : (n_subjects,) exposure.
    Y : (n_subjects,) outcome.
    covariates : (n_subjects, q) optional confounders.
    tract_name : label for reporting.
    n_boot : bootstrap iterations.
    ci_level : CI level.
    fdr_q : FDR threshold.
    seed : random seed.
    verbose : print progress.

    Returns
    -------
    TractMediationResult
    """
    # load
    if isinstance(tract_profiles, (str, Path)):
        tract_profiles = pd.read_csv(tract_profiles).values
    elif isinstance(tract_profiles, pd.DataFrame):
        tract_profiles = tract_profiles.values
    tract_profiles = np.asarray(tract_profiles, dtype=np.float64)

    ns, nn = tract_profiles.shape
    X = np.asarray(X, dtype=np.float64).ravel()
    Y = np.asarray(Y, dtype=np.float64).ravel()
    if len(X) != ns or len(Y) != ns:
        raise ValueError(
            f"Subject count mismatch: profiles={ns}, X={len(X)}, Y={len(Y)}"
        )

    if covariates is not None:
        covariates = np.asarray(covariates, dtype=np.float64)
        if covariates.ndim == 1:
            covariates = covariates[:, np.newaxis]

    node_results: List[MediationResult] = []
    ab = np.zeros(nn)
    ci_lo = np.zeros(nn)
    ci_hi = np.zeros(nn)
    pvals = np.ones(nn)
    a_arr = np.zeros(nn)
    b_arr = np.zeros(nn)

    step = max(1, nn // 10)

    for node in range(nn):
        res = mediation_analysis(
            X, tract_profiles[:, node], Y,
            covariates=covariates,
            n_boot=n_boot, ci_level=ci_level, seed=seed,
        )
        node_results.append(res)
        ab[node] = res.indirect_effect
        ci_lo[node] = res.indirect_ci[0]
        ci_hi[node] = res.indirect_ci[1]
        pvals[node] = res.sobel_pval
        a_arr[node] = res.a_path
        b_arr[node] = res.b_path

        if verbose and (node + 1) % step == 0:
            print(f"  {tract_name}: node {node + 1}/{nn}")

    sig = np.array([r.significant for r in node_results])
    fdr_p = _fdr_bh(pvals)
    fdr_sig = fdr_p < fdr_q

    return TractMediationResult(
        tract_name=tract_name,
        n_nodes=nn,
        node_results=node_results,
        indirect_effects=ab,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        p_values=pvals,
        a_paths=a_arr,
        b_paths=b_arr,
        significant_nodes=sig,
        fdr_significant=fdr_sig,
    )


def multi_tract_mediation(
    tract_dict: Dict[str, Union[np.ndarray, str, Path]],
    X: np.ndarray,
    Y: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    n_boot: int = 5000,
    ci_level: float = 0.95,
    fdr_q: float = 0.05,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, TractMediationResult]:
    """Run along-tract mediation for multiple tracts.

    Parameters
    ----------
    tract_dict : {tract_name: profiles_array_or_csv_path}
    X, Y, covariates, n_boot, ci_level, fdr_q, seed : see tract_mediation.

    Returns
    -------
    dict  {tract_name: TractMediationResult}
    """
    out: Dict[str, TractMediationResult] = {}
    for name, profiles in tract_dict.items():
        if verbose:
            print(f"\n--- {name} ---")
        out[name] = tract_mediation(
            profiles, X, Y,
            covariates=covariates,
            tract_name=name,
            n_boot=n_boot,
            ci_level=ci_level,
            fdr_q=fdr_q,
            seed=seed,
            verbose=verbose,
        )
    return out


def tract_results_to_dataframe(
    result: TractMediationResult,
) -> pd.DataFrame:
    """One row per node."""
    rows = []
    for i, r in enumerate(result.node_results):
        rows.append(dict(
            tract=result.tract_name,
            node=i,
            a_path=r.a_path,
            b_path=r.b_path,
            indirect_effect=r.indirect_effect,
            ci_lower=r.indirect_ci[0],
            ci_upper=r.indirect_ci[1],
            significant=r.significant,
            p_value=result.p_values[i],
            fdr_significant=bool(result.fdr_significant[i]),
            proportion_mediated=r.proportion_mediated,
        ))
    return pd.DataFrame(rows)
