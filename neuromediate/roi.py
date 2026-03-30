"""
ROI-level mediation analysis.

Accepts CSV / DataFrame input with columns for exposure, brain ROI
measures, behavioural outcomes, and covariates.  Supports single-ROI
and multi-ROI analyses with Benjamini–Hochberg FDR correction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .core import MediationResult, mediation_analysis, sensitivity_analysis


# =========================================================================== #
#  FDR (Benjamini–Hochberg)                                                   #
# =========================================================================== #

def _fdr_bh(p: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR correction.  Returns q-values."""
    m = len(p)
    order = np.argsort(p)
    sorted_p = p[order]
    q = np.empty(m)
    q[m - 1] = sorted_p[m - 1]
    for i in range(m - 2, -1, -1):
        q[i] = min(q[i + 1], sorted_p[i] * m / (i + 1))
    out = np.empty(m)
    out[order] = q
    return out


# =========================================================================== #
#  Public API                                                                 #
# =========================================================================== #

def roi_mediation(
    data: Union[pd.DataFrame, str, Path],
    exposure: str,
    mediators: Union[str, List[str]],
    outcome: str,
    covariates: Optional[Union[str, List[str]]] = None,
    n_boot: int = 5000,
    ci_level: float = 0.95,
    fdr_correct: bool = True,
    seed: Optional[int] = None,
) -> Dict[str, MediationResult]:
    """Run mediation analysis for one or more ROI mediators.

    Parameters
    ----------
    data : DataFrame or path to CSV/TSV.
    exposure : column name for X.
    mediators : column name(s) for M.
    outcome : column name for Y.
    covariates : column name(s) for confounders.
    n_boot : bootstrap iterations (default 5 000).
    ci_level : confidence level (default 0.95).
    fdr_correct : apply FDR across mediators (default True).
    seed : random seed.

    Returns
    -------
    dict  {mediator_name: MediationResult}
    """
    # ----- load ---------------------------------------------------------------
    if isinstance(data, (str, Path)):
        p = Path(data)
        sep = "\t" if p.suffix in (".tsv", ".tab") else ","
        data = pd.read_csv(p, sep=sep)

    if isinstance(mediators, str):
        mediators = [mediators]
    if isinstance(covariates, str):
        covariates = [covariates]

    # ----- validate -----------------------------------------------------------
    needed = [exposure, outcome] + mediators + (covariates or [])
    missing = [c for c in needed if c not in data.columns]
    if missing:
        raise ValueError(f"Columns not found in data: {missing}")

    X = data[exposure].values
    Y = data[outcome].values
    cov = data[covariates].values if covariates else None

    # ----- run ----------------------------------------------------------------
    results: Dict[str, MediationResult] = {}
    pvals = []

    for med in mediators:
        M = data[med].values
        res = mediation_analysis(
            X, M, Y,
            covariates=cov,
            n_boot=n_boot,
            ci_level=ci_level,
            seed=seed,
        )
        results[med] = res
        pvals.append(res.sobel_pval)

    # ----- FDR ----------------------------------------------------------------
    if fdr_correct and len(mediators) > 1:
        qvals = _fdr_bh(np.array(pvals))
        for i, med in enumerate(mediators):
            results[med].fdr_pval = qvals[i]                      # type: ignore[attr-defined]
            results[med].fdr_significant = qvals[i] < (1 - ci_level)  # type: ignore[attr-defined]

    return results


def roi_sensitivity(
    data: Union[pd.DataFrame, str, Path],
    exposure: str,
    mediator: str,
    outcome: str,
    covariates: Optional[Union[str, List[str]]] = None,
    rho_range: Optional[np.ndarray] = None,
    n_boot: int = 1000,
    seed: Optional[int] = None,
) -> dict:
    """Sensitivity analysis for a single ROI mediator.

    Thin wrapper around :func:`core.sensitivity_analysis` that accepts
    a DataFrame / CSV path instead of raw arrays.
    """
    if isinstance(data, (str, Path)):
        data = pd.read_csv(data)
    if isinstance(covariates, str):
        covariates = [covariates]

    X = data[exposure].values
    M = data[mediator].values
    Y = data[outcome].values
    cov = data[covariates].values if covariates else None

    return sensitivity_analysis(
        X, M, Y,
        covariates=cov,
        rho_range=rho_range,
        n_boot=n_boot,
        seed=seed,
    )


def results_to_dataframe(results: Dict[str, MediationResult]) -> pd.DataFrame:
    """Convert ``roi_mediation()`` output to a summary DataFrame."""
    rows = []
    for name, r in results.items():
        row = dict(
            mediator=name,
            a_path=r.a_path, a_pval=r.a_pval,
            b_path=r.b_path, b_pval=r.b_pval,
            c_path=r.c_path, c_pval=r.c_pval,
            c_prime=r.c_prime, c_prime_pval=r.c_prime_pval,
            indirect_effect=r.indirect_effect,
            indirect_ci_lo=r.indirect_ci[0],
            indirect_ci_hi=r.indirect_ci[1],
            significant=r.significant,
            proportion_mediated=r.proportion_mediated,
            sobel_z=r.sobel_z, sobel_pval=r.sobel_pval,
            r2_m_model=r.r2_m_model,
            r2_y_full=r.r2_y_full,
            n=r.n_obs,
        )
        if hasattr(r, "fdr_pval"):
            row["fdr_pval"] = r.fdr_pval
            row["fdr_significant"] = r.fdr_significant
        rows.append(row)
    return pd.DataFrame(rows)
