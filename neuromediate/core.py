"""
Core mediation analysis engine.

Implements the Baron & Kenny causal steps framework with bootstrap inference
for the indirect effect (a × b). Supports single-mediator models using a
counterfactual / potential-outcomes approach.

Statistical inference
---------------------
* Point estimates via OLS (normal equations).
* Bootstrap confidence intervals: percentile **and** bias-corrected-
  accelerated (BCa) are both computed; BCa is the default report.
* Sobel test provided for reference (bootstrap CI is the primary test).
* Sensitivity analysis for unmeasured M–Y confounding (Imai et al., 2010).

References
----------
Baron, R. M., & Kenny, D. A. (1986). J Pers Soc Psychol, 51(6), 1173.
Preacher, K. J., & Hayes, A. F. (2008). Behav Res Methods, 40(3), 879-891.
Imai, K., Keele, L., & Tingley, D. (2010). Stat Sci, 25(1), 51-71.
Efron, B. (1987). J Am Stat Assoc, 82(397), 171-185.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy import stats as sp_stats


# =========================================================================== #
#  Result container                                                           #
# =========================================================================== #

@dataclass
class MediationResult:
    """Container for a single-mediator mediation analysis.

    Every field is documented in the class body.  The object is intentionally
    a plain dataclass so it can be serialised, compared, and inspected easily.
    """

    # --- path coefficients ---------------------------------------------------
    a_path: float = 0.0
    """Effect of X on M (exposure → brain)."""
    a_se: float = 0.0
    a_pval: float = 1.0

    b_path: float = 0.0
    """Effect of M on Y, controlling for X (brain → behavior | exposure)."""
    b_se: float = 0.0
    b_pval: float = 1.0

    c_path: float = 0.0
    """Total effect of X on Y (exposure → behavior)."""
    c_se: float = 0.0
    c_pval: float = 1.0

    c_prime: float = 0.0
    """Direct effect of X on Y, controlling for M."""
    c_prime_se: float = 0.0
    c_prime_pval: float = 1.0

    # --- indirect effect -----------------------------------------------------
    indirect_effect: float = 0.0
    """Point estimate of the indirect (mediation) effect  a × b."""
    indirect_ci: Tuple[float, float] = (0.0, 0.0)
    """BCa bootstrap confidence interval for the indirect effect."""
    indirect_ci_percentile: Tuple[float, float] = (0.0, 0.0)
    """Simple percentile bootstrap CI (for comparison)."""
    proportion_mediated: float = 0.0
    """ab / c.  NaN when c ≈ 0."""

    # --- bootstrap -----------------------------------------------------------
    boot_distribution: np.ndarray = field(default_factory=lambda: np.array([]))
    """Full bootstrap distribution of indirect effects."""
    n_obs: int = 0
    n_boot: int = 0
    ci_level: float = 0.95

    # --- Sobel ---------------------------------------------------------------
    sobel_z: float = 0.0
    sobel_pval: float = 1.0

    # --- model diagnostics ---------------------------------------------------
    r2_m_model: float = 0.0
    """R² for the M ~ X + covariates model (a-path model)."""
    r2_y_full: float = 0.0
    """R² for Y ~ M + X + covariates (full model)."""
    r2_y_reduced: float = 0.0
    """R² for Y ~ X + covariates (without mediator)."""

    # ---- convenience --------------------------------------------------------
    @property
    def significant(self) -> bool:
        """Whether the BCa CI excludes zero."""
        return not (self.indirect_ci[0] <= 0 <= self.indirect_ci[1])

    def summary(self) -> str:
        """Human-readable summary table."""
        lines = [
            "=" * 65,
            "  Mediation Analysis Results",
            "=" * 65,
            f"  N = {self.n_obs}    Bootstrap iterations = {self.n_boot}"
            f"    CI level = {self.ci_level * 100:.0f}%",
            "-" * 65,
            "  Path coefficients:",
            f"    a   (X → M)   = {self.a_path:>9.4f}"
            f"   SE = {self.a_se:.4f}   p = {self.a_pval:.4e}",
            f"    b   (M → Y|X) = {self.b_path:>9.4f}"
            f"   SE = {self.b_se:.4f}   p = {self.b_pval:.4e}",
            f"    c   (X → Y)   = {self.c_path:>9.4f}"
            f"   SE = {self.c_se:.4f}   p = {self.c_pval:.4e}",
            f"    c'  (X → Y|M) = {self.c_prime:>9.4f}"
            f"   SE = {self.c_prime_se:.4f}   p = {self.c_prime_pval:.4e}",
            "-" * 65,
            "  Indirect effect (a × b):",
            f"    Point estimate   = {self.indirect_effect:>9.4f}",
            f"    BCa CI           = [{self.indirect_ci[0]:.4f},"
            f" {self.indirect_ci[1]:.4f}]",
            f"    Percentile CI    = [{self.indirect_ci_percentile[0]:.4f},"
            f" {self.indirect_ci_percentile[1]:.4f}]",
            f"    Significant      = {self.significant}",
            f"    Prop. mediated   = {self.proportion_mediated:>9.4f}",
            f"    Sobel z = {self.sobel_z:.4f}   p = {self.sobel_pval:.4e}",
            "-" * 65,
            "  Model fit:",
            f"    R²  M-model  = {self.r2_m_model:.4f}",
            f"    R²  Y-full   = {self.r2_y_full:.4f}"
            f"    R²  Y-reduced = {self.r2_y_reduced:.4f}",
            "=" * 65,
        ]
        return "\n".join(lines)


# =========================================================================== #
#  Internal OLS helpers                                                       #
# =========================================================================== #

def _ols_fit(Y: np.ndarray, X: np.ndarray):
    """OLS via normal equations.

    Returns (betas, residuals, se, t_stats, p_vals, r_squared).
    """
    n, k = X.shape
    betas, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)

    fitted = X @ betas
    residuals = Y - fitted
    dof = n - k

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    if dof <= 0:
        warnings.warn("Degrees of freedom ≤ 0.  Results unreliable.")
        nan_k = np.full(k, np.nan)
        return betas, residuals, nan_k, nan_k, nan_k, r2

    mse = ss_res / dof
    try:
        cov = mse * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        nan_k = np.full(k, np.nan)
        return betas, residuals, nan_k, nan_k, nan_k, r2

    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = np.where(se > 0, betas / se, 0.0)
    p_vals = 2.0 * sp_stats.t.sf(np.abs(t_stats), dof)

    return betas, residuals, se, t_stats, p_vals, r2


def _ols_fast(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Minimal OLS — returns only *betas* for bootstrap speed."""
    betas, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    return betas


def _build_design(
    X_var: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    add_intercept: bool = True,
) -> np.ndarray:
    """Assemble design matrix:  [X_var | covariates | 1]."""
    n = X_var.shape[0]
    if X_var.ndim == 1:
        X_var = X_var[:, np.newaxis]
    parts = [X_var]
    if covariates is not None:
        c = np.asarray(covariates, dtype=np.float64)
        if c.ndim == 1:
            c = c[:, np.newaxis]
        parts.append(c)
    if add_intercept:
        parts.append(np.ones((n, 1)))
    return np.column_stack(parts)


# =========================================================================== #
#  BCa confidence interval                                                    #
# =========================================================================== #

def _bca_ci(
    boot_dist: np.ndarray,
    observed: float,
    X: np.ndarray,
    M: np.ndarray,
    Y: np.ndarray,
    covariates: Optional[np.ndarray],
    alpha: float,
) -> Tuple[float, float]:
    """Bias-corrected and accelerated bootstrap CI.

    Falls back to percentile CI if numerical issues arise.
    """
    n = len(X)

    # --- bias correction -----------------------------------------------------
    z0 = sp_stats.norm.ppf(np.mean(boot_dist < observed))
    if not np.isfinite(z0):
        lo = np.percentile(boot_dist, 100 * alpha / 2)
        hi = np.percentile(boot_dist, 100 * (1 - alpha / 2))
        return (lo, hi)

    # --- acceleration (jackknife) --------------------------------------------
    jack = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        cov_j = covariates[mask] if covariates is not None else None
        a_j = _ols_fast(M[mask], _build_design(X[mask], cov_j))[0]
        bc_j = _ols_fast(
            Y[mask],
            _build_design(np.column_stack([M[mask], X[mask]]), cov_j),
        )
        jack[i] = a_j * bc_j[0]

    jm = np.mean(jack)
    jd = jm - jack
    denom = np.sum(jd ** 2) ** 1.5
    acc = np.sum(jd ** 3) / (6.0 * denom) if denom > 0 else 0.0

    # --- adjusted percentiles ------------------------------------------------
    za_lo = sp_stats.norm.ppf(alpha / 2)
    za_hi = sp_stats.norm.ppf(1 - alpha / 2)

    def _pct(za):
        num = z0 + za
        den = 1 - acc * num
        if abs(den) < 1e-10:
            return alpha / 2 if za < 0 else 1 - alpha / 2
        return sp_stats.norm.cdf(z0 + num / den)

    p_lo = np.clip(_pct(za_lo), 0.001, 0.999)
    p_hi = np.clip(_pct(za_hi), 0.001, 0.999)

    return (
        float(np.percentile(boot_dist, 100 * p_lo)),
        float(np.percentile(boot_dist, 100 * p_hi)),
    )


# =========================================================================== #
#  Main mediation function                                                    #
# =========================================================================== #

def mediation_analysis(
    X: np.ndarray,
    M: np.ndarray,
    Y: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    n_boot: int = 5000,
    ci_level: float = 0.95,
    seed: Optional[int] = None,
) -> MediationResult:
    """Single-mediator mediation with bootstrap inference.

    Model
    -----
    Path a :  M  =  a · X  +  covariates  +  ε_M
    Path b :  Y  =  b · M  +  c' · X  +  covariates  +  ε_Y
    Path c :  Y  =  c · X  +  covariates  +  ε_Y
    Indirect effect  =  a × b
    Direct effect    =  c'
    Total effect     =  c  ≈  c' + a · b

    Parameters
    ----------
    X : (n,) array — independent variable (e.g. exposure).
    M : (n,) array — mediator (e.g. brain measure).
    Y : (n,) array — outcome (e.g. behaviour).
    covariates : (n, q) array, optional — confounders.
    n_boot : int — bootstrap iterations (default 5 000).
    ci_level : float — confidence level (default 0.95).
    seed : int, optional — for reproducibility.

    Returns
    -------
    MediationResult
    """
    rng = np.random.RandomState(seed)

    # --- validate & clean ----------------------------------------------------
    X = np.asarray(X, dtype=np.float64).ravel()
    M = np.asarray(M, dtype=np.float64).ravel()
    Y = np.asarray(Y, dtype=np.float64).ravel()
    n = len(X)
    if len(M) != n or len(Y) != n:
        raise ValueError("X, M, Y must have the same length.")

    valid = np.isfinite(X) & np.isfinite(M) & np.isfinite(Y)
    if covariates is not None:
        covariates = np.asarray(covariates, dtype=np.float64)
        if covariates.ndim == 1:
            covariates = covariates[:, np.newaxis]
        valid &= np.all(np.isfinite(covariates), axis=1)
        covariates = covariates[valid]
    X, M, Y = X[valid], M[valid], Y[valid]
    n = len(X)

    if n < 10:
        warnings.warn(f"Very small sample (n={n}).  Results may be unreliable.")

    # --- point estimates -----------------------------------------------------
    # a-path:  M = a·X + cov
    des_a = _build_design(X, covariates)
    b_a, _, se_a, _, p_a, r2_m = _ols_fit(M, des_a)
    a = b_a[0]

    # b-path + c':  Y = b·M + c'·X + cov
    des_bc = _build_design(np.column_stack([M, X]), covariates)
    b_bc, _, se_bc, _, p_bc, r2_y_full = _ols_fit(Y, des_bc)
    b = b_bc[0]
    cprime = b_bc[1]

    # c-path (total):  Y = c·X + cov
    des_c = _build_design(X, covariates)
    b_c, _, se_c, _, p_c, r2_y_red = _ols_fit(Y, des_c)
    c = b_c[0]

    ab = a * b
    prop = ab / c if abs(c) > 1e-10 else np.nan

    # Sobel
    sob_se = np.sqrt(a ** 2 * se_bc[0] ** 2 + b ** 2 * se_a[0] ** 2)
    sob_z = ab / sob_se if sob_se > 0 else 0.0
    sob_p = 2.0 * sp_stats.norm.sf(abs(sob_z))

    # --- bootstrap -----------------------------------------------------------
    alpha = 1.0 - ci_level
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        Xb, Mb, Yb = X[idx], M[idx], Y[idx]
        covb = covariates[idx] if covariates is not None else None
        a_b = _ols_fast(Mb, _build_design(Xb, covb))[0]
        bc_b = _ols_fast(Yb, _build_design(np.column_stack([Mb, Xb]), covb))
        boot[i] = a_b * bc_b[0]

    pct_ci = (
        float(np.percentile(boot, 100 * alpha / 2)),
        float(np.percentile(boot, 100 * (1 - alpha / 2))),
    )
    bca_ci = _bca_ci(boot, ab, X, M, Y, covariates, alpha)

    return MediationResult(
        a_path=a, a_se=se_a[0], a_pval=p_a[0],
        b_path=b, b_se=se_bc[0], b_pval=p_bc[0],
        c_path=c, c_se=se_c[0], c_pval=p_c[0],
        c_prime=cprime, c_prime_se=se_bc[1], c_prime_pval=p_bc[1],
        indirect_effect=ab,
        indirect_ci=bca_ci,
        indirect_ci_percentile=pct_ci,
        proportion_mediated=prop,
        boot_distribution=boot,
        n_obs=n, n_boot=n_boot, ci_level=ci_level,
        sobel_z=sob_z, sobel_pval=sob_p,
        r2_m_model=r2_m,
        r2_y_full=r2_y_full,
        r2_y_reduced=r2_y_red,
    )


# =========================================================================== #
#  Sensitivity analysis                                                       #
# =========================================================================== #

def sensitivity_analysis(
    X: np.ndarray,
    M: np.ndarray,
    Y: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    rho_range: Optional[np.ndarray] = None,
    n_boot: int = 1000,
    seed: Optional[int] = None,
) -> dict:
    """Sensitivity analysis for unmeasured M–Y confounding.

    Adjusts the b-path coefficient for a range of hypothetical correlations
    (ρ) between the M-model and Y-model residuals — a proxy for unmeasured
    confounding of the mediator–outcome relationship.

    Returns
    -------
    dict with keys: rho, indirect_effect, ci_lower, ci_upper, rho_at_zero.
    """
    if rho_range is None:
        rho_range = np.arange(-0.9, 0.95, 0.05)

    X = np.asarray(X, dtype=np.float64).ravel()
    M = np.asarray(M, dtype=np.float64).ravel()
    Y = np.asarray(Y, dtype=np.float64).ravel()

    valid = np.isfinite(X) & np.isfinite(M) & np.isfinite(Y)
    if covariates is not None:
        covariates = np.asarray(covariates, dtype=np.float64)
        if covariates.ndim == 1:
            covariates = covariates[:, np.newaxis]
        valid &= np.all(np.isfinite(covariates), axis=1)
        covariates = covariates[valid]
    X, M, Y = X[valid], M[valid], Y[valid]
    n = len(X)
    rng = np.random.RandomState(seed)

    des_a = _build_design(X, covariates)
    b_a, res_m, *_ = _ols_fit(M, des_a)

    des_bc = _build_design(np.column_stack([M, X]), covariates)
    b_bc, res_y, *_ = _ols_fit(Y, des_bc)

    sig_m = np.std(res_m)
    sig_y = np.std(res_y)

    ab_arr, ci_lo_arr, ci_hi_arr = [], [], []

    for rho in rho_range:
        adj = rho * sig_y / sig_m if sig_m > 0 else 0.0
        ab_adj = b_a[0] * (b_bc[0] - adj)

        boots = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.choice(n, n, replace=True)
            Xb, Mb, Yb = X[idx], M[idx], Y[idx]
            covb = covariates[idx] if covariates is not None else None

            ba_b, rm_b, *_ = _ols_fit(Mb, _build_design(Xb, covb))
            bbc_b, ry_b, *_ = _ols_fit(
                Yb, _build_design(np.column_stack([Mb, Xb]), covb)
            )
            sm = np.std(rm_b)
            sy = np.std(ry_b)
            adj_b = rho * sy / sm if sm > 0 else 0.0
            boots[i] = ba_b[0] * (bbc_b[0] - adj_b)

        ab_arr.append(ab_adj)
        ci_lo_arr.append(float(np.percentile(boots, 2.5)))
        ci_hi_arr.append(float(np.percentile(boots, 97.5)))

    ab_arr = np.asarray(ab_arr)
    ci_lo_arr = np.asarray(ci_lo_arr)
    ci_hi_arr = np.asarray(ci_hi_arr)

    # find rho where ab crosses zero (linear interpolation)
    sc = np.where(np.diff(np.sign(ab_arr)))[0]
    if len(sc):
        i = sc[0]
        rho_z = rho_range[i] - ab_arr[i] * (
            (rho_range[i + 1] - rho_range[i]) /
            (ab_arr[i + 1] - ab_arr[i])
        )
    else:
        rho_z = np.nan

    return dict(
        rho=rho_range,
        indirect_effect=ab_arr,
        ci_lower=ci_lo_arr,
        ci_upper=ci_hi_arr,
        rho_at_zero=float(rho_z),
    )
