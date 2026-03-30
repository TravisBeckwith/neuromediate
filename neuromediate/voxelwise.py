"""
Voxelwise (mass-univariate) mediation analysis.

Runs the single-mediator mediation model at every brain voxel inside a
mask, producing 3-D statistical maps of a, b, a×b, c, c', p-values,
and confidence intervals.  Maps can be saved as NIfTI for viewing in
FSLeyes / freeview.

Post-hoc correction
-------------------
* FDR (Benjamini–Hochberg) on the indirect-effect p-map.
* Cluster-extent thresholding with a user-defined primary threshold.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np

try:
    import nibabel as nib
except ImportError:          # nibabel optional at import time
    nib = None               # type: ignore[assignment]

from .core import _build_design, _ols_fast


# =========================================================================== #
#  Result container                                                           #
# =========================================================================== #

@dataclass
class VoxelwiseResult:
    """Statistical maps from a voxelwise mediation analysis.

    Every ``*_map`` attribute is a 3-D numpy array in the same space as the
    input 4-D NIfTI.
    """

    a_map: Optional[np.ndarray] = None
    b_map: Optional[np.ndarray] = None
    ab_map: Optional[np.ndarray] = None
    c_map: Optional[np.ndarray] = None
    c_prime_map: Optional[np.ndarray] = None
    p_map: Optional[np.ndarray] = None
    ci_lower_map: Optional[np.ndarray] = None
    ci_upper_map: Optional[np.ndarray] = None
    sig_map: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    affine: Optional[np.ndarray] = None
    header: object = None
    n_obs: int = 0
    n_boot: int = 0
    n_voxels: int = 0

    # ------------------------------------------------------------------ save
    def save_maps(
        self,
        output_dir: str,
        prefix: str = "mediation",
    ) -> Dict[str, str]:
        """Write every map to ``output_dir/prefix_<name>.nii.gz``."""
        if nib is None:
            raise ImportError("nibabel is required to save NIfTI files.")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        names = dict(
            a_path=self.a_map,
            b_path=self.b_map,
            indirect_ab=self.ab_map,
            total_c=self.c_map,
            direct_cprime=self.c_prime_map,
            indirect_pval=self.p_map,
            indirect_ci_lower=self.ci_lower_map,
            indirect_ci_upper=self.ci_upper_map,
            significant=self.sig_map.astype(np.float32) if self.sig_map is not None else None,
        )

        saved: Dict[str, str] = {}
        for tag, arr in names.items():
            if arr is not None:
                img = nib.Nifti1Image(arr.astype(np.float32),
                                      self.affine, self.header)
                fp = out / f"{prefix}_{tag}.nii.gz"
                nib.save(img, str(fp))
                saved[tag] = str(fp)
        return saved

    # ------------------------------------------------------------------ FDR
    def apply_fdr(self, q: float = 0.05) -> np.ndarray:
        """Return a boolean 3-D map of FDR-significant voxels."""
        from .roi import _fdr_bh
        pf = self.p_map[self.mask]
        qf = _fdr_bh(pf)
        out = np.zeros_like(self.mask, dtype=bool)
        out[self.mask] = qf < q
        return out

    # -------------------------------------------------------- cluster thresh
    def cluster_threshold(
        self,
        primary_p: float = 0.001,
        min_cluster_size: int = 10,
    ) -> np.ndarray:
        """Cluster-extent thresholding.

        Returns an integer 3-D map where each surviving cluster has a
        unique label (0 = sub-threshold).
        """
        from scipy.ndimage import label
        thr = (self.p_map < primary_p) & self.mask
        labeled, nc = label(thr)
        out = np.zeros_like(labeled)
        for c in range(1, nc + 1):
            if np.sum(labeled == c) >= min_cluster_size:
                out[labeled == c] = c
        return out

    def summary(self) -> str:
        """One-line summary."""
        n_sig = int(np.sum(self.sig_map)) if self.sig_map is not None else 0
        return (
            f"VoxelwiseResult: {self.n_voxels} voxels tested, "
            f"{self.n_obs} subjects, {self.n_boot} bootstraps, "
            f"{n_sig} significant (uncorrected)"
        )


# =========================================================================== #
#  Single-voxel worker                                                        #
# =========================================================================== #

def _voxel_mediation(
    X: np.ndarray,
    Y: np.ndarray,
    M: np.ndarray,
    covariates: Optional[np.ndarray],
    n_boot: int,
    alpha: float,
    rng: np.random.RandomState,
) -> dict:
    """Run the mediation model at one voxel."""
    n = len(X)
    if np.std(M) < 1e-10:
        return dict(a=0, b=0, ab=0, c=0, cp=0, p=1.0, ci_lo=0, ci_hi=0)

    a = _ols_fast(M, _build_design(X, covariates))[0]
    bc = _ols_fast(Y, _build_design(np.column_stack([M, X]), covariates))
    b, cp = bc[0], bc[1]
    c = _ols_fast(Y, _build_design(X, covariates))[0]
    ab = a * b

    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        Xb, Mb, Yb = X[idx], M[idx], Y[idx]
        covb = covariates[idx] if covariates is not None else None
        a_b = _ols_fast(Mb, _build_design(Xb, covb))[0]
        bc_b = _ols_fast(Yb, _build_design(np.column_stack([Mb, Xb]), covb))
        boot[i] = a_b * bc_b[0]

    ci_lo = float(np.percentile(boot, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    p = min(1.0, 2.0 * np.mean(boot <= 0) if ab >= 0 else 2.0 * np.mean(boot >= 0))

    return dict(a=a, b=b, ab=ab, c=c, cp=cp, p=p, ci_lo=ci_lo, ci_hi=ci_hi)


# =========================================================================== #
#  Serial / parallel runners                                                  #
# =========================================================================== #

def _run_serial(X, Y, vdata, cov, n_boot, alpha, rng, nv, verbose):
    results = []
    step = max(1, nv // 20)
    for i in range(nv):
        results.append(_voxel_mediation(X, Y, vdata[i], cov, n_boot, alpha, rng))
        if verbose and (i + 1) % step == 0:
            print(f"  {100 * (i + 1) / nv:.0f}%  ({i + 1}/{nv})")
    return results


def _run_parallel(X, Y, vdata, cov, n_boot, alpha, seed, n_jobs, nv, verbose):
    try:
        from joblib import Parallel, delayed
    except ImportError:
        warnings.warn("joblib unavailable — falling back to serial.")
        return _run_serial(X, Y, vdata, cov, n_boot, alpha,
                           np.random.RandomState(seed), nv, verbose)

    def _w(i):
        return _voxel_mediation(
            X, Y, vdata[i], cov, n_boot, alpha,
            np.random.RandomState(seed + i if seed else None),
        )

    if verbose:
        print(f"  Dispatching {nv} voxels across {n_jobs} workers …")
    return Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
        delayed(_w)(i) for i in range(nv)
    )


# =========================================================================== #
#  Public entry point                                                         #
# =========================================================================== #

def voxelwise_mediation(
    brain_4d: str,
    X: np.ndarray,
    Y: np.ndarray,
    mask: Optional[str] = None,
    covariates: Optional[np.ndarray] = None,
    n_boot: int = 5000,
    ci_level: float = 0.95,
    n_jobs: int = 1,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> VoxelwiseResult:
    """Mass-univariate mediation across every voxel in a brain mask.

    Parameters
    ----------
    brain_4d : path to a 4-D NIfTI (subjects in the 4th dim).
    X : (n_subjects,) exposure.
    Y : (n_subjects,) outcome.
    mask : path to a binary 3-D NIfTI brain mask (optional).
    covariates : (n_subjects, q) covariates (optional).
    n_boot : bootstrap iterations.
    ci_level : confidence level.
    n_jobs : parallel workers (−1 = all cores).
    seed : random seed.
    verbose : print progress.

    Returns
    -------
    VoxelwiseResult
    """
    if nib is None:
        raise ImportError("nibabel is required.  pip install nibabel")

    img = nib.load(str(brain_4d))
    data = img.get_fdata()
    if data.ndim != 4:
        raise ValueError(f"Expected 4-D NIfTI, got {data.ndim}-D.")

    ns = data.shape[3]
    shape = data.shape[:3]
    X = np.asarray(X, dtype=np.float64).ravel()
    Y = np.asarray(Y, dtype=np.float64).ravel()
    if len(X) != ns or len(Y) != ns:
        raise ValueError(
            f"X ({len(X)}) / Y ({len(Y)}) length ≠ 4th dim ({ns})."
        )

    if covariates is not None:
        covariates = np.asarray(covariates, dtype=np.float64)
        if covariates.ndim == 1:
            covariates = covariates[:, np.newaxis]

    # mask
    if mask is not None:
        mdata = nib.load(str(mask)).get_fdata().astype(bool)
    else:
        mdata = np.mean(data, axis=3) > 0
        if verbose:
            print(f"  Auto mask: {int(np.sum(mdata))} voxels")

    vidx = np.argwhere(mdata)          # (nv, 3)
    nv = len(vidx)
    vdata = data[mdata].T.T           # (nv, ns) — T.T keeps C-order copy
    # Actually reshape properly:
    vdata = data[mdata]               # (nv, ns) when mask applied to first 3 dims

    if verbose:
        print(f"  {nv} voxels × {ns} subjects × {n_boot} bootstraps")

    alpha = 1.0 - ci_level
    rng = np.random.RandomState(seed)

    if n_jobs == 1:
        res = _run_serial(X, Y, vdata, covariates, n_boot, alpha, rng, nv, verbose)
    else:
        res = _run_parallel(X, Y, vdata, covariates, n_boot, alpha, seed,
                            n_jobs, nv, verbose)

    # --- fill maps -----------------------------------------------------------
    a_map = np.zeros(shape, np.float32)
    b_map = np.zeros(shape, np.float32)
    ab_map = np.zeros(shape, np.float32)
    c_map = np.zeros(shape, np.float32)
    cp_map = np.zeros(shape, np.float32)
    p_map = np.ones(shape, np.float32)
    lo_map = np.zeros(shape, np.float32)
    hi_map = np.zeros(shape, np.float32)

    for i, (vi, vj, vk) in enumerate(vidx):
        r = res[i]
        a_map[vi, vj, vk] = r["a"]
        b_map[vi, vj, vk] = r["b"]
        ab_map[vi, vj, vk] = r["ab"]
        c_map[vi, vj, vk] = r["c"]
        cp_map[vi, vj, vk] = r["cp"]
        p_map[vi, vj, vk] = r["p"]
        lo_map[vi, vj, vk] = r["ci_lo"]
        hi_map[vi, vj, vk] = r["ci_hi"]

    sig = ((lo_map > 0) | (hi_map < 0)) & mdata

    return VoxelwiseResult(
        a_map=a_map, b_map=b_map, ab_map=ab_map,
        c_map=c_map, c_prime_map=cp_map,
        p_map=p_map, ci_lower_map=lo_map, ci_upper_map=hi_map,
        sig_map=sig, mask=mdata,
        affine=img.affine, header=img.header,
        n_obs=ns, n_boot=n_boot, n_voxels=nv,
    )
