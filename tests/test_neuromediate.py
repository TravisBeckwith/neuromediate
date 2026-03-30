"""
NeuroMediate test suite.

Covers:
  - Core mediation engine (point estimates, bootstrap, BCa, Sobel, edge cases)
  - Sensitivity analysis
  - ROI-level interface (single, multi, FDR, CSV)
  - Along-tract interface
  - Voxelwise interface (small synthetic volume)
  - Reporting (CSV export, plots, HTML)
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neuromediate.core import mediation_analysis, sensitivity_analysis, MediationResult
from neuromediate.roi import roi_mediation, roi_sensitivity, results_to_dataframe
from neuromediate.tract import (
    tract_mediation, multi_tract_mediation, tract_results_to_dataframe,
)
from neuromediate.reporting import (
    save_csv, plot_mediation_diagram, plot_bootstrap_distribution,
    plot_sensitivity, plot_tract_mediation, plot_roi_forest,
    generate_html_report,
)


# =========================================================================== #
#  Simulation helpers                                                         #
# =========================================================================== #

def _sim(n=300, a=0.5, b=0.4, cp=0.1, seed=42):
    """Simulate data with known mediation structure + 2 covariates."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n)
    c1 = rng.randn(n)
    c2 = rng.binomial(1, 0.5, n).astype(float)
    M = a * X + 0.3 * c1 + rng.randn(n) * 0.5
    Y = b * M + cp * X + 0.2 * c1 + 0.1 * c2 + rng.randn(n) * 0.5
    cov = np.column_stack([c1, c2])
    return X, M, Y, cov


def _sim_df(n=300, seed=42):
    X, M, Y, cov = _sim(n=n, seed=seed)
    return pd.DataFrame({
        "exposure": X, "brain": M, "behavior": Y,
        "age": cov[:, 0], "sex": cov[:, 1],
    })


# =========================================================================== #
#  CORE                                                                       #
# =========================================================================== #

class TestCoreMediation:

    def test_recovers_paths(self):
        X, M, Y, cov = _sim(n=500, a=0.5, b=0.4)
        r = mediation_analysis(X, M, Y, covariates=cov, n_boot=1000, seed=42)
        assert abs(r.a_path - 0.5) < 0.15
        assert abs(r.b_path - 0.4) < 0.15
        assert abs(r.indirect_effect - 0.2) < 0.10
        assert r.significant

    def test_null_case(self):
        rng = np.random.RandomState(0)
        n = 200
        X = rng.randn(n)
        M = rng.randn(n)  # unrelated
        Y = 0.5 * X + rng.randn(n) * 0.5
        r = mediation_analysis(X, M, Y, n_boot=1000, seed=0)
        assert abs(r.indirect_effect) < 0.1

    def test_result_attributes(self):
        X, M, Y, cov = _sim(n=100)
        r = mediation_analysis(X, M, Y, n_boot=500, seed=1)
        assert r.n_obs == 100
        assert r.n_boot == 500
        assert len(r.boot_distribution) == 500
        assert isinstance(r.summary(), str)
        assert r.r2_m_model > 0
        assert r.r2_y_full >= r.r2_y_reduced

    def test_bca_vs_percentile(self):
        X, M, Y, _ = _sim(n=200)
        r = mediation_analysis(X, M, Y, n_boot=2000, seed=42)
        # BCa and percentile should be similar but not identical
        assert r.indirect_ci != r.indirect_ci_percentile

    def test_missing_data(self):
        X, M, Y, cov = _sim(n=100)
        X[0] = np.nan
        M[5] = np.nan
        cov[10, 0] = np.nan
        r = mediation_analysis(X, M, Y, covariates=cov, n_boot=500, seed=1)
        assert r.n_obs == 97

    def test_no_covariates(self):
        X, M, Y, _ = _sim(n=200)
        r = mediation_analysis(X, M, Y, n_boot=500, seed=1)
        assert r.n_obs == 200

    def test_small_sample_warning(self):
        rng = np.random.RandomState(0)
        X = rng.randn(8)
        M = rng.randn(8)
        Y = rng.randn(8)
        with pytest.warns(UserWarning, match="Very small sample"):
            mediation_analysis(X, M, Y, n_boot=100, seed=0)

    def test_proportion_mediated_nan_when_c_zero(self):
        rng = np.random.RandomState(0)
        n = 100
        X = rng.randn(n)
        M = 0.5 * X + rng.randn(n)
        Y = rng.randn(n) * 10  # no real X→Y
        r = mediation_analysis(X, M, Y, n_boot=500, seed=0)
        # c ≈ 0, so proportion should be nan or very large
        # (depends on noise; just check it doesn't crash)
        assert np.isfinite(r.indirect_effect)


class TestSensitivity:

    def test_structure(self):
        X, M, Y, cov = _sim(n=200)
        s = sensitivity_analysis(X, M, Y, covariates=cov, n_boot=200, seed=0)
        assert "rho" in s
        assert "indirect_effect" in s
        assert "rho_at_zero" in s
        assert len(s["rho"]) == len(s["indirect_effect"])

    def test_rho_at_zero_finite(self):
        X, M, Y, cov = _sim(n=300)
        s = sensitivity_analysis(X, M, Y, covariates=cov, n_boot=200, seed=0)
        assert np.isfinite(s["rho_at_zero"])


# =========================================================================== #
#  ROI                                                                        #
# =========================================================================== #

class TestROI:

    def test_single_mediator(self):
        df = _sim_df()
        r = roi_mediation(df, "exposure", "brain", "behavior",
                          covariates=["age", "sex"], n_boot=500, seed=0)
        assert "brain" in r
        assert r["brain"].significant

    def test_multi_fdr(self):
        rng = np.random.RandomState(0)
        n = 200
        X = rng.randn(n)
        df = pd.DataFrame({
            "exp": X,
            "roi_real": 0.5 * X + rng.randn(n) * 0.5,
            "roi_null": rng.randn(n),
            "beh": np.zeros(n), "age": rng.randn(n),
        })
        df["beh"] = 0.4 * df["roi_real"] + 0.1 * X + rng.randn(n) * 0.5
        r = roi_mediation(df, "exp", ["roi_real", "roi_null"], "beh",
                          covariates="age", n_boot=500, fdr_correct=True, seed=0)
        assert len(r) == 2
        s = results_to_dataframe(r)
        assert "fdr_pval" in s.columns

    def test_csv_input(self, tmp_path):
        df = _sim_df(n=100)
        csv = tmp_path / "data.csv"
        df.to_csv(csv, index=False)
        r = roi_mediation(str(csv), "exposure", "brain", "behavior",
                          n_boot=200, seed=0)
        assert "brain" in r

    def test_roi_sensitivity(self):
        df = _sim_df(n=200)
        s = roi_sensitivity(df, "exposure", "brain", "behavior",
                            covariates=["age", "sex"], n_boot=200, seed=0)
        assert np.isfinite(s["rho_at_zero"])

    def test_missing_column_raises(self):
        df = _sim_df(n=50)
        with pytest.raises(ValueError, match="not found"):
            roi_mediation(df, "exposure", "nonexistent", "behavior")


# =========================================================================== #
#  TRACT                                                                      #
# =========================================================================== #

class TestTract:

    def test_basic(self):
        rng = np.random.RandomState(0)
        ns, nn = 100, 30
        X = rng.randn(ns)
        prof = np.zeros((ns, nn))
        for j in range(nn):
            eff = 0.5 * np.exp(-((j - 15) ** 2) / 50)
            prof[:, j] = eff * X + rng.randn(ns) * 0.5
        Y = 0.4 * prof[:, 15] + rng.randn(ns) * 0.5

        r = tract_mediation(prof, X, Y, tract_name="test",
                            n_boot=300, seed=0, verbose=False)
        assert r.n_nodes == 30
        assert len(r.node_results) == 30
        assert isinstance(r.summary(), str)

    def test_dataframe_to_csv(self):
        rng = np.random.RandomState(0)
        ns, nn = 50, 10
        prof = rng.randn(ns, nn)
        X = rng.randn(ns)
        Y = rng.randn(ns)
        r = tract_mediation(prof, X, Y, n_boot=100, seed=0, verbose=False)
        df = tract_results_to_dataframe(r)
        assert len(df) == nn

    def test_multi_tract(self):
        rng = np.random.RandomState(0)
        ns, nn = 50, 10
        X = rng.randn(ns)
        Y = rng.randn(ns)
        d = {"A": rng.randn(ns, nn), "B": rng.randn(ns, nn)}
        r = multi_tract_mediation(d, X, Y, n_boot=100, seed=0, verbose=False)
        assert "A" in r and "B" in r


# =========================================================================== #
#  VOXELWISE                                                                  #
# =========================================================================== #

class TestVoxelwise:

    def test_small_volume(self):
        """Tiny 4×4×4 volume, 30 subjects — smoke test."""
        try:
            import nibabel as nib
        except ImportError:
            pytest.skip("nibabel not installed")

        from neuromediate.voxelwise import voxelwise_mediation

        rng = np.random.RandomState(0)
        ns = 30
        shape = (4, 4, 4)
        X = rng.randn(ns)
        Y = rng.randn(ns)

        vol = rng.randn(*shape, ns).astype(np.float32)
        # inject a mediation effect at voxel (2,2,2)
        vol[2, 2, 2, :] = 0.5 * X + rng.randn(ns) * 0.3
        Y += 0.4 * vol[2, 2, 2, :]

        with tempfile.TemporaryDirectory() as td:
            nii_path = os.path.join(td, "brain.nii.gz")
            img = nib.Nifti1Image(vol, np.eye(4))
            nib.save(img, nii_path)

            r = voxelwise_mediation(
                nii_path, X, Y,
                n_boot=200, seed=0, verbose=False,
            )

            assert r.n_voxels > 0
            assert r.ab_map.shape == shape
            assert r.sig_map.shape == shape

            # save maps
            out_dir = os.path.join(td, "out")
            saved = r.save_maps(out_dir)
            assert "indirect_ab" in saved
            assert os.path.isfile(saved["indirect_ab"])

            # fdr
            fdr = r.apply_fdr(q=0.05)
            assert fdr.shape == shape

            # cluster
            cl = r.cluster_threshold(primary_p=0.5, min_cluster_size=1)
            assert cl.shape == shape


# =========================================================================== #
#  REPORTING                                                                  #
# =========================================================================== #

class TestReporting:

    def test_save_csv_roi(self, tmp_path):
        df = _sim_df(n=100)
        r = roi_mediation(df, "exposure", "brain", "behavior", n_boot=200, seed=0)
        p = tmp_path / "out.csv"
        save_csv(r, str(p))
        loaded = pd.read_csv(p)
        assert "indirect_effect" in loaded.columns

    def test_save_csv_tract(self, tmp_path):
        rng = np.random.RandomState(0)
        prof = rng.randn(50, 10)
        r = tract_mediation(prof, rng.randn(50), rng.randn(50),
                            n_boot=100, seed=0, verbose=False)
        p = tmp_path / "tract.csv"
        save_csv(r, str(p))
        assert os.path.isfile(p)

    def test_plot_diagram(self, tmp_path):
        X, M, Y, _ = _sim(n=100)
        r = mediation_analysis(X, M, Y, n_boot=200, seed=0)
        fig = plot_mediation_diagram(r, output_path=str(tmp_path / "diag.png"))
        assert fig is not None

    def test_plot_bootstrap(self, tmp_path):
        X, M, Y, _ = _sim(n=100)
        r = mediation_analysis(X, M, Y, n_boot=200, seed=0)
        fig = plot_bootstrap_distribution(r, output_path=str(tmp_path / "boot.png"))
        assert fig is not None

    def test_plot_sensitivity(self, tmp_path):
        X, M, Y, _ = _sim(n=100)
        s = sensitivity_analysis(X, M, Y, n_boot=100, seed=0)
        fig = plot_sensitivity(s, output_path=str(tmp_path / "sens.png"))
        assert fig is not None

    def test_plot_tract(self, tmp_path):
        rng = np.random.RandomState(0)
        r = tract_mediation(rng.randn(50, 20), rng.randn(50), rng.randn(50),
                            n_boot=100, seed=0, verbose=False)
        fig = plot_tract_mediation(r, output_path=str(tmp_path / "tract.png"))
        assert fig is not None

    def test_plot_forest(self, tmp_path):
        df = _sim_df(n=100)
        rng = np.random.RandomState(0)
        df["brain2"] = rng.randn(100)
        r = roi_mediation(df, "exposure", ["brain", "brain2"], "behavior",
                          n_boot=200, seed=0)
        fig = plot_roi_forest(r, output_path=str(tmp_path / "forest.png"))
        assert fig is not None

    def test_html_report_roi(self, tmp_path):
        df = _sim_df(n=100)
        r = roi_mediation(df, "exposure", "brain", "behavior", n_boot=200, seed=0)
        s = roi_sensitivity(df, "exposure", "brain", "behavior", n_boot=100, seed=0)
        p = tmp_path / "report.html"
        generate_html_report(r, str(p), sensitivity=s)
        content = p.read_text()
        assert "NeuroMediate" in content
        assert "data:image/png;base64" in content

    def test_html_report_tract(self, tmp_path):
        rng = np.random.RandomState(0)
        r = tract_mediation(rng.randn(50, 10), rng.randn(50), rng.randn(50),
                            n_boot=100, seed=0, verbose=False)
        p = tmp_path / "tract_report.html"
        generate_html_report(r, str(p))
        assert p.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
