#!/usr/bin/env python3
"""
Example: Lead Exposure → Brain → Psychopathy
=============================================

Full demonstration of neuromediate at all three analysis levels
using simulated data that mimics a childhood lead-exposure cohort.

Generative model
----------------
  Lead → ↓ PFC volume  (a = −0.35)
  Lead → ↓ Hippocampus  (a = −0.20)
  Lead → ↓ Amygdala     (a = −0.05, essentially null)
  ↓ PFC → ↑ Psychopathy (b = −0.40)
  Lead → ↑ Psychopathy  (c' = 0.15, direct)

Expected results
  PFC volume:   strong significant mediation (~45 % mediated)
  Hippocampus:  weak / marginal mediation
  Amygdala:     non-significant

Usage
-----
  python lead_exposure_example.py
"""

import os
import numpy as np
import pandas as pd

from neuromediate import (
    # ROI
    roi_mediation,
    roi_sensitivity,
    results_to_dataframe,
    # Tract
    tract_mediation,
    # Reporting
    save_csv,
    generate_html_report,
    plot_mediation_diagram,
    plot_bootstrap_distribution,
    plot_sensitivity,
    plot_tract_mediation,
    plot_roi_forest,
)


# =========================================================================== #
#  1.  Simulate data                                                          #
# =========================================================================== #

def simulate_lead_study(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Simulate a lead-exposure → brain → psychopathy dataset."""
    rng = np.random.RandomState(seed)

    age = rng.normal(30, 8, n).clip(18, 60)
    sex = rng.binomial(1, 0.5, n).astype(float)
    ses = rng.normal(50, 10, n).clip(20, 80)
    lead = rng.exponential(1.5, n) + 0.3

    pfc = (-0.35 * lead + 0.01 * age - 0.15 * sex
           + 0.02 * ses + rng.randn(n) * 0.8)
    hippo = (-0.20 * lead - 0.005 * age + 0.05 * sex
             + 0.01 * ses + rng.randn(n) * 0.9)
    amyg = -0.05 * lead + rng.randn(n) * 1.0

    psych = (-0.40 * pfc - 0.10 * hippo + 0.15 * lead
             + 0.01 * age + 0.10 * sex - 0.005 * ses
             + rng.randn(n) * 0.6)

    return pd.DataFrame({
        "blood_lead": lead,
        "pfc_volume": pfc,
        "hippocampus_volume": hippo,
        "amygdala_volume": amyg,
        "psychopathy_score": psych,
        "age": age, "sex": sex, "ses_index": ses,
    })


def simulate_tract_profiles(
    X: np.ndarray, n_nodes: int = 100, seed: int = 42,
) -> np.ndarray:
    """Simulate along-tract FA profiles with a localised exposure effect."""
    rng = np.random.RandomState(seed)
    ns = len(X)
    profiles = np.zeros((ns, n_nodes))
    for j in range(n_nodes):
        # Gaussian-shaped exposure effect, peaking at node 45
        effect = 0.35 * np.exp(-((j - 45) ** 2) / (2 * 12 ** 2))
        profiles[:, j] = 0.45 - effect * X + rng.randn(ns) * 0.04
    return profiles


# =========================================================================== #
#  2.  Run analyses                                                           #
# =========================================================================== #

def main():
    out = "example_output"
    os.makedirs(out, exist_ok=True)

    print("=" * 65)
    print("  NeuroMediate — Lead Exposure Example")
    print("=" * 65)

    # ---------------------------------------------------------------------- #
    #  Simulate                                                               #
    # ---------------------------------------------------------------------- #
    print("\n1. Simulating data (n = 300) …")
    df = simulate_lead_study(n=300, seed=42)
    df.to_csv(os.path.join(out, "simulated_subjects.csv"), index=False)
    print(f"   Columns: {list(df.columns)}")

    covariates = ["age", "sex", "ses_index"]
    mediators  = ["pfc_volume", "hippocampus_volume", "amygdala_volume"]

    # ---------------------------------------------------------------------- #
    #  ROI-level mediation                                                    #
    # ---------------------------------------------------------------------- #
    print("\n2. ROI-level mediation …")
    roi_results = roi_mediation(
        df,
        exposure="blood_lead",
        mediators=mediators,
        outcome="psychopathy_score",
        covariates=covariates,
        n_boot=5000,
        fdr_correct=True,
        seed=42,
    )

    for name, res in roi_results.items():
        print(f"\n{res.summary()}")

    # CSV
    save_csv(roi_results, os.path.join(out, "roi_results.csv"))
    print("   → roi_results.csv")

    # Forest plot
    plot_roi_forest(roi_results, output_path=os.path.join(out, "forest_plot.png"))
    print("   → forest_plot.png")

    # Path diagram for PFC
    plot_mediation_diagram(
        roi_results["pfc_volume"],
        labels={"x": "Blood Lead (µg/dL)",
                "m": "PFC Volume",
                "y": "Psychopathy Score"},
        output_path=os.path.join(out, "path_diagram_pfc.png"),
    )
    print("   → path_diagram_pfc.png")

    # Bootstrap distribution
    plot_bootstrap_distribution(
        roi_results["pfc_volume"],
        output_path=os.path.join(out, "bootstrap_pfc.png"),
    )
    print("   → bootstrap_pfc.png")

    # ---------------------------------------------------------------------- #
    #  Sensitivity analysis                                                   #
    # ---------------------------------------------------------------------- #
    print("\n3. Sensitivity analysis (PFC) …")
    sens = roi_sensitivity(
        df,
        exposure="blood_lead",
        mediator="pfc_volume",
        outcome="psychopathy_score",
        covariates=covariates,
        n_boot=1000,
        seed=42,
    )
    rz = sens["rho_at_zero"]
    print(f"   ρ at zero = {rz:.3f}")
    plot_sensitivity(sens, output_path=os.path.join(out, "sensitivity_pfc.png"))
    print("   → sensitivity_pfc.png")

    # ---------------------------------------------------------------------- #
    #  Along-tract mediation                                                  #
    # ---------------------------------------------------------------------- #
    print("\n4. Along-tract mediation (simulated uncinate fasciculus) …")
    X = df["blood_lead"].values
    Y = df["psychopathy_score"].values
    cov_arr = df[covariates].values
    profiles = simulate_tract_profiles(X, n_nodes=100, seed=42)

    # The tract profile affects Y — add contribution
    Y_tract = Y + 0.4 * profiles[:, 45]   # add mediator link at peak node

    tract_result = tract_mediation(
        profiles, X, Y_tract,
        covariates=cov_arr,
        tract_name="Uncinate Fasciculus (FA)",
        n_boot=2000,
        seed=42,
        verbose=True,
    )
    print(f"\n{tract_result.summary()}")

    save_csv(tract_result, os.path.join(out, "tract_results.csv"))
    plot_tract_mediation(
        tract_result,
        output_path=os.path.join(out, "tract_mediation.png"),
    )
    print("   → tract_results.csv")
    print("   → tract_mediation.png")

    # ---------------------------------------------------------------------- #
    #  HTML report                                                            #
    # ---------------------------------------------------------------------- #
    print("\n5. Generating HTML report …")
    generate_html_report(
        roi_results,
        output_path=os.path.join(out, "roi_report.html"),
        title="Lead Exposure → Brain Volume → Psychopathy",
        labels={"x": "Blood Lead (µg/dL)",
                "m": "Brain Volume",
                "y": "Psychopathy Score"},
        sensitivity=sens,
    )
    print("   → roi_report.html")

    generate_html_report(
        tract_result,
        output_path=os.path.join(out, "tract_report.html"),
        title="Along-Tract Mediation — Uncinate Fasciculus",
    )
    print("   → tract_report.html")

    print("\n" + "=" * 65)
    print("  Done!  All outputs in ./" + out + "/")
    print("=" * 65)


if __name__ == "__main__":
    main()
