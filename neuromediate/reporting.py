"""
Reporting and visualisation module.

* ``save_csv``                   — summary table to disk
* ``plot_mediation_diagram``     — classic path diagram
* ``plot_bootstrap_distribution``— histogram of bootstrap indirect effects
* ``plot_sensitivity``           — sensitivity-to-confounding curve
* ``plot_tract_mediation``       — along-tract indirect-effect profile
* ``plot_roi_forest``            — forest plot comparing multiple ROIs
* ``generate_html_report``       — self-contained HTML with embedded PNGs
"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .core import MediationResult
from .roi import results_to_dataframe
from .tract import TractMediationResult, tract_results_to_dataframe


# =========================================================================== #
#  Helpers                                                                    #
# =========================================================================== #

def _ensure_agg():
    import matplotlib
    matplotlib.use("Agg")


def _fig_to_b64(fig) -> str:
    import matplotlib.pyplot as plt
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


def _sig_label(coef: float, pval: float) -> str:
    stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
    return f"{coef:.3f}{stars}"


# =========================================================================== #
#  CSV export                                                                 #
# =========================================================================== #

def save_csv(
    results: Union[Dict[str, MediationResult], TractMediationResult],
    output_path: str,
) -> str:
    """Write summary table to CSV.

    Parameters
    ----------
    results : output from ``roi_mediation()`` or ``tract_mediation()``.
    output_path : destination path.

    Returns
    -------
    str — written path.
    """
    if isinstance(results, TractMediationResult):
        df = tract_results_to_dataframe(results)
    elif isinstance(results, dict):
        df = results_to_dataframe(results)
    else:
        raise TypeError(f"Unsupported type: {type(results)}")
    df.to_csv(output_path, index=False)
    return str(output_path)


# =========================================================================== #
#  Path diagram                                                               #
# =========================================================================== #

def plot_mediation_diagram(
    result: MediationResult,
    labels: Optional[Dict[str, str]] = None,
    output_path: Optional[str] = None,
    figsize: tuple = (8, 5),
):
    """Classic three-variable mediation path diagram.

    Parameters
    ----------
    result : a single MediationResult.
    labels : ``{'x': ..., 'm': ..., 'y': ...}`` display names.
    output_path : save to file (PNG, PDF, SVG …).  None → return fig only.
    figsize : matplotlib figure size.
    """
    _ensure_agg()
    import matplotlib.pyplot as plt

    if labels is None:
        labels = {"x": "X (Exposure)", "m": "M (Brain)", "y": "Y (Behavior)"}

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    box_x = dict(boxstyle="round,pad=0.5", fc="#d5e8f0", ec="#2c3e50", lw=1.5)
    box_m = dict(boxstyle="round,pad=0.5", fc="#fdf2d5", ec="#2c3e50", lw=1.5)
    box_y = dict(boxstyle="round,pad=0.5", fc="#d5f0df", ec="#2c3e50", lw=1.5)

    ax.text(1.5, 2, labels["x"], ha="center", va="center",
            fontsize=12, fontweight="bold", bbox=box_x)
    ax.text(5.0, 5.5, labels["m"], ha="center", va="center",
            fontsize=12, fontweight="bold", bbox=box_m)
    ax.text(8.5, 2, labels["y"], ha="center", va="center",
            fontsize=12, fontweight="bold", bbox=box_y)

    arr = dict(arrowstyle="->", lw=2, color="#2c3e50")

    # a
    ax.annotate("", xy=(4.0, 5.2), xytext=(2.2, 2.8), arrowprops=arr)
    ax.text(2.8, 4.3, f"a = {_sig_label(result.a_path, result.a_pval)}",
            fontsize=10, ha="center", color="#2980b9")
    # b
    ax.annotate("", xy=(7.8, 2.8), xytext=(6.0, 5.2), arrowprops=arr)
    ax.text(7.2, 4.3, f"b = {_sig_label(result.b_path, result.b_pval)}",
            fontsize=10, ha="center", color="#2980b9")
    # c'
    ax.annotate("", xy=(7.4, 2.0), xytext=(2.6, 2.0),
                arrowprops=dict(arrowstyle="->", lw=2, color="#95a5a6",
                                linestyle="--"))
    ax.text(5.0, 1.2,
            f"c' = {_sig_label(result.c_prime, result.c_prime_pval)}",
            fontsize=10, ha="center", color="#7f8c8d")

    # indirect
    sig_str = "SIG" if result.significant else "n.s."
    ci = result.indirect_ci
    ax.text(5.0, 0.25,
            f"Indirect (a×b) = {result.indirect_effect:.4f}   "
            f"{result.ci_level*100:.0f}% CI [{ci[0]:.4f}, {ci[1]:.4f}]  {sig_str}",
            ha="center", fontsize=9, style="italic",
            color="#c0392b" if result.significant else "#7f8c8d")

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    return fig


# =========================================================================== #
#  Bootstrap distribution                                                     #
# =========================================================================== #

def plot_bootstrap_distribution(
    result: MediationResult,
    output_path: Optional[str] = None,
    figsize: tuple = (7, 4),
):
    """Histogram of the bootstrap indirect-effect distribution."""
    _ensure_agg()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(result.boot_distribution, bins=80, density=True,
            color="#3498db", alpha=0.7, edgecolor="white", lw=0.4)
    ax.axvline(result.indirect_effect, color="#e74c3c", lw=2,
               label=f"Observed: {result.indirect_effect:.4f}")
    ax.axvline(result.indirect_ci[0], color="#e67e22", lw=1.5, ls="--",
               label=f"CI lo: {result.indirect_ci[0]:.4f}")
    ax.axvline(result.indirect_ci[1], color="#e67e22", lw=1.5, ls="--",
               label=f"CI hi: {result.indirect_ci[1]:.4f}")
    ax.axvline(0, color="black", lw=0.5, ls=":")
    ax.set_xlabel("Indirect effect (a × b)")
    ax.set_ylabel("Density")
    ax.set_title("Bootstrap Distribution of Indirect Effect")
    ax.legend(fontsize=9)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    return fig


# =========================================================================== #
#  Sensitivity plot                                                           #
# =========================================================================== #

def plot_sensitivity(
    sens: dict,
    output_path: Optional[str] = None,
    figsize: tuple = (8, 5),
):
    """Sensitivity-to-confounding plot."""
    _ensure_agg()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    rho = sens["rho"]
    ab = sens["indirect_effect"]
    ax.fill_between(rho, sens["ci_lower"], sens["ci_upper"],
                     alpha=0.25, color="#3498db")
    ax.plot(rho, ab, color="#3498db", lw=2)
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.axvline(0, color="#95a5a6", lw=0.5, ls=":")
    rz = sens.get("rho_at_zero", np.nan)
    if np.isfinite(rz):
        ax.axvline(rz, color="#e74c3c", lw=1.5, ls="--",
                   label=f"ρ at zero = {rz:.2f}")
        ax.legend(fontsize=10)
    ax.set_xlabel("Residual correlation (ρ)")
    ax.set_ylabel("Indirect effect (a × b)")
    ax.set_title("Sensitivity to Unmeasured M–Y Confounding")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    return fig


# =========================================================================== #
#  Along-tract profile plot                                                   #
# =========================================================================== #

def plot_tract_mediation(
    result: TractMediationResult,
    metric_label: str = "Indirect effect (a × b)",
    output_path: Optional[str] = None,
    figsize: tuple = (10, 4),
):
    """Along-tract mediation profile with CI band and FDR highlights."""
    _ensure_agg()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(figsize[0], figsize[1] * 2),
                              sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    nodes = np.arange(result.n_nodes)

    # --- top panel: indirect effect ---
    ax = axes[0]
    ax.fill_between(nodes, result.ci_lower, result.ci_upper,
                     alpha=0.25, color="#3498db", label="95% CI")
    ax.plot(nodes, result.indirect_effects, color="#3498db", lw=2,
            label="Indirect effect")
    if np.any(result.fdr_significant):
        sig_n = nodes[result.fdr_significant]
        sig_v = result.indirect_effects[result.fdr_significant]
        ax.scatter(sig_n, sig_v, color="#e74c3c", s=25, zorder=5,
                   label="FDR significant")
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_ylabel(metric_label)
    ax.set_title(f"Along-Tract Mediation: {result.tract_name}")
    ax.legend(fontsize=8)

    # --- bottom panel: a and b paths ---
    ax2 = axes[1]
    ax2.plot(nodes, result.a_paths, color="#e67e22", lw=1.5, label="a path (X→M)")
    ax2.plot(nodes, result.b_paths, color="#27ae60", lw=1.5, label="b path (M→Y|X)")
    ax2.axhline(0, color="black", lw=0.5, ls="--")
    ax2.set_xlabel("Tract node")
    ax2.set_ylabel("Path coefficient")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    return fig


# =========================================================================== #
#  Forest plot for multi-ROI comparison                                       #
# =========================================================================== #

def plot_roi_forest(
    results: Dict[str, MediationResult],
    output_path: Optional[str] = None,
    figsize: Optional[tuple] = None,
):
    """Forest plot of indirect effects across multiple ROI mediators."""
    _ensure_agg()
    import matplotlib.pyplot as plt

    names = list(results.keys())
    n = len(names)
    if figsize is None:
        figsize = (7, max(3, 0.6 * n + 1.5))

    ab = [results[k].indirect_effect for k in names]
    lo = [results[k].indirect_ci[0] for k in names]
    hi = [results[k].indirect_ci[1] for k in names]
    sig = [results[k].significant for k in names]
    y_pos = np.arange(n)

    fig, ax = plt.subplots(figsize=figsize)
    for i in range(n):
        colour = "#e74c3c" if sig[i] else "#95a5a6"
        ax.plot([lo[i], hi[i]], [y_pos[i], y_pos[i]], color=colour, lw=2)
        ax.scatter(ab[i], y_pos[i], color=colour, s=60, zorder=5)

    ax.axvline(0, color="black", lw=0.5, ls="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Indirect effect (a × b)")
    ax.set_title("Forest Plot — ROI Mediation")
    ax.invert_yaxis()
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    return fig


# =========================================================================== #
#  HTML report                                                                #
# =========================================================================== #

_CSS = """
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    max-width: 920px; margin: 40px auto; padding: 0 20px;
    color: #333; line-height: 1.6;
}
h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
h2 { color: #2980b9; margin-top: 40px; }
h3 { color: #7f8c8d; }
.tbl {
    border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 13px;
}
.tbl th {
    background: #3498db; color: #fff; padding: 10px 8px; text-align: left;
}
.tbl td { padding: 8px; border-bottom: 1px solid #ddd; }
.tbl tr:nth-child(even) { background: #f8f9fa; }
.tbl tr:hover { background: #e8f4fd; }
pre {
    background: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 6px;
    padding: 15px; overflow-x: auto; font-size: 13px;
}
img { border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,.1); margin: 10px 0; }
.foot {
    margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd;
    color: #95a5a6; font-size: 12px;
}
"""


def generate_html_report(
    results: Union[Dict[str, MediationResult], TractMediationResult],
    output_path: str,
    title: str = "NeuroMediate Analysis Report",
    labels: Optional[Dict[str, str]] = None,
    sensitivity: Optional[dict] = None,
) -> str:
    """Self-contained HTML report with embedded PNG figures.

    Parameters
    ----------
    results : ``roi_mediation()`` dict or ``TractMediationResult``.
    output_path : destination HTML path.
    title : report title.
    labels : variable display labels.
    sensitivity : output from ``sensitivity_analysis()`` / ``roi_sensitivity()``.

    Returns
    -------
    str — path written.
    """
    _ensure_agg()
    import matplotlib.pyplot as plt

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sections: List[str] = []

    # ---- ROI results --------------------------------------------------------
    if isinstance(results, dict):
        df = results_to_dataframe(results)
        sections.append(f"<h2>Summary Table</h2>\n"
                        f"{df.to_html(index=False, float_format='%.4f', classes='tbl')}")

        # forest plot
        fig_for = plot_roi_forest(results)
        sections.append(f'<h2>Forest Plot</h2>\n'
                        f'<img src="data:image/png;base64,{_fig_to_b64(fig_for)}"'
                        f' style="max-width:100%">')

        for name, res in results.items():
            fig_d = plot_mediation_diagram(res, labels=labels)
            fig_b = plot_bootstrap_distribution(res)
            sections.append(
                f'<h2>Path Diagram — {name}</h2>\n'
                f'<img src="data:image/png;base64,{_fig_to_b64(fig_d)}"'
                f' style="max-width:100%">\n'
                f'<pre>{res.summary()}</pre>\n'
                f'<h3>Bootstrap Distribution — {name}</h3>\n'
                f'<img src="data:image/png;base64,{_fig_to_b64(fig_b)}"'
                f' style="max-width:100%">'
            )

    # ---- Tract results ------------------------------------------------------
    elif isinstance(results, TractMediationResult):
        df = tract_results_to_dataframe(results)
        sections.append(f"<h2>Node-Level Summary</h2>\n"
                        f"{df.to_html(index=False, float_format='%.4f', classes='tbl')}")
        fig_t = plot_tract_mediation(results)
        sections.append(
            f'<h2>Along-Tract Profile</h2>\n'
            f'<img src="data:image/png;base64,{_fig_to_b64(fig_t)}"'
            f' style="max-width:100%">\n'
            f'<pre>{results.summary()}</pre>'
        )

    # ---- Sensitivity --------------------------------------------------------
    if sensitivity is not None:
        fig_s = plot_sensitivity(sensitivity)
        rz = sensitivity.get("rho_at_zero", np.nan)
        rz_s = f"{rz:.3f}" if np.isfinite(rz) else "N/A"
        sections.append(
            f'<h2>Sensitivity Analysis</h2>\n'
            f'<p>Indirect effect crosses zero at ρ = {rz_s}.  '
            f'Larger |ρ| at zero → more robust mediation.</p>\n'
            f'<img src="data:image/png;base64,{_fig_to_b64(fig_s)}"'
            f' style="max-width:100%">'
        )

    html = (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title><style>{_CSS}</style></head><body>"
        f"<h1>{title}</h1>\n"
        + "\n".join(sections)
        + "\n<div class='foot'>Generated by NeuroMediate</div>"
          "</body></html>"
    )

    out.write_text(html)
    return str(out)
