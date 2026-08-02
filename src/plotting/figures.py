from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Optional

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#111111",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "text.color": "#111111",
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.6,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlepad": 10,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#CCCCCC",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

REGIME_COLORS = {"A": "#1565C0", "B": "#E65100", "C": "#2E7D32", "D": "#C62828"}
METHOD_COLORS = {"Entropy": "#1976D2", "Self-consist.": "#F57C00", "SUA (ours)": "#D32F2F"}
COVERAGES = np.linspace(0.1, 1.0, 40)

FIGURES_DIR = Path("figures")


def _save(fig: plt.Figure, name: str, subdir: str = "png") -> None:
    out = FIGURES_DIR / subdir
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=200)
    fig.savefig((FIGURES_DIR / "pdf" / f"{name}.pdf"), dpi=200)
    (FIGURES_DIR / "pdf").mkdir(parents=True, exist_ok=True)
    plt.close(fig)


def plot_sh_scatter(
    S: np.ndarray,
    H: np.ndarray,
    errors: np.ndarray,
    regime_d_mask: Optional[np.ndarray] = None,
    title: str = "S × H Scatter",
    lambda_val: float = 1.0,
    save_name: Optional[str] = None,
) -> plt.Figure:
    """Fig 1 / Fig 15: Sensitivity vs Entropy scatter coloured by error."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(H[errors == 0], S[errors == 0], c="#90CAF9", alpha=0.30,
               s=12, label="Correct", rasterized=True)
    ax.scatter(H[errors == 1], S[errors == 1], c="#EF9A9A", alpha=0.60,
               s=14, label="Error", rasterized=True)
    if regime_d_mask is not None and regime_d_mask.sum() > 0:
        ax.scatter(H[regime_d_mask], S[regime_d_mask], s=60,
                   facecolors="none", edgecolors="#C62828", lw=1.5,
                   label="Regime D", rasterized=True)
    h_line = np.linspace(0, H.max() * 1.05, 200)
    ax.plot(h_line, lambda_val * h_line, "--", color="#F9A825",
            lw=2, label=f"SUA=0 (λ={lambda_val})")
    ax.set_xlabel("Entropy H(Y|x)  →  expressed uncertainty")
    ax.set_ylabel("Sensitivity S(x;ε)  →  output instability")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.35)
    if save_name:
        _save(fig, save_name)
    return fig


def plot_regime_auroc_bars(
    regime_aurocs: dict[str, list[float]],
    regime_labels: list[str] = ["A", "B", "C", "D"],
    title: str = "Failure-Prediction AUROC by Regime",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """Fig 2: Per-regime AUROC bar chart."""
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(regime_labels))
    w = 0.22
    colors = list(METHOD_COLORS.values())
    for j, (method, vals) in enumerate(regime_aurocs.items()):
        bars = ax.bar(x + (j - 1) * w, vals, w, label=method,
                      color=colors[j % len(colors)], alpha=0.85, zorder=3)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0.5, color="#777", ls="--", lw=1.2, alpha=0.8, label="Random (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Regime {r}" for r in regime_labels])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("AUROC (failure prediction)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.35)
    # Highlight Regime D
    ax.axvspan(len(regime_labels) - 1.5, len(regime_labels) - 0.5,
               color="#FFCDD2", alpha=0.4, zorder=1)
    if save_name:
        _save(fig, save_name)
    return fig


def plot_selective_accuracy(
    method_scores: dict[str, np.ndarray],
    errors: np.ndarray,
    coverage: float = 0.80,
    title: str = "Selective Accuracy vs Coverage",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """Fig 5 / Fig 14: Selective accuracy curves."""
    from src.evaluation.selective_prediction import selective_accuracy
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = list(METHOD_COLORS.values())
    for j, (method, scores) in enumerate(method_scores.items()):
        sas = [selective_accuracy(scores, errors, c) for c in COVERAGES]
        ax.plot(COVERAGES, sas, "-", color=colors[j % len(colors)],
                lw=2.5, label=method)
    ax.axvline(coverage, color="#F9A825", ls="--", lw=1.5,
               label=f"Coverage={coverage}")
    ax.set_xlabel("Coverage (fraction of samples kept)")
    ax.set_ylabel("Selective Accuracy")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.35)
    if save_name:
        _save(fig, save_name)
    return fig


def plot_sua_distribution(
    sua_scores: np.ndarray,
    regime_labels: np.ndarray,
    title: str = "SUA Score Distribution by Regime",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """Fig 4: SUA histogram per regime."""
    fig, ax = plt.subplots(figsize=(8, 5))
    regime_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    for r in range(4):
        mask = regime_labels == r
        if mask.sum() == 0:
            continue
        ax.hist(sua_scores[mask], bins=40, alpha=0.60,
                color=REGIME_COLORS[regime_map[r]],
                label=f"Regime {regime_map[r]}", density=True, zorder=3 + r)
    ax.axvline(0, color="#F9A825", ls="--", lw=2.0, label="SUA=0", zorder=10)
    ax.set_xlabel("SUA score = S − λH")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.35)
    if save_name:
        _save(fig, save_name)
    return fig


def plot_regime_d_vs_delta(
    regime_d_pcts: list[float],
    delta_aurocs: list[float],
    labels: Optional[list[str]] = None,
    title: str = "Regime D Population vs SUA Advantage",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """Fig 1 (main paper): Regime D % vs ΔAUROC scatter."""
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axhline(0, color="#999", ls="--", lw=1.2, alpha=0.8)
    ax.axvline(20, color="#999", ls=":", lw=1, alpha=0.6)
    ax.axvspan(20, max(regime_d_pcts) * 1.1, alpha=0.08, color="#1565C0")
    ax.scatter(regime_d_pcts, delta_aurocs, s=80, zorder=5, color="#C62828",
               edgecolors="black", lw=0.5)
    if labels:
        for x, y, l in zip(regime_d_pcts, delta_aurocs, labels):
            ax.annotate(l, (x, y), textcoords="offset points",
                        xytext=(5, 3), fontsize=8)
    # Linear trend
    if len(regime_d_pcts) >= 2:
        z = np.polyfit(regime_d_pcts, delta_aurocs, 1)
        xfit = np.linspace(0, max(regime_d_pcts) * 1.1, 100)
        ax.plot(xfit, np.polyval(z, xfit), "--", color="#C62828",
                alpha=0.6, lw=1.5, label="Linear trend")
    ax.set_xlabel("Regime D population (%)")
    ax.set_ylabel("ΔAUROC (SUA − Entropy)")
    ax.set_title(title)
    ax.text(5, ax.get_ylim()[1] * 0.9, "Entropy sufficient", color="#555", fontsize=9)
    ax.text(25, ax.get_ylim()[1] * 0.9, "SUA helps", color="#1565C0", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.35)
    if save_name:
        _save(fig, save_name)
    return fig


def plot_k_sweep(
    k_values: list[int],
    aurocs: list[float],
    spearman_rhos: Optional[list[float]] = None,
    title: str = "K Sweep: AUROC and Spearman ρ",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """Fig 17: K perturbations sweep."""
    fig, ax = plt.subplots(figsize=(8, 5))
    xp = np.arange(len(k_values))
    b1 = ax.bar(xp - 0.2, aurocs, 0.35, label="SUA AUROC",
                color="#D32F2F", alpha=0.85)
    for bar, v in zip(b1, aurocs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    if spearman_rhos:
        b2 = ax.bar(xp + 0.2, spearman_rhos, 0.35,
                    label="Spearman ρ(S_K, S_8)", color="#1565C0", alpha=0.85)
        for bar, v in zip(b2, spearman_rhos):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xp)
    ax.set_xticklabels([f"K={k}" for k in k_values])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.35)
    if save_name:
        _save(fig, save_name)
    return fig


def plot_calibration_curve(
    bin_centers: np.ndarray,
    accuracies: np.ndarray,
    confidences: np.ndarray,
    title: str = "Calibration / Reliability Diagram",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """Reliability diagram."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="#999", lw=1.5, label="Perfect calibration")
    ax.plot(confidences, accuracies, "o-", color="#D32F2F", lw=2, ms=6, label="Model")
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Fraction correct")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.35)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    if save_name:
        _save(fig, save_name)
    return fig


def plot_combined_results(
    S: np.ndarray, H: np.ndarray, errors: np.ndarray,
    regime_labels: np.ndarray, sua_scores: np.ndarray,
    regime_aurocs: dict, method_scores: dict,
    save_name: str = "fig_combined_results",
) -> plt.Figure:
    """6-panel combined results figure."""
    from src.evaluation.selective_prediction import selective_accuracy

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Panel 1: S×H scatter
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(H[errors == 0], S[errors == 0], c="#90CAF9", alpha=0.25, s=8,
                label="Correct", rasterized=True)
    ax1.scatter(H[errors == 1], S[errors == 1], c="#EF9A9A", alpha=0.55, s=10,
                label="Error", rasterized=True)
    hl = np.linspace(0, H.max(), 200)
    ax1.plot(hl, hl, "--", color="#F9A825", lw=1.8, label="SUA=0")
    ax1.set_xlabel("H(Y|x)"); ax1.set_ylabel("S(x;ε)")
    ax1.set_title("Fig 1 — S×H: Errors in Regime D")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # Panel 2: regime AUROC bars
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(4); w = 0.22
    colors = list(METHOD_COLORS.values())
    for j, (mn, vals) in enumerate(regime_aurocs.items()):
        ax2.bar(x + (j - 1) * w, vals, w, label=mn, color=colors[j], alpha=0.85)
    ax2.axhline(0.5, color="#777", ls="--", lw=1)
    ax2.set_xticks(x); ax2.set_xticklabels(["A", "B", "C", "D"])
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("AUROC")
    ax2.set_title("Fig 2 — AUROC by Regime"); ax2.legend(fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)

    # Panel 3: selective accuracy
    ax3 = fig.add_subplot(gs[0, 2])
    for j, (mn, sc) in enumerate(method_scores.items()):
        sas = [selective_accuracy(sc, errors, c) for c in COVERAGES]
        ax3.plot(COVERAGES, sas, "-", color=colors[j], lw=2, label=mn)
    ax3.axvline(0.8, color="#F9A825", ls="--", lw=1.2)
    ax3.set_xlabel("Coverage"); ax3.set_ylabel("Sel. Accuracy")
    ax3.set_title("Fig 5 — SelAcc vs Coverage")
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

    # Panel 4: SUA distribution
    ax4 = fig.add_subplot(gs[1, 0])
    rmap = {0: "A", 1: "B", 2: "C", 3: "D"}
    for r in range(4):
        mask = regime_labels == r
        if mask.sum():
            ax4.hist(sua_scores[mask], bins=35, alpha=0.6,
                     color=REGIME_COLORS[rmap[r]], label=f"Regime {rmap[r]}",
                     density=True, zorder=3 + r)
    ax4.axvline(0, color="#F9A825", ls="--", lw=2, label="SUA=0", zorder=10)
    ax4.set_xlabel("SUA = S − λH"); ax4.set_ylabel("Density")
    ax4.set_title("Fig 4 — SUA Distribution"); ax4.legend(fontsize=8)
    ax4.grid(True, axis="y", alpha=0.3)

    # Panel 5: Regime map
    ax5 = fig.add_subplot(gs[1, 1])
    for r in range(4):
        mask = regime_labels == r
        mk = "D" if r == 3 else "o"
        ax5.scatter(H[mask], S[mask], c=REGIME_COLORS[rmap[r]],
                    alpha=0.28, s=14 if r == 3 else 7,
                    label=f"Regime {rmap[r]}", marker=mk, rasterized=True)
    ax5.plot(hl, hl, "--", color="#F9A825", lw=2, label="SUA=0", zorder=10)
    ax5.set_xlabel("H(Y|x)"); ax5.set_ylabel("S(x;ε)")
    ax5.set_title("Fig 6 — Regime Map (S×H)"); ax5.legend(fontsize=7)
    ax5.grid(True, alpha=0.3)

    # Panel 6: placeholder for additional figure
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.text(0.5, 0.5, "See\nFigure 1 (paper)\nRegime D vs ΔAUROC",
             ha="center", va="center", fontsize=12, color="#555",
             transform=ax6.transAxes)
    ax6.set_title("Fig — Regime D vs ΔAUROC")
    ax6.grid(True, alpha=0.3)

    fig.suptitle(
        "SUA Empirical Validation — NeurIPS 2026\n"
        "Hiremath & Hiremath · AoE Research Group",
        fontsize=14, fontweight="bold", y=0.99,
    )
    if save_name:
        _save(fig, save_name)
    return fig
