import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

from src.datasets.synthetic import generate_synthetic
from src.metrics.sua import SUAScore, DistributionalSensitivity
from src.metrics.entropy import PredictiveEntropy
from src.metrics.self_consistency import SelfConsistency
from src.metrics.semantic_entropy import SemanticEntropy
from src.metrics.spuq import SPUQ
from src.evaluation.auroc import compute_auroc, compute_regime_auroc
from src.evaluation.calibration import compute_ece
from src.evaluation.selective_prediction import selective_accuracy, coverage_curve
from src.evaluation.regime_analysis import RegimeAnalyzer
from src.plotting.paper_figures import (
    plot_sh_scatter, plot_regime_auroc_bars, plot_selective_accuracy,
    plot_sua_distribution, plot_combined_results,
)
from src.utils.io import save_results
from src.utils.logging import get_logger

logger = get_logger("tier1-synthetic")

SEED = 2026
LAMBDA = 1.0
COVERAGE = 0.80
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")


def main():
    logger.info("=" * 65)
    logger.info("TIER 1: Synthetic Four-Regime Benchmark")
    logger.info("=" * 65)

    # ── Generate dataset ──────────────────────────────────────────
    logger.info("Generating synthetic dataset (2,400 samples, 4 regimes)...")
    data = generate_synthetic(
        n_per_regime=600,
        n_classes=3,
        n_perturbations=8,
        seed=SEED,
    )
    logger.info(f"  Total samples: {len(data.probs)}")
    logger.info(f"  Regime counts: {[(data.regime_labels == r).sum() for r in range(4)]}")
    logger.info(f"  Error rates: {[(data.errors[data.regime_labels==r].mean()*100):.1f}% for r in range(4)]}")

    # ── Compute metrics ───────────────────────────────────────────
    sua_scorer = SUAScore(lambda_val=LAMBDA)
    out = sua_scorer.compute(data.probs, data.pert_probs)
    S, H, SUA = out["S"], out["H"], out["sua"]

    sc_fn = SelfConsistency()
    SC = sc_fn.from_pert_probs(data.probs, data.pert_probs)

    se_fn = SemanticEntropy()
    SE = se_fn.from_pert_probs(data.probs, data.pert_probs)

    spuq_fn = SPUQ()
    SPQ = spuq_fn(data.probs, data.pert_probs)

    logger.info(f"  Mean S by regime:   {[round(S[data.regime_labels==r].mean(),3) for r in range(4)]}")
    logger.info(f"  Mean H by regime:   {[round(H[data.regime_labels==r].mean(),3) for r in range(4)]}")
    logger.info(f"  Mean SUA by regime: {[round(SUA[data.regime_labels==r].mean(),3) for r in range(4)]}")

    # ── Lambda sweep ──────────────────────────────────────────────
    lambda_sweep = SUAScore.sweep_lambda(
        data.probs, data.pert_probs, data.errors,
        lambdas=[0.0, 0.1, 0.5, 1.0, 2.0],
    )
    logger.info(f"  Lambda sweep: best λ*={lambda_sweep['best_lambda']} "
                f"AUROC={lambda_sweep['best_auroc']:.4f}")

    # ── Per-regime AUROC (Table 1 of paper) ───────────────────────
    methods = {
        "Entropy":       H,
        "Self-consist.": 1 - SC,
        "SUA (ours)":    SUA,
        "Sem. Entropy":  SE,
        "SPUQ":          SPQ,
    }

    analyzer = RegimeAnalyzer()
    analyzer.fit(S, H)

    print("\n" + "=" * 70)
    print("TABLE 1 — FAILURE-PREDICTION AUROC BY REGIME")
    print("(Reproduces tab:regime_diagnostic from paper)")
    print("=" * 70)
    print(f"{'Method':<22} {'A':>8} {'B':>8} {'C':>8} {'D':>8}")
    print("-" * 54)

    regime_aurocs_table = {}
    for name, scores in methods.items():
        row = compute_regime_auroc(scores, data.errors, data.regime_labels)
        regime_aurocs_table[name] = [row["A"], row["B"], row["C"], row["D"]]
        vals = "  ".join([f"{v:>6.3f}" if not np.isnan(v) else "   nan" for v in regime_aurocs_table[name]])
        print(f"{name:<22}  {vals}")

    print("-" * 54)
    print("PAPER TARGETS: Entropy D=0.228, SC D=0.500, SUA D=0.877")

    sua_D = regime_aurocs_table["SUA (ours)"][3]
    ent_D = regime_aurocs_table["Entropy"][3]
    delta_D = sua_D - ent_D
    print(f"\nΔAUROC_D (SUA − Entropy) = {delta_D:+.4f}  (paper: +0.649)")

    # ── Overall metrics (Table 2) ─────────────────────────────────
    overall_acc = 1.0 - data.errors.mean()
    ece_val = compute_ece(data.probs, data.errors)

    print("\n" + "=" * 70)
    print("TABLE 2 — OVERALL METRICS")
    print("=" * 70)
    print(f"{'Method':<22} {'Acc':>7} {'ECE':>7} {'AUROC':>8} {'SelAcc@80':>11}")
    print("-" * 58)
    for name, scores in methods.items():
        auc = compute_auroc(scores, data.errors)
        sel = selective_accuracy(scores, data.errors, COVERAGE)
        print(f"{name:<22} {overall_acc:>7.4f} {ece_val:>7.4f} {auc:>8.4f} {sel:>11.4f}")

    # ── Regime breakdown ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("REGIME STATISTICS")
    print("=" * 70)
    print(f"{'Regime':<12} {'n':>5} {'Error%':>8} {'Mean S':>8} {'Mean H':>8} {'Mean SUA':>10}")
    print("-" * 55)
    rnames = {0: "A (lo S,lo H)", 1: "B (hi S,hi H)", 2: "C (lo S,hi H)", 3: "D (hi S,lo H)"}
    for r in range(4):
        mask = data.regime_labels == r
        print(f"{rnames[r]:<12} {mask.sum():>5} {data.errors[mask].mean()*100:>7.1f}% "
              f"{S[mask].mean():>8.4f} {H[mask].mean():>8.4f} {SUA[mask].mean():>10.4f}")

    # ── Assertions: verify paper claims ──────────────────────────
    print("\n" + "=" * 70)
    print("VERIFICATION AGAINST PAPER CLAIMS")
    print("=" * 70)
    checks = [
        ("Entropy AUROC on D < 0.65",    ent_D < 0.65),
        ("SUA AUROC on D > 0.75",        sua_D > 0.75),
        ("ΔAUROC_D > 0.30",              delta_D > 0.30),
        ("SUA overall AUROC > Entropy",
         compute_auroc(SUA, data.errors) > compute_auroc(H, data.errors)),
    ]
    for desc, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {desc}")
    all_pass = all(p for _, p in checks)
    print(f"\n  {'All checks passed ✓' if all_pass else 'Some checks failed ✗'}")

    # ── Save results ──────────────────────────────────────────────
    results = {
        "tier": 1,
        "n_total": int(len(data.probs)),
        "seed": SEED,
        "lambda": LAMBDA,
        "lambda_sweep": lambda_sweep,
        "regime_aurocs": regime_aurocs_table,
        "delta_auroc_D": float(delta_D),
        "overall_acc": float(overall_acc),
        "ece": float(ece_val),
        "overall_aurocs": {
            name: float(compute_auroc(sc, data.errors))
            for name, sc in methods.items()
        },
        "sel_acc_80": {
            name: float(selective_accuracy(sc, data.errors, COVERAGE))
            for name, sc in methods.items()
        },
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    save_results(results, RESULTS_DIR / "tier1_synthetic.json")
    logger.info(f"Results saved to {RESULTS_DIR}/tier1_synthetic.json")

    # ── Figures ───────────────────────────────────────────────────
    logger.info("Generating figures...")
    FIGURES_DIR.mkdir(exist_ok=True)
    (FIGURES_DIR / "pdf").mkdir(exist_ok=True)

    plot_sh_scatter(
        S, H, data.errors,
        regime_d_mask=analyzer.regime_d_mask(S, H),
        title="S × H Map: Errors Concentrate in Regime D\nNeurIPS 2026 · Hiremath & Hiremath",
        save_name="fig1_sh_scatter",
    )
    plot_regime_auroc_bars(
        {k: v for k, v in list(regime_aurocs_table.items())[:3]},
        title="Failure-Prediction AUROC by Regime\nNeurIPS 2026 · Hiremath & Hiremath",
        save_name="fig2_regime_auroc",
    )
    plot_selective_accuracy(
        {k: v for k, v in list(methods.items())[:3]},
        data.errors,
        title="Selective Accuracy vs Coverage\nNeurIPS 2026 · Hiremath & Hiremath",
        save_name="fig5_selective_accuracy",
    )
    plot_sua_distribution(
        SUA, data.regime_labels,
        title="SUA Score Distribution by Regime\nNeurIPS 2026 · Hiremath & Hiremath",
        save_name="fig4_sua_distribution",
    )
    plot_combined_results(
        S, H, data.errors, data.regime_labels, SUA,
        {k: v for k, v in list(regime_aurocs_table.items())[:3]},
        {k: v for k, v in list(methods.items())[:3]},
        save_name="fig_combined_synthetic",
    )
    logger.info("Figures saved to figures/")
    logger.info("Tier 1 complete.")


if __name__ == "__main__":
    main()
