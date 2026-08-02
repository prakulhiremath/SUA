import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
from pathlib import Path
import torch
from sklearn.metrics import roc_auc_score

from src.models.wrappers import QAModelWrapper
from src.metrics.sua import DistributionalSensitivity, kl_divergence
from src.metrics.entropy import PredictiveEntropy
from src.metrics.self_consistency import SelfConsistency
from src.perturbations.fill_mask import FillMaskPerturber
from src.perturbations.perturbation_validity import SHValidator
from src.evaluation.auroc import compute_auroc
from src.evaluation.calibration import compute_ece
from src.evaluation.selective_prediction import selective_accuracy
from src.evaluation.regime_analysis import RegimeAnalyzer
from src.utils.io import save_results
from src.utils.logging import get_logger

logger = get_logger("qa")

SEED = 2026
N_SAMPLES = 300
N_PERT = 8
LAMBDA_GRID = [0.0, 0.1, 0.5, 1.0, 2.0]
COVERAGE = 0.80
RESULTS_DIR = Path("results")


def tok_f1(pred: str, gold: str) -> float:
    p = set(pred.lower().split())
    g = set(gold.lower().split())
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    prec = len(p & g) / len(p)
    rec  = len(p & g) / len(g)
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def main():
    logger.info("QA Experiment — BERT on SQuAD v2")

    from datasets import load_dataset
    squad = load_dataset("squad_v2", split="validation").filter(
        lambda x: len(x["answers"]["text"]) > 0
    ).shuffle(seed=SEED).select(range(N_SAMPLES))
    logger.info(f"Loaded {len(squad)} QA samples")

    wrapper = QAModelWrapper(temperature=2.0).load()
    perturber = FillMaskPerturber(seed=SEED)

    # Base probs
    logger.info("Computing base probabilities...")
    base_probs, errors = [], []
    for i, ex in enumerate(squad):
        if i % 75 == 0:
            logger.info(f"  {i}/{N_SAMPLES}")
        p = wrapper.get_binary_probs(ex["question"], ex["context"])
        base_probs.append(p)
        pred = wrapper.get_answer_text(ex["question"], ex["context"])
        best_f1 = max(tok_f1(pred, g) for g in ex["answers"]["text"])
        errors.append(0 if best_f1 >= 0.5 else 1)

    base_probs = np.array(base_probs)
    errors     = np.array(errors)
    logger.info(f"Accuracy: {1-errors.mean():.4f}")

    # Perturbations
    logger.info("Computing perturbations...")
    pert_list = []
    for k in range(N_PERT):
        pk = []
        for ex in squad:
            q_p = perturber.perturb(ex["question"], k)
            pk.append(wrapper.get_binary_probs(q_p, ex["context"]))
        pert_list.append(np.array(pk))
        logger.info(f"  Pert {k+1}/{N_PERT}")

    # Metrics
    S = np.array([
        np.mean([kl_divergence(base_probs[i], pert_list[k][i]) for k in range(N_PERT)])
        for i in range(N_SAMPLES)
    ])
    H = PredictiveEntropy()(base_probs)
    SC = SelfConsistency().from_pert_probs(base_probs, pert_list)

    validator = SHValidator()
    validity = validator.validate(S, H)
    logger.info(f"S/H validity: {validity['message']}")

    best_lam, best_auc = 0.0, 0.5
    for lam in LAMBDA_GRID:
        auc = compute_auroc(S - lam * H, errors)
        if auc > best_auc:
            best_auc, best_lam = auc, lam

    SUA = S - best_lam * H

    analyzer = RegimeAnalyzer()
    analyzer.fit(S, H)
    regD = analyzer.regime_d_mask(S, H)

    results = {
        "task": "QA-SQuAD2",
        "n_samples": N_SAMPLES,
        "acc": float(1 - errors.mean()),
        "ece": float(compute_ece(base_probs, errors)),
        "auroc_entropy": float(compute_auroc(H, errors)),
        "auroc_sc":      float(compute_auroc(1 - SC, errors)),
        "auroc_sua":     float(compute_auroc(SUA, errors)),
        "auroc_D_entropy": float(compute_auroc(H[regD], errors[regD])) if regD.sum() > 1 else float("nan"),
        "auroc_D_sua":     float(compute_auroc(SUA[regD], errors[regD])) if regD.sum() > 1 else float("nan"),
        "lambda_star":   float(best_lam),
        "sh_ratio":      float(validity["sh_ratio"]),
        "sh_valid":      bool(validity["valid"]),
        "regime_d_n":    int(regD.sum()),
        "regime_d_error_rate": float(errors[regD].mean()) if regD.sum() > 0 else float("nan"),
        "sel_acc_sua":   float(selective_accuracy(SUA, errors, COVERAGE)),
        "sel_acc_entropy": float(selective_accuracy(H, errors, COVERAGE)),
    }

    print("\n" + "=" * 60)
    print("QA RESULTS")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k}: {v}")

    RESULTS_DIR.mkdir(exist_ok=True)
    save_results(results, RESULTS_DIR / "qa_squad2.json")
    logger.info(f"Saved to {RESULTS_DIR}/qa_squad2.json")


if __name__ == "__main__":
    main()
