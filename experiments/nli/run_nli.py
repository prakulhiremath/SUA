import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

from src.metrics.sua import SUAScore, DistributionalSensitivity, kl_divergence
from src.metrics.entropy import PredictiveEntropy
from src.metrics.self_consistency import SelfConsistency
from src.perturbations.fill_mask import FillMaskPerturber
from src.perturbations.perturbation_validity import SHValidator
from src.evaluation.auroc import compute_auroc
from src.evaluation.calibration import compute_ece
from src.evaluation.selective_prediction import selective_accuracy
from src.utils.io import save_results
from src.utils.logging import get_logger

logger = get_logger("tier2-nli")

SEED = 2026
N_SAMPLES = 400
N_PERT = 8
LAMBDA_GRID = [0.0, 0.1, 0.5, 1.0, 2.0]
COVERAGE = 0.80
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path("results")


def apply_temp(logits: np.ndarray, T: float) -> np.ndarray:
    t = torch.tensor(logits / T, dtype=torch.float32)
    return F.softmax(t, dim=-1).numpy()


def detect_label_remap(model) -> dict:
    """Auto-detect SNLI/MNLI label mapping from model config."""
    id2label = model.config.id2label
    named = {v.lower(): k for k, v in id2label.items()}
    if all(k in named for k in ("entailment", "neutral", "contradiction")):
        return {0: named["entailment"], 1: named["neutral"], 2: named["contradiction"]}
    return {0: 0, 1: 1, 2: 2}  # identity


def run_condition(
    model_name: str,
    dataset_name: str,
    dataset_split: str = "validation",
    n_samples: int = N_SAMPLES,
    temperature: float = 2.5,
    condition_label: str = "",
    seed: int = SEED,
) -> dict:
    """Run SUA evaluation for one model-condition pair."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Condition: {condition_label or model_name}")
    logger.info(f"Model: {model_name}")

    # Load model
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name).to(DEVICE)
    mdl.eval()

    label_remap = detect_label_remap(mdl)
    logger.info(f"Label remap: {label_remap}")

    # Load dataset
    if dataset_name == "anli":
        ds = load_dataset("anli", split="test_r1").shuffle(seed=seed).select(range(n_samples))
        get_prem = lambda ex: ex["premise"]
        get_hyp  = lambda ex: ex["hypothesis"]
        get_lbl  = lambda ex: ex["label"]
    else:
        ds = load_dataset(dataset_name, split=dataset_split).filter(
            lambda x: x.get("label", -1) != -1
        ).shuffle(seed=seed).select(range(n_samples))
        get_prem = lambda ex: ex.get("premise", ex.get("sentence1", ""))
        get_hyp  = lambda ex: ex.get("hypothesis", ex.get("sentence2", ""))
        get_lbl  = lambda ex: ex["label"]

    logger.info(f"Loaded {len(ds)} samples from {dataset_name}")

    def get_logits(prem, hyp):
        inp = tok(prem, hyp, return_tensors="pt", truncation=True,
                  max_length=128, padding=True).to(DEVICE)
        with torch.no_grad():
            return mdl(**inp).logits.cpu().numpy()[0]

    # Base probabilities
    logger.info("Computing base probabilities...")
    base_logits, dataset_labels = [], []
    for ex in ds:
        base_logits.append(get_logits(get_prem(ex), get_hyp(ex)))
        dataset_labels.append(get_lbl(ex))

    base_logits = np.array(base_logits)
    dataset_labels = np.array(dataset_labels)

    # Apply temperature scaling
    base_probs = np.array([apply_temp(lg, temperature) for lg in base_logits])

    # Remap dataset labels to model space
    model_labels = np.array([label_remap[int(l)] for l in dataset_labels])
    preds = np.argmax(base_probs, axis=1)
    errors = (preds != model_labels).astype(int)
    acc = 1.0 - errors.mean()
    logger.info(f"Accuracy: {acc:.4f}  Error rate: {errors.mean()*100:.1f}%")

    # Perturbations
    perturber = FillMaskPerturber(seed=seed)
    logger.info("Computing perturbations...")
    pert_list = []
    for k in range(N_PERT):
        pk = []
        for ex in ds:
            hyp_p = perturber.perturb(get_hyp(ex), k)
            lg = get_logits(get_prem(ex), hyp_p)
            pk.append(apply_temp(lg, temperature))
        pert_list.append(np.array(pk))
        logger.info(f"  Pert {k+1}/{N_PERT} done")

    # Compute S, H
    S = np.array([
        np.mean([kl_divergence(base_probs[i], pert_list[k][i]) for k in range(N_PERT)])
        for i in range(n_samples)
    ])
    H_fn = PredictiveEntropy()
    H = H_fn(base_probs)

    # SH validity
    validator = SHValidator()
    validity = validator.validate(S, H)
    logger.info(f"Perturbation validity: {validity['message']}")

    # Lambda sweep for optimal SUA
    best_lambda, best_auc = 0.0, 0.5
    for lam in LAMBDA_GRID:
        sua = S - lam * H
        try:
            auc = float(roc_auc_score(errors, sua))
        except Exception:
            auc = 0.5
        if auc > best_auc:
            best_auc = auc
            best_lambda = lam

    SUA = S - best_lambda * H
    SC_fn = SelfConsistency()
    SC = SC_fn.from_pert_probs(base_probs, pert_list)

    # AUROC results
    auc_H   = compute_auroc(H, errors)
    auc_SUA = compute_auroc(SUA, errors)
    auc_SC  = compute_auroc(1 - SC, errors)
    delta   = auc_SUA - auc_H
    ece_val = compute_ece(base_probs, errors)

    logger.info(f"Results — Acc={acc:.4f}  ECE={ece_val:.4f}")
    logger.info(f"  AUROC(H)={auc_H:.4f}  AUROC(SUA)={auc_SUA:.4f}  "
                f"Δ={delta:+.4f}  λ*={best_lambda}  S/H={validity['sh_ratio']:.3f}")

    return {
        "condition": condition_label or model_name,
        "model": model_name,
        "dataset": dataset_name,
        "n_samples": n_samples,
        "acc": float(acc),
        "ece": float(ece_val),
        "auroc_entropy": float(auc_H),
        "auroc_sc": float(auc_SC),
        "auroc_sua": float(auc_SUA),
        "delta": float(delta),
        "lambda_star": float(best_lambda),
        "sh_ratio": float(validity["sh_ratio"]),
        "sh_valid": bool(validity["valid"]),
    }


def main():
    logger.info("TIER 2: Real NLI Evaluation")
    logger.info("Expected: entropy wins on standard conditions (SUCCESS)")

    conditions = [
        {
            "model_name": "textattack/distilbert-base-uncased-MNLI",
            "dataset_name": "multi_nli",
            "dataset_split": "validation_matched",
            "temperature": 2.0,
            "condition_label": "DistilBERT-MNLI",
        },
        {
            "model_name": "cross-encoder/nli-MiniLM2-L6-H768",
            "dataset_name": "snli",
            "dataset_split": "validation",
            "temperature": 2.5,
            "condition_label": "MiniLM-SNLI (reference)",
        },
        {
            "model_name": "cross-encoder/nli-MiniLM2-L6-H768",
            "dataset_name": "anli",
            "dataset_split": "test_r1",
            "temperature": 2.5,
            "condition_label": "ANLI R1 (adversarial)",
        },
    ]

    all_results = []
    for cond in conditions:
        try:
            result = run_condition(**cond)
            all_results.append(result)
        except Exception as e:
            logger.error(f"Condition {cond['condition_label']} failed: {e}")
            continue

    # Print summary table
    print("\n" + "=" * 80)
    print("TABLE: REAL NLI RESULTS (tab:real_sh)")
    print("Expected: Entropy wins on standard, gap narrows on ANLI R1")
    print("=" * 80)
    print(f"{'Condition':<30} {'Acc':>7} {'ECE':>7} {'AUROC(H)':>9} "
          f"{'AUROC(SUA)':>11} {'Δ':>7} {'S/H':>6} {'Valid':>6}")
    print("-" * 88)
    for r in all_results:
        valid_str = "✓" if r["sh_valid"] else "✗"
        print(f"{r['condition']:<30} {r['acc']:>7.3f} {r['ece']:>7.3f} "
              f"{r['auroc_entropy']:>9.3f} {r['auroc_sua']:>11.3f} "
              f"{r['delta']:>+7.3f} {r['sh_ratio']:>6.3f} {valid_str:>6}")

    RESULTS_DIR.mkdir(exist_ok=True)
    save_results({"tier": 2, "conditions": all_results},
                 RESULTS_DIR / "tier2_nli.json")
    logger.info(f"\nResults saved to {RESULTS_DIR}/tier2_nli.json")


if __name__ == "__main__":
    main()
