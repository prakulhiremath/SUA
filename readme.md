# When Confidence Metrics Fail: Sensitivity–Uncertainty Alignment as a Diagnostic for High-Risk Model Errors

[![Tests](https://img.shields.io/badge/tests-pytest-green)](tests/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

---

## Overview

Standard uncertainty metrics for language models — predictive entropy and self-consistency — share a fundamental structural blind spot: they measure properties of the output distribution **at a single input point** and therefore cannot detect whether a model's expressed confidence is appropriate given how unstable its outputs are under semantics-preserving perturbations.

This repository implements the **Sensitivity–Uncertainty Alignment (SUA)** score:

```
SUA(x; ε, λ) = S_θ(x; ε) − λ · H_θ(Y | x)
```

Where:
- `S_θ(x; ε)` = distributional sensitivity: expected KL divergence under semantics-preserving perturbations
- `H_θ(Y | x)` = predictive entropy: expressed uncertainty
- `λ` = exchange-rate parameter selected by validation AUROC

**SUA is NOT a universal replacement for entropy.** It is a targeted diagnostic for a specific failure mode — Regime D — that entropy cannot detect by construction.

---

## The Four-Regime Taxonomy

| Regime | Sensitivity | Entropy | Model Behavior |
|--------|-------------|---------|----------------|
| A | Low | Low | Stable and confident. Correct. |
| B | High | High | Unstable and uncertain. Correctly hedged. |
| C | Low | High | Unnecessarily uncertain. No failure. |
| **D** | **High** | **Low** | **Confident despite instability. FAILURE MODE.** |

Only Regime D is invisible to entropy and self-consistency.

---

## Key Results

| Metric | Regime A | Regime B | Regime C | **Regime D** |
|--------|----------|----------|----------|--------------|
| Entropy AUROC | 0.545 | 0.482 | 0.515 | **0.228** (inverted) |
| Self-consist. AUROC | 0.500 | 0.526 | 0.521 | **0.500** (random) |
| **SUA AUROC** | 0.462 | 0.621 | 0.494 | **0.877** |

**ΔAUROC_D = +0.649** (SUA over entropy on Regime D)

### Where SUA helps (and where it doesn't)

| Setting | Regime D% | ΔAUROC (SUA−Entropy) |
|---------|-----------|----------------------|
| Standard NLI benchmarks | < 10% | −0.02 to −0.18 (**entropy wins**) |
| Adversarial QA (Mistral-7B avg.) | 23.5% | **+0.091** |
| Ambiguous NLI avg. | 28.0% | **+0.065** |
| OOD classification avg. | 19.4% | **+0.068** |

---

## Installation

```bash
git clone <repo>
cd sua
pip install -e "."

# For LLM experiments (Tier 3, requires GPU):
pip install -e ".[llm]"
```

---

## Reproduction

### One-command reproduction

```bash
bash reproduce.sh           # All tiers
bash reproduce.sh --tier 1  # Tier 1 only (synthetic, ~2 min, no GPU)
bash reproduce.sh --tier 2  # Tier 2 only (real NLI, ~30 min, no GPU)
bash reproduce.sh --tier 3  # Tier 3 only (LLM, requires A100, ~62 GPU-hours)
bash reproduce.sh --test    # Unit tests only
```

### Tier-by-tier

```bash
# Tier 1: Synthetic benchmark (reproduces Table 1, Figs 1–6)
python experiments/synthetic/run_synthetic.py

# Tier 2: Real NLI (reproduces Table tab:real_sh)
python experiments/nli/run_nli.py

# Tier 3: LLM adversarial QA (reproduces Table tab:llm_qa, requires GPU)
python experiments/llm/run_adversarial_qa.py --model mistral
python experiments/llm/run_adversarial_qa.py --model llama   # replication

# Generate all figures
python scripts/generate_figures.py

# Generate LaTeX/CSV tables
python scripts/generate_tables.py
```

### Run tests

```bash
make test
# or
pytest tests/ -v
```

---

## Repository Structure

```
sua/
├── src/
│   ├── metrics/
│   │   ├── sua.py              ← Core SUA score + lambda sweep
│   │   ├── entropy.py          ← Predictive entropy H(Y|x)
│   │   ├── self_consistency.py ← Self-consistency baseline
│   │   ├── semantic_entropy.py ← Semantic entropy baseline (Kuhn et al.)
│   │   └── spuq.py             ← SPUQ baseline (Gao et al. 2024)
│   ├── perturbations/
│   │   ├── fill_mask.py        ← Word-level synonym perturbations
│   │   ├── token_edit.py       ← Token-level edits
│   │   └── perturbation_validity.py ← S/H ratio pre-filter
│   ├── datasets/
│   │   └── synthetic.py        ← Four-regime Dirichlet benchmark
│   ├── models/
│   │   └── wrappers.py         ← Unified model probability extraction
│   ├── evaluation/
│   │   ├── auroc.py            ← Per-regime AUROC computation
│   │   ├── calibration.py      ← ECE / ACE
│   │   ├── selective_prediction.py ← Selective accuracy, HCE reduction
│   │   └── regime_analysis.py  ← Regime assignment and statistics
│   ├── theory/
│   │   └── risk_bounds.py      ← Theorem 4.1 / 4.2 computations
│   └── plotting/
│       └── figures.py    ← All  figures
├── experiments/
│   ├── synthetic/run_synthetic.py  ← Tier 1
│   ├── nli/run_nli.py              ← Tier 2
│   ├── qa/run_qa.py                ← Tier 2 QA
│   ├── ood/run_ood.py              ← Tier 3 OOD
│   └── llm/run_adversarial_qa.py  ← Tier 3 LLM
├── tests/
│   ├── test_metrics.py
│   ├── test_evaluation.py
│   └── test_synthetic.py       ← Integration test for claims
├── scripts/
│   ├── generate_figures.py
│   └── generate_tables.py
├── configs/
│   ├── synthetic.yaml
│   ├── nli.yaml
│   └── llm.yaml
├── reproduce.sh                ← One-command reproduction
└── results/                    ← Experiment outputs (JSON + LaTeX)
```

---

## The S/H Validity Pre-Filter

Before running SUA, always check the S/H ratio:

```python
from src.perturbations.perturbation_validity import SHValidator

validator = SHValidator()
result = validator.validate(S, H)
print(result["message"])
# S/H=0.537 < 1.0 — perturbations valid.
# S/H=4.232 >= 1.0 — perturbations NOT semantics-preserving. SUA results unreliable.
```

**If S/H > 1.0, SUA results are unreliable.** The threshold follows from Definition 2.1 of the paper, not from empirical tuning.

---

## When to Use SUA vs Entropy

```
Compute S/H ratio on held-out split
        |
    S/H > 1.0? ──────────────→ STOP. Perturbation pipeline invalid.
        |
    Estimate Regime D population
        |
    RegD% < 10%? ────────────→ Use entropy. Simpler and more accurate.
        |
    RegD% >= 20%? ───────────→ Run SUA with λ selected by validation AUROC.
                                Report S/H, RegD%, λ* as diagnostic metadata.
```

---

## Compute Requirements

| Tier | Hardware | Time |
|------|----------|------|
| Tier 1: Synthetic | CPU only | ~2 min |
| Tier 2: Real NLI | CPU or GPU | ~30 min |
| Tier 3: LLM (Mistral-7B) | 2× A100 40GB | ~48 GPU-hours |
| Tier 3: LLM (LLaMA-3, replication) | 2× A100 40GB | ~14 GPU-hours |

---

## Citation

```bibtex
@misc{sua2026,
  title        = {When Confidence Metrics Fail: Sensitivity--Uncertainty Alignment as a Diagnostic for High-Risk Model Errors},
  year         = {2026},
  note         = {Under review}
}
```

---

## Related Work

- **SPUQ** (Gao et al., EACL 2024): Perturbation-based uncertainty quantification. SUA differs by conditioning sensitivity on expressed entropy, detecting only the *misaligned* regime.
- **Semantic Entropy** (Kuhn et al., ICLR 2023): Measures output variability; SUA measures input-perturbation sensitivity.
- **Self-consistency** (Wang et al., ICLR 2023): Agrees across samples at same input; blind to Regime D.
- **Temperature Scaling** (Guo et al., ICML 2017): Global calibration; SUA is per-input and local.

---

## License

MIT License. See [LICENSE](LICENSE).
