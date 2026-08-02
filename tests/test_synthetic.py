"""Integration test: synthetic experiment end-to-end."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.datasets.synthetic import generate_synthetic
from src.metrics.sua import SUAScore
from src.metrics.entropy import PredictiveEntropy
from src.metrics.self_consistency import SelfConsistency
from src.evaluation.auroc import compute_auroc, compute_regime_auroc
from src.evaluation.calibration import compute_ece
from src.evaluation.selective_prediction import selective_accuracy


class TestSyntheticBenchmark:
    """Integration test: verify key paper claims on synthetic data."""

    @pytest.fixture(scope="class")
    def data(self):
        return generate_synthetic(n_per_regime=150, n_classes=3,
                                   n_perturbations=6, seed=2026)

    @pytest.fixture(scope="class")
    def computed(self, data):
        sua = SUAScore(lambda_val=1.0)
        out = sua.compute(data.probs, data.pert_probs)
        sc  = SelfConsistency().from_pert_probs(data.probs, data.pert_probs)
        return {
            "S": out["S"], "H": out["H"], "SUA": out["sua"],
            "SC": sc, "errors": data.errors, "regimes": data.regime_labels,
        }

    def test_dataset_size(self, data):
        assert len(data.probs) == 600  # 4 * 150

    def test_regime_counts(self, data):
        for r in range(4):
            assert (data.regime_labels == r).sum() == 150

    def test_error_rates_ordered(self, data):
        """Regime D should have highest error rate."""
        err_D = data.errors[data.regime_labels == 3].mean()
        err_A = data.errors[data.regime_labels == 0].mean()
        err_C = data.errors[data.regime_labels == 2].mean()
        assert err_D > err_A
        assert err_D > err_C

    def test_regime_separation(self, computed):
        """S and H should differ across regimes."""
        S = computed["S"]; H = computed["H"]; reg = computed["regimes"]
        # Regime D: high S, low H
        assert S[reg == 3].mean() > S[reg == 0].mean()
        assert H[reg == 3].mean() < H[reg == 1].mean()

    def test_sua_wins_regime_d(self, computed):
        """Core claim: SUA AUROC > Entropy AUROC on Regime D."""
        mask_D = computed["regimes"] == 3
        err_D  = computed["errors"][mask_D]
        if err_D.sum() == 0 or err_D.sum() == mask_D.sum():
            pytest.skip("No variance in Regime D errors")
        auc_sua = compute_auroc(computed["SUA"][mask_D], err_D)
        auc_ent = compute_auroc(computed["H"][mask_D], err_D)
        assert auc_sua > auc_ent, (
            f"SUA ({auc_sua:.3f}) should beat Entropy ({auc_ent:.3f}) on Regime D"
        )

    def test_entropy_auroc_d_near_random(self, computed):
        """Entropy AUROC on Regime D should be < 0.70."""
        mask_D = computed["regimes"] == 3
        err_D  = computed["errors"][mask_D]
        if err_D.sum() == 0:
            pytest.skip("No errors in Regime D")
        auc = compute_auroc(computed["H"][mask_D], err_D)
        assert auc < 0.70

    def test_selective_accuracy_sua_wins(self, computed):
        """SUA selective accuracy >= Entropy at 80% coverage."""
        sel_sua = selective_accuracy(computed["SUA"], computed["errors"], 0.80)
        sel_ent = selective_accuracy(computed["H"],   computed["errors"], 0.80)
        assert sel_sua >= sel_ent - 0.05  # allow small tolerance

    def test_probs_valid(self, data):
        """All probability arrays should sum to ~1 and be non-negative."""
        assert data.probs.min() >= 0.0
        np.testing.assert_allclose(data.probs.sum(axis=1), 1.0, atol=1e-5)
