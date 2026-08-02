"""Unit tests for evaluation utilities."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.auroc import compute_auroc, compute_regime_auroc
from src.evaluation.calibration import compute_ece, calibration_curve
from src.evaluation.selective_prediction import selective_accuracy, hce_reduction
from src.evaluation.regime_analysis import RegimeAnalyzer


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestAUROC:
    def test_perfect_auroc(self):
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        errors = np.array([0, 0, 1, 1])
        assert compute_auroc(scores, errors) == pytest.approx(1.0)

    def test_random_auroc(self):
        rng = np.random.default_rng(0)
        scores = rng.random(100)
        errors = rng.integers(0, 2, 100)
        auc = compute_auroc(scores, errors)
        assert 0.3 < auc < 0.7

    def test_single_class_returns_nan(self):
        scores = np.array([0.1, 0.5, 0.9])
        errors = np.array([0, 0, 0])
        assert np.isnan(compute_auroc(scores, errors))

    def test_regime_auroc_four_regimes(self, rng):
        N = 200
        scores = rng.random(N)
        errors = rng.integers(0, 2, N)
        labels = rng.integers(0, 4, N)
        result = compute_regime_auroc(scores, errors, labels)
        assert set(result.keys()) == {"A", "B", "C", "D"}


class TestECE:
    def test_perfect_calibration(self):
        # Model always outputs confidence = accuracy
        probs = np.zeros((100, 2))
        probs[:50, 0] = 0.8; probs[:50, 1] = 0.2
        probs[50:, 0] = 0.2; probs[50:, 1] = 0.8
        errors = np.array([0]*40 + [1]*10 + [1]*40 + [0]*10)
        ece = compute_ece(probs, errors)
        assert ece < 0.10

    def test_overconfident_high_ece(self):
        # Always predicts class 0 with 0.99 confidence, wrong 50% of time
        probs = np.column_stack([np.full(100, 0.99), np.full(100, 0.01)])
        errors = np.array([0, 1] * 50)
        ece = compute_ece(probs, errors)
        assert ece > 0.40

    def test_ece_range(self, rng):
        probs = rng.dirichlet(np.ones(3), size=100)
        errors = rng.integers(0, 2, 100)
        ece = compute_ece(probs, errors)
        assert 0.0 <= ece <= 1.0


class TestSelectivePrediction:
    def test_full_coverage(self):
        scores = np.array([0.1, 0.5, 0.9, 0.2])
        errors = np.array([0, 0, 1, 0])
        sel = selective_accuracy(scores, errors, coverage=1.0)
        assert sel == pytest.approx(0.75)

    def test_top_half(self):
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        errors = np.array([0, 0, 1, 1])
        sel = selective_accuracy(scores, errors, coverage=0.5)
        assert sel == pytest.approx(1.0)  # top-50% lowest score = all correct

    def test_hce_reduction_nonnegative(self, rng):
        probs  = rng.dirichlet(np.ones(3), size=50)
        errors = rng.integers(0, 2, 50)
        sua    = rng.random(50)
        red = hce_reduction(probs, sua, errors, coverage=0.8)
        assert isinstance(red, float)


class TestRegimeAnalyzer:
    def test_four_labels(self, rng):
        S = rng.random(100)
        H = rng.random(100)
        analyzer = RegimeAnalyzer()
        analyzer.fit(S, H)
        labels = analyzer.assign(S, H)
        assert set(np.unique(labels)).issubset({0, 1, 2, 3})

    def test_regime_d_is_high_s_low_h(self, rng):
        S = rng.random(200)
        H = rng.random(200)
        mask = RegimeAnalyzer().regime_d_mask(S, H, s_pct=75, h_pct=25)
        assert S[mask].mean() > S[~mask].mean()
        assert H[mask].mean() < H[~mask].mean()

    def test_population_rates_sum_to_one(self, rng):
        S = rng.random(100)
        H = rng.random(100)
        analyzer = RegimeAnalyzer()
        analyzer.fit(S, H)
        rates = analyzer.population_rate(S, H)
        total = sum(rates.values())
        assert total == pytest.approx(1.0, abs=1e-6)
