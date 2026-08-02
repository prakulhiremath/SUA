import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.metrics.entropy import PredictiveEntropy
from src.metrics.sua import SUAScore, kl_divergence, DistributionalSensitivity
from src.metrics.self_consistency import SelfConsistency
from src.metrics.semantic_entropy import SemanticEntropy


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def uniform_probs(rng):
    """Near-uniform distributions — high entropy."""
    return rng.dirichlet(np.ones(3) * 3.0, size=20)


@pytest.fixture
def peaked_probs(rng):
    """Peaked distributions — low entropy."""
    probs = np.zeros((20, 3))
    for i in range(20):
        p = np.zeros(3)
        p[rng.integers(0, 3)] = 0.95
        p += 0.025
        p /= p.sum()
        probs[i] = p
    return probs


class TestKLDivergence:
    def test_kl_zero_identical(self):
        p = np.array([0.5, 0.3, 0.2])
        assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-6)

    def test_kl_nonnegative(self, rng):
        for _ in range(10):
            p = rng.dirichlet(np.ones(4))
            q = rng.dirichlet(np.ones(4))
            assert kl_divergence(p, q) >= 0.0

    def test_kl_asymmetric(self, rng):
        p = np.array([0.9, 0.05, 0.05])
        q = np.array([0.1, 0.45, 0.45])
        assert kl_divergence(p, q) != pytest.approx(kl_divergence(q, p), abs=0.1)


class TestPredictiveEntropy:
    def test_uniform_max_entropy(self):
        H_fn = PredictiveEntropy()
        uniform = np.ones((5, 3)) / 3.0
        H = H_fn(uniform)
        expected = np.log(3)
        np.testing.assert_allclose(H, expected, atol=1e-6)

    def test_peaked_low_entropy(self, peaked_probs):
        H_fn = PredictiveEntropy()
        H = H_fn(peaked_probs)
        assert H.mean() < 0.3

    def test_uniform_higher_than_peaked(self, uniform_probs, peaked_probs):
        H_fn = PredictiveEntropy()
        assert H_fn(uniform_probs).mean() > H_fn(peaked_probs).mean()

    def test_shape(self, uniform_probs):
        H_fn = PredictiveEntropy()
        assert H_fn(uniform_probs).shape == (20,)


class TestSUAScore:
    def make_pert(self, base, rng, tight=True):
        N, C = base.shape
        pert_list = []
        for k in range(4):
            pk = np.zeros((N, C))
            for i in range(N):
                if tight:
                    conc = 30.0 * base[i] + 0.5
                else:
                    pk[i] = rng.dirichlet(np.ones(C) * 0.3)
                    continue
                pk[i] = rng.dirichlet(conc)
            pert_list.append(pk)
        return pert_list

    def test_sua_positive_regimeD(self, peaked_probs, rng):
        """Regime D: peaked + high-shift perturbations → SUA > 0."""
        # Shift-perturbation: peak moves to another class
        N, C = peaked_probs.shape
        pert_list = []
        for _ in range(4):
            pk = np.zeros((N, C))
            for i in range(N):
                dom = np.argmax(peaked_probs[i])
                nd  = (dom + 1) % C
                p   = np.zeros(C); p[nd] = 0.9; p += 0.05; p /= p.sum()
                pk[i] = p
            pert_list.append(pk)
        sua = SUAScore(lambda_val=1.0)
        out = sua.compute(peaked_probs, pert_list)
        # Regime D: high S, low H → SUA should be positive for most
        assert out["sua"].mean() > 0.0

    def test_sua_negative_regimeA(self, peaked_probs, rng):
        """Regime A: peaked + tight perturbations → SUA < 0."""
        pert_list = self.make_pert(peaked_probs, rng, tight=True)
        sua = SUAScore(lambda_val=1.0)
        out = sua.compute(peaked_probs, pert_list)
        # Low S, low H: SUA ≈ -H < 0
        assert out["sua"].mean() < 0.0

    def test_regime_labels(self, uniform_probs, peaked_probs, rng):
        """Regime assignment returns correct integer codes."""
        sua = SUAScore()
        pert_u = self.make_pert(uniform_probs, rng, tight=False)
        pert_p = self.make_pert(peaked_probs,  rng, tight=True)
        out_u = sua.compute(uniform_probs, pert_u)
        out_p = sua.compute(peaked_probs,  pert_p)

        S = np.concatenate([out_u["S"], out_p["S"]])
        H = np.concatenate([out_u["H"], out_p["H"]])

        labels = sua.regime_labels(S, H)
        assert set(np.unique(labels)).issubset({0, 1, 2, 3})

    def test_sh_ratio_validity(self, peaked_probs, rng):
        """S/H ratio correctly identifies invalid perturbations."""
        from src.perturbations.perturbation_validity import SHValidator
        pert_list = self.make_pert(peaked_probs, rng, tight=True)
        out = SUAScore().compute(peaked_probs, pert_list)
        validator = SHValidator()
        result = validator.validate(out["S"], out["H"])
        assert "sh_ratio" in result
        assert "valid" in result


class TestSelfConsistency:
    def test_perfect_consistency(self):
        probs = np.array([[0.9, 0.05, 0.05]] * 5)
        pert  = [probs.copy() for _ in range(4)]
        sc    = SelfConsistency().from_pert_probs(probs, pert)
        np.testing.assert_allclose(sc, 1.0, atol=1e-6)

    def test_zero_consistency(self):
        base = np.array([[0.9, 0.05, 0.05]] * 5)
        # Perturbs always predict a different class
        pert_p = base.copy()
        pert_p[:, 0] = 0.05; pert_p[:, 1] = 0.9; pert_p[:, 2] = 0.05
        pert  = [pert_p for _ in range(4)]
        sc    = SelfConsistency().from_pert_probs(base, pert)
        np.testing.assert_allclose(sc, 0.0, atol=1e-6)

    def test_range(self, rng):
        probs = rng.dirichlet(np.ones(3), size=10)
        pert  = [rng.dirichlet(np.ones(3), size=10) for _ in range(4)]
        sc    = SelfConsistency().from_pert_probs(probs, pert)
        assert sc.min() >= 0.0
        assert sc.max() <= 1.0


class TestSemanticEntropy:
    def test_higher_than_base_entropy(self, peaked_probs, rng):
        """SE from heterogeneous perturbations > base entropy."""
        from src.metrics.entropy import PredictiveEntropy
        N, C = peaked_probs.shape
        pert = [rng.dirichlet(np.ones(C) * 0.3, size=N) for _ in range(4)]
        se = SemanticEntropy().from_pert_probs(peaked_probs, pert)
        H  = PredictiveEntropy()(peaked_probs)
        # Mixture should be more uniform → higher entropy
        assert se.mean() >= H.mean()
