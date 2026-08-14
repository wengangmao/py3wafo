import numpy as np
import pytest

from wafo import cov2mod, rindmod
from wafo.gaussian import Rind


def test_rind_probability_fallback():
    n = 5
    covariance = (np.ones((n, n)) - np.eye(n)) * 0.3 + np.eye(n)
    value, error, truncation = Rind(seed=17)(
        covariance, np.zeros(n), -np.inf, -1.2, [-1, n - 1])
    assert np.isclose(value[0], 0.001946, atol=5e-5)
    assert error[0] >= 0
    assert truncation[0] >= 0


def test_rindmod_import_and_configuration():
    rindmod.set_constants(maxpts=2048)
    assert callable(rindmod.rind)


def test_rind_weighted_expectation_fallback():
    rho = 0.3
    covariance = np.array([[1.0, rho], [rho, 1.0]])
    value, error, _ = Rind(seed=23, maxpts=65536)(
        covariance, [0, 0], 0, np.inf, [-1, 1], nt=0)
    exact = (rho * (np.pi / 2 + np.arcsin(rho)) +
             np.sqrt(1 - rho**2)) / (2 * np.pi)
    assert abs(value[0] - exact) < max(4 * error[0], 5e-4)


def test_cov2mod_fails_explicitly_instead_of_returning_bad_density():
    cov2mod.initinteg()
    with pytest.raises(NotImplementedError, match="CROSSPACK"):
        cov2mod.cov2mmpdfreg([], [], [], [], [], [])
