import numpy as np
from src.count_data import poisson_fit, poisson_predict

def test_poisson():
    np.random.seed(1)
    X = np.random.randn(300, 1)
    beta_true = np.array([0.8])
    lam = np.exp(0.2 + X @ beta_true)
    y = np.random.poisson(lam)
    b = poisson_fit(X, y)
    assert np.isfinite(b).all()
    mu = poisson_predict(b, X)
    assert (mu > 0).all()
