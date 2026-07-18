"""
Poisson and Negative Binomial count-data regression (from scratch).
"""
import numpy as np
from scipy import optimize
from scipy.special import gammaln


def _design(X):
    X = np.asarray(X, float)
    return np.column_stack([np.ones(len(X)), X]) if X.ndim == 1 else X


def poisson_fit(X, y):
    """Poisson GLM via IRLS."""
    X = _design(X); y = np.asarray(y, float)
    beta = np.zeros(X.shape[1])
    for _ in range(100):
        eta = X @ beta
        mu = np.exp(eta)
        W = mu
        z = eta + (y - mu) / mu
        XtW = X.T * W
        H = XtW @ X
        grad = XtW @ (z - eta)
        try:
            delta = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            delta = np.zeros_like(beta)
        beta += delta
        if np.max(np.abs(delta)) < 1e-8:
            break
    return beta


def negbin_fit(X, y, alpha=1.0):
    """Negative Binomial with fixed dispersion alpha."""
    X = _design(X); y = np.asarray(y, float)
    beta = np.zeros(X.shape[1])
    for _ in range(200):
        eta = X @ beta
        mu = np.exp(eta)
        # variance = mu + alpha * mu^2
        V = mu + alpha * mu ** 2
        W = 1.0 / V
        z = eta + (y - mu) / mu
        XtW = X.T * W
        H = XtW @ X
        grad = XtW @ (z - eta)
        try:
            delta = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            delta = np.zeros_like(beta)
        beta += delta
        if np.max(np.abs(delta)) < 1e-8:
            break
    return beta


def poisson_predict(beta, X):
    return np.exp(_design(X) @ beta)
