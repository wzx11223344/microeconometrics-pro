"""
Cox Proportional Hazards model (from scratch) using Breslow approximation.
"""
import numpy as np
from scipy import optimize


def cox_ph_fit(X, time, event):
    """
    X : (n, k) covariates
    time : (n,) observed times
    event : (n,) event indicator (1 = event, 0 = censored)
    """
    X = np.asarray(X, float); time = np.asarray(time, float)
    event = np.asarray(event, float)
    n, k = X.shape

    def neg_log_lik(beta):
        eta = X @ beta
        risk = np.exp(eta)
        ll = 0.0
        for i in range(n):
            if event[i] == 1:
                num = risk[i]
                # sum risk over subjects with time >= time[i] (Breslow)
                denom = np.sum(risk[time >= time[i]])
                ll += eta[i] - np.log(denom)
        return -ll

    beta0 = np.zeros(k)
    res = optimize.minimize(neg_log_lik, beta0, method='BFGS')
    return res.x


def hazard_ratio(beta, X_new):
    return np.exp(X_new @ beta)
