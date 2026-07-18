"""
Heckman two-step selection model (from scratch).

Models: y1* = x1*b1 + u1  (selection equation, observed if y1* > 0)
        y2  = x2*b2 + u2  (outcome equation, observed only when selected)
"""
import numpy as np
from scipy import stats
from scipy.stats import norm


def heckman_two_step(x_sel, x_out, y_sel, y_out_obs):
    """
    Parameters
    ----------
    x_sel : (n, k1) selection covariates
    x_out : (n, k2) outcome covariates
    y_sel : (n,) binary selection indicator (1 = selected)
    y_out_obs : (n,) outcome, NaN/0 where not selected

    Returns
    -------
    beta_sel, beta_out, rho, lambda_(mills), inverse_mills
    """
    x_sel = np.asarray(x_sel); x_out = np.asarray(x_out)
    y_sel = np.asarray(y_sel).astype(float)
    y_out_obs = np.asarray(y_out_obs, dtype=float)

    # Step 1: Probit for selection
    n, k1 = x_sel.shape
    beta_sel = np.zeros(k1)
    for _ in range(100):  # Newton-Raphson
        z = x_sel @ beta_sel
        p = norm.cdf(z)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        # gradient
        w = (norm.pdf(z) ** 2) / (p * (1 - p))
        grad = x_sel.T @ (y_sel - p)
        # Hessian (diagonal approx for stability)
        H = x_sel.T * w @ x_sel
        try:
            delta = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            delta = grad * 0.01
        beta_sel = beta_sel + delta
        if np.max(np.abs(delta)) < 1e-6:
            break

    # Inverse Mills Ratio for selected observations
    z = x_sel @ beta_sel
    pdf = norm.pdf(z); cdf = norm.cdf(z)
    imr = pdf / cdf  # lambda for selected (y_sel==1)

    # Step 2: Outcome regression augmented with IMR
    sel_idx = y_sel == 1
    X_aug = np.column_stack([x_out[sel_idx], imr[sel_idx]])
    y = y_out_obs[sel_idx]
    beta_aug, *_ = np.linalg.lstsq(X_aug, y, rcond=None)

    beta_out = beta_aug[:-1]
    rho_coef = beta_aug[-1]  # coefficient on IMR
    return beta_sel, beta_out, rho_coef, imr
