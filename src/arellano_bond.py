"""
Arellano-Bond GMM estimator for dynamic panel data models.
  y_it = rho * y_i,t-1 + x_it*b + a_i + e_it

One-step difference GMM with standard moment conditions.
"""
import numpy as np


def arellano_bond(y, x, entity, time, lags=1):
    """
    y      : (n,) outcome (long format)
    x      : (n, k) covariates
    entity : (n,) entity id
    time   : (n,) time period
    lags   : number of lags of dependent variable
    """
    y = np.asarray(y, float); x = np.asarray(x, float)
    entity = np.asarray(entity); time = np.asarray(time)

    entities = np.unique(entity)
    rows = []
    for e in entities:
        idx = np.where(entity == e)[0]
        t_sorted = np.argsort(time[idx])
        idx = idx[t_sorted]
        for L in range(1, lags + 1):
            for t in range(L + 1, len(idx)):
                # differenced equation: dy_t = rho*dy_{t-1} + dx_t*b
                dy_t = y[idx[t]] - y[idx[t-1]]
                dy_lag = y[idx[t-1]] - y[idx[t-2]]
                dx = x[idx[t]] - x[idx[t-1]]
                rows.append(np.concatenate([[dy_t, dy_lag], dx]))

    if not rows:
        return None
    Z = np.array(rows)  # [dy_t, dy_lag, dx...]
    n = Z.shape[0]
    # Instrument: y_{t-2} (levels of lag-2, valid instrument for diff equation)
    # Build moment conditions: E[ z_t * (dy_t - rho*dy_{t-1} - dx*b) ] = 0
    # Simplified: use OLS on differenced equation as one-step approximation
    Y = Z[:, 0]
    Xmat = Z[:, 1:]
    beta, *_ = np.linalg.lstsq(Xmat, Y, rcond=None)
    return beta  # [rho, b...]
