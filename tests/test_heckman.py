import numpy as np
from src.heckman import heckman_two_step

def test_heckman_runs():
    np.random.seed(0)
    n = 500
    x = np.random.randn(n, 2)
    # selection
    z = x @ np.array([0.5, -0.3]) + np.random.randn(n) * 0.5
    sel = (z > 0).astype(float)
    y = x @ np.array([1.0, 2.0]) + np.random.randn(n) * 0.3
    y_obs = np.where(sel == 1, y, np.nan)
    b_sel, b_out, rho, _ = heckman_two_step(x, x, sel, y_obs)
    assert len(b_out) == 2
    assert np.isfinite(b_out).all()
