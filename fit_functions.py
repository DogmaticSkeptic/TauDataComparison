import numpy as np
from scipy.optimize import curve_fit

# =============================================================================
# Models for Curve Fitting
# =============================================================================
def model_B_k(x, B, kappa):
    return 0.5 * (1 + kappa * B * x)

def model_C(x, C, k0, k1):
    eps = 1e-6
    return -0.5 * (1 + k0 * k1 * C * x) * np.log(np.abs(x) + eps)
