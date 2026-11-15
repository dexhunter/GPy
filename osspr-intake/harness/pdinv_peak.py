"""Peak memory of GPy.util.linalg.pdinv.

Workload: inverse, Cholesky factor, and inverse-Cholesky of a 1000 x 1000
positive definite covariance, the matrix exact Gaussian inference factorizes
once per marginal-likelihood evaluation.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from GPy.util.linalg import pdinv  # noqa: E402

from peakutil import peak_bytes  # noqa: E402


def main():
    rng = np.random.RandomState(3)
    n = 1000
    factors = rng.randn(n, 40)
    covariance = np.ascontiguousarray(factors.dot(factors.T))
    covariance.flat[:: n + 1] += 1e-2
    holder = {}

    def run():
        # pdinv / jitchol overwrite their argument; copy inside the measurement
        # the same way ExactGaussianInference copies K before factorizing.
        holder["out"] = pdinv(covariance.copy())

    peak = peak_bytes(run)
    inverse, cholesky, inverse_cholesky, logdet = holder["out"]
    if inverse.shape != (n, n) or cholesky.shape != (n, n) or inverse_cholesky.shape != (n, n):
        raise SystemExit("pdinv returned unexpected shapes")
    if not np.isfinite(logdet) or not np.isfinite(inverse).all():
        raise SystemExit("pdinv returned non-finite values")
    print(f"peak_memory_bytes={peak}")


if __name__ == "__main__":
    main()
