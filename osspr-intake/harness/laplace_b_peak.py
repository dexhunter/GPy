"""Peak memory of Laplace._compute_B_statistics.

Workload: one Newton step of Laplace inference on 1000 observations.
K is a dense prior covariance; W is a positive diagonal negative-Hessian
(log-concave likelihood: Gaussian, Poisson, or Gamma).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from GPy.inference.latent_function_inference.laplace import Laplace  # noqa: E402

from peakutil import peak_bytes  # noqa: E402


def main():
    rng = np.random.RandomState(2)
    n = 1000
    factors = rng.randn(n, 16)
    covariance = factors.dot(factors.T)
    covariance.flat[:: n + 1] += 1.0
    hessian_diag = rng.rand(n, 1) + 0.2
    laplace = Laplace()
    holder = {}

    def run():
        holder["out"] = laplace._compute_B_statistics(covariance, hessian_diag, True)

    peak = peak_bytes(run)
    k_wi_i, logdet, i_kw_i, ki_w_i = holder["out"]
    if k_wi_i.shape != (n, n) or i_kw_i.shape != (n, n) or ki_w_i.shape != (n, n):
        raise SystemExit("Laplace._compute_B_statistics returned unexpected shapes")
    if not np.isfinite(logdet) or not np.isfinite(k_wi_i).all():
        raise SystemExit("Laplace._compute_B_statistics returned non-finite values")
    print(f"peak_memory_bytes={peak}")


if __name__ == "__main__":
    main()
