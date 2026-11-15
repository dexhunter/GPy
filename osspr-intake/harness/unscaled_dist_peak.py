"""Peak memory of Stationary._unscaled_dist for a cross-covariance.

Workload: Euclidean distances between 1800 training rows and 1800 test
rows in 8 dimensions (prediction / inducing cross-kernel, not the self
Gram path that delegates to tdot).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import GPy  # noqa: E402

from peakutil import peak_bytes  # noqa: E402


def main():
    rng = np.random.RandomState(1)
    kern = GPy.kern.RBF(8, variance=1.3, lengthscale=0.8, ARD=False)
    x_train = np.ascontiguousarray(rng.randn(1800, 8))
    x_test = np.ascontiguousarray(rng.randn(1800, 8))
    holder = {}

    def run():
        holder["out"] = kern._unscaled_dist(x_train, x_test)

    peak = peak_bytes(run)
    out = holder["out"]
    if out.shape != (1800, 1800) or not np.isfinite(out).all() or np.any(out < 0):
        raise SystemExit("Stationary._unscaled_dist returned a bad distance matrix")
    print(f"peak_memory_bytes={peak}")


if __name__ == "__main__":
    main()
