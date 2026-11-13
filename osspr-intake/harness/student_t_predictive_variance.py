"""Time StudentT.predictive_variance on a diagonal prediction.

GP.predict with a Student-T likelihood and full_cov=False reaches this method
through Likelihood.predictive_values. The mean is computed first, as predict
does, and is not part of the timed call.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from GPy.likelihoods import StudentT


def build_workload():
    rng = np.random.default_rng(0)
    # Enough test points that one call stays above the 10 ms floor.
    n = 32
    mu = np.ascontiguousarray(rng.standard_normal((n, 1)))
    var = np.ascontiguousarray(np.abs(rng.standard_normal((n, 1))) + 0.2)
    likelihood = StudentT(deg_free=5, sigma2=0.7)
    mean = likelihood.predictive_mean(mu, var)
    return likelihood, mu, var, mean


def measure(call, repeats=7, warmup=2):
    for _ in range(warmup):
        result = call()
    times_ms = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = call()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return result, times_ms


def main():
    likelihood, mu, var, mean = build_workload()
    variance, times_ms = measure(lambda: likelihood.predictive_variance(mu, var, mean))
    checksum = float(variance[0, 0] + variance[-1, 0])
    for elapsed_ms in times_ms:
        print(f"call_ms={elapsed_ms:.6f}")
    print(f"checksum={checksum:.16e}")


if __name__ == "__main__":
    main()
