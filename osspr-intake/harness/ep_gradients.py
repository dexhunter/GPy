"""Time Likelihood.ep_gradients in the Gauss-Hermite mode EP actually uses.

Student-T does not override ep_gradients, so expectation propagation calls
this base method once per hyperparameter step. quad_mode='gh' is the mode
in EP._inference.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from GPy.likelihoods import StudentT


def build_workload():
    rng = np.random.default_rng(0)
    n = 800
    y = np.ascontiguousarray(rng.standard_normal((n, 1)))
    tau = np.full((n, 1), 1.35)
    v = np.ascontiguousarray(rng.standard_normal((n, 1)) * 0.25)
    dL_dKdiag = np.ones((n, 1))
    likelihood = StudentT()
    return likelihood, y, tau, v, dL_dKdiag


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
    likelihood, y, tau, v, dL_dKdiag = build_workload()

    def call():
        return likelihood.ep_gradients(y, tau, v, dL_dKdiag, quad_mode="gh")

    gradients, times_ms = measure(call)
    checksum = float(np.sum(gradients))
    for elapsed_ms in times_ms:
        print(f"call_ms={elapsed_ms:.6f}")
    print(f"checksum={checksum:.16e}")


if __name__ == "__main__":
    main()
