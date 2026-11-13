"""Time Stationary._unscaled_dist on a symmetric training covariance.

RBF, Matern, and the other stationary kernels call this with X2=None while
building K(X) on every hyperparameter step. The harness times that path.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from GPy.kern import RBF


def build_workload():
    rng = np.random.default_rng(0)
    # N by D inputs, the shape used when a dense GP fits a stationary kernel.
    x = np.ascontiguousarray(rng.standard_normal((2000, 8)))
    kernel = RBF(x.shape[1], variance=0.9, lengthscale=1.2)
    return kernel, x


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
    kernel, x = build_workload()
    distance, times_ms = measure(lambda: kernel._unscaled_dist(x))
    # Fingerprint is outside the timer and keeps the distance live.
    checksum = float(distance[0, 1] + distance[-1, -2])
    for elapsed_ms in times_ms:
        print(f"call_ms={elapsed_ms:.6f}")
    print(f"checksum={checksum:.16e}")


if __name__ == "__main__":
    main()
