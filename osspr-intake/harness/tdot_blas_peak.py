"""Peak memory of GPy.util.linalg.tdot_blas on a realistic Gram matrix.

Workload: 1800 x 8 design matrix, the same shape linear kernels and
stationary self-distances pass into tdot (exact GP, N about 1800).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from GPy.util.linalg import tdot_blas  # noqa: E402

from peakutil import peak_bytes  # noqa: E402


def main():
    rng = np.random.RandomState(0)
    mat = np.ascontiguousarray(rng.randn(1800, 8))
    holder = {}

    def run():
        holder["out"] = tdot_blas(mat)

    peak = peak_bytes(run)
    out = holder["out"]
    if out.shape != (1800, 1800) or not np.isfinite(out).all():
        raise SystemExit("tdot_blas returned a bad Gram matrix")
    print(f"peak_memory_bytes={peak}")


if __name__ == "__main__":
    main()
