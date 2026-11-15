"""Shared peak-memory helper for intake harnesses.

tracemalloc is started after inputs exist, so the reported peak is bytes
allocated inside the timed call (NumPy buffers included).
"""

import gc
import tracemalloc


def peak_bytes(fn):
    gc.collect()
    tracemalloc.start()
    try:
        fn()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return int(peak)
