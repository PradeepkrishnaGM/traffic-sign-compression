"""Model size and CPU latency helpers."""
import gzip
import time
import os

import numpy as np
import torch

from . import config as C
from . import data, onnx_utils


def file_size_mb(path):
    return os.path.getsize(path) / 1e6


def gzip_size_mb(path):
    """Size after gzip: shows what sparsity buys for storage (zeros compress well)."""
    with open(path, "rb") as f:
        raw = f.read()
    return len(gzip.compress(raw, compresslevel=6)) / 1e6


def _summarise(times_s):
    ms = np.array(times_s) * 1000.0
    return float(np.median(ms)), float(np.percentile(ms, 90))


def ort_latency(path, sample_uint8):
    """ONNX Runtime CPU latency for one image: median and p90 in ms.

    Batch 1 and a single thread imitate a small embedded core; the median is used
    because a few slow outliers (OS interrupts) would distort a mean.
    """
    session = onnx_utils.make_session(path, C.LAT_THREADS)
    x = data.to_tensor(sample_uint8[:1]).numpy()
    for _ in range(C.LAT_WARMUP):
        session.run(None, {"input": x})
    times = []
    for _ in range(C.LAT_RUNS):
        t0 = time.perf_counter()
        session.run(None, {"input": x})
        times.append(time.perf_counter() - t0)
    return _summarise(times)


def torch_latency(model, sample_uint8):
    """PyTorch FP32 CPU latency (batch 1, one thread), for reference only."""
    torch.set_num_threads(C.LAT_THREADS)
    model.eval()
    x = data.to_tensor(sample_uint8[:1])
    times = []
    with torch.no_grad():
        for _ in range(C.LAT_WARMUP):
            model(x)
        for _ in range(C.LAT_RUNS):
            t0 = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - t0)
    return _summarise(times)
