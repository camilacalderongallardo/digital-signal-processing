"""Objective reconstruction and compression metrics."""

import numpy as np


def mse(reference, estimate):
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    return float(np.mean((reference - estimate) ** 2))


def rmse(reference, estimate):
    return float(np.sqrt(mse(reference, estimate)))


def psnr(reference, estimate, peak=255.0):
    error = mse(reference, estimate)
    if error == 0:
        return float("inf")
    return float(10 * np.log10((peak ** 2) / error))
