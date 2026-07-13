"""Sampling, aliasing, and ideal sinc reconstruction utilities."""

from __future__ import annotations
import numpy as np


def sample_signal(function, sample_rate, duration):
    """Sample a continuous-time function on a uniform grid."""
    if sample_rate <= 0 or duration <= 0:
        raise ValueError("sample_rate and duration must be positive")

    t = np.arange(0, duration, 1 / sample_rate)
    return t, function(t)


def sinc_reconstruct(sample_times, samples, reconstruction_times, sample_rate):
    """
    Reconstruct a bandlimited signal with truncated ideal sinc interpolation.
    """
    sample_times = np.asarray(sample_times)
    samples = np.asarray(samples)
    reconstruction_times = np.asarray(reconstruction_times)

    result = np.zeros_like(reconstruction_times, dtype=float)

    for n in range(len(samples)):
        result += samples[n] * np.sinc(
            sample_rate * (reconstruction_times - sample_times[n])
        )

    return result
