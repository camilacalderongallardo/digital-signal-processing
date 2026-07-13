"""FIR/IIR design, filtering, and frequency-response analysis."""

from __future__ import annotations
import numpy as np
from scipy import signal


def design_fir_lowpass(sample_rate, passband_hz, stopband_hz, attenuation_db=60):
    """
    Design a Kaiser-window FIR low-pass filter from transition specifications.
    """
    if not 0 < passband_hz < stopband_hz < sample_rate / 2:
        raise ValueError("Frequencies must satisfy 0 < passband < stopband < Nyquist")

    transition = (stopband_hz - passband_hz) / (sample_rate / 2)
    num_taps, beta = signal.kaiserord(attenuation_db, transition)

    if num_taps % 2 == 0:
        num_taps += 1

    cutoff = (passband_hz + stopband_hz) / 2
    taps = signal.firwin(
        num_taps,
        cutoff,
        window=("kaiser", beta),
        fs=sample_rate,
    )

    return taps


def design_iir_butterworth(sample_rate, passband_hz, stopband_hz,
                           passband_ripple_db=1, stopband_attenuation_db=50):
    """
    Design a minimum-order digital Butterworth low-pass filter from specifications.
    """
    order, wn = signal.buttord(
        passband_hz,
        stopband_hz,
        passband_ripple_db,
        stopband_attenuation_db,
        fs=sample_rate,
    )
    sos = signal.butter(order, wn, btype="low", fs=sample_rate, output="sos")
    return order, sos


def apply_filter(x, coefficients, mode="fir"):
    """Apply either FIR coefficients or IIR second-order sections."""
    x = np.asarray(x)

    if mode == "fir":
        return signal.lfilter(coefficients, [1.0], x)

    if mode == "iir":
        return signal.sosfilt(coefficients, x)

    raise ValueError("mode must be 'fir' or 'iir'")


def frequency_response(coefficients, sample_rate, mode="fir", points=4096):
    """Compute magnitude, phase, and group delay."""
    if mode == "fir":
        frequency, response = signal.freqz(
            coefficients, [1.0], worN=points, fs=sample_rate
        )
        _, group_delay = signal.group_delay(
            (coefficients, [1.0]), w=points, fs=sample_rate
        )
    elif mode == "iir":
        frequency, response = signal.sosfreqz(
            coefficients, worN=points, fs=sample_rate
        )
        b, a = signal.sos2tf(coefficients)
        _, group_delay = signal.group_delay(
            (b, a), w=points, fs=sample_rate
        )
    else:
        raise ValueError("mode must be 'fir' or 'iir'")

    phase = np.unwrap(np.angle(response))
    return frequency, response, phase, group_delay
