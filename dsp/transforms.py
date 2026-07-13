"""Transform algorithms implemented for inspection and numerical comparison."""

from __future__ import annotations
import numpy as np


def dft_sum(x):
    """Compute the DFT directly from its summation definition."""
    x = np.asarray(x, dtype=complex)
    N = len(x)
    X = np.zeros(N, dtype=complex)

    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-1j * 2 * np.pi * k * n / N)

    return X


def idft_sum(X):
    """Compute the inverse DFT directly from its summation definition."""
    X = np.asarray(X, dtype=complex)
    N = len(X)
    x = np.zeros(N, dtype=complex)

    for n in range(N):
        for k in range(N):
            x[n] += X[k] * np.exp(1j * 2 * np.pi * k * n / N)

    return x / N


def dft_matrix(N):
    """Return the NxN complex DFT matrix."""
    n = np.arange(N)
    k = n.reshape((N, 1))
    return np.exp(-2j * np.pi * k * n / N)


def radix2_fft(x):
    """
    Recursive radix-2 decimation-in-time FFT.

    The input length must be a power of two.
    """
    x = np.asarray(x, dtype=complex)
    N = len(x)

    if N == 0 or N & (N - 1):
        raise ValueError("Input length must be a nonzero power of two")

    if N == 1:
        return x.copy()

    X_even = radix2_fft(x[::2])
    X_odd = radix2_fft(x[1::2])

    twiddle = np.exp(-2j * np.pi * np.arange(N // 2) / N)
    top = X_even + twiddle * X_odd
    bottom = X_even - twiddle * X_odd

    return np.concatenate((top, bottom))


def dtft(x, omega, n0=0):
    """Evaluate the DTFT of a finite sequence at arbitrary frequencies."""
    x = np.asarray(x, dtype=complex)
    omega = np.asarray(omega, dtype=float)
    n = np.arange(len(x)) + n0

    X = np.zeros(len(omega), dtype=complex)
    for i in range(len(omega)):
        X[i] = np.sum(x * np.exp(-1j * omega[i] * n))

    return X
