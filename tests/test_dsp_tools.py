import numpy as np
import pytest

from src.dsp_tools import (
    apply_fir_filter,
    design_lowpass_fir,
    divide_conquer_dft,
    dft_sum,
    generate_test_signal,
    single_sided_spectrum,
)


def test_dft_sum_matches_numpy_fft():
    x = np.array([1.0, 2.0, -1.0, 0.5])
    assert np.allclose(dft_sum(x), np.fft.fft(x), atol=1e-10)


def test_divide_conquer_dft_matches_numpy_fft():
    x = np.arange(8, dtype=float)
    assert np.allclose(divide_conquer_dft(x), np.fft.fft(x), atol=1e-10)


def test_divide_conquer_rejects_odd_length():
    with pytest.raises(ValueError):
        divide_conquer_dft(np.arange(5))


def test_filter_reduces_high_frequency_interference():
    sample_rate = 2000
    _, _, measured = generate_test_signal(
        sample_rate=sample_rate,
        noise_level=0,
    )
    taps = design_lowpass_fir(sample_rate, 250, number_of_taps=81)
    filtered = apply_fir_filter(measured, taps)

    frequency_before, amplitude_before = single_sided_spectrum(
        measured,
        sample_rate,
    )
    frequency_after, amplitude_after = single_sided_spectrum(
        filtered,
        sample_rate,
    )

    index_420_before = np.argmin(np.abs(frequency_before - 420))
    index_420_after = np.argmin(np.abs(frequency_after - 420))

    assert amplitude_after[index_420_after] < 0.15 * amplitude_before[index_420_before]
