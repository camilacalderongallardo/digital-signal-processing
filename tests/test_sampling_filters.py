import numpy as np

from dsp.filters import design_fir_lowpass, design_iir_butterworth
from dsp.sampling import sample_signal, sinc_reconstruct


def test_sinc_reconstruction_of_low_frequency_tone():
    frequency = 40
    sample_rate = 500
    duration = 0.2

    function = lambda t: np.sin(2 * np.pi * frequency * t)
    sample_time, samples = sample_signal(function, sample_rate, duration)
    test_time = np.linspace(0.02, 0.18, 500)
    reconstructed = sinc_reconstruct(
        sample_time, samples, test_time, sample_rate
    )

    assert np.sqrt(np.mean((function(test_time) - reconstructed) ** 2)) < 0.03


def test_filter_design_returns_valid_filters():
    fir = design_fir_lowpass(2000, 180, 300, 60)
    order, sos = design_iir_butterworth(2000, 180, 300, 1, 50)

    assert len(fir) > 10
    assert order >= 1
    assert sos.ndim == 2
