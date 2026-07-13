import numpy as np

from dsp.transforms import dft_matrix, dft_sum, idft_sum, radix2_fft


def test_transform_implementations_match_numpy():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(16)
    expected = np.fft.fft(x)

    assert np.allclose(dft_sum(x), expected, atol=1e-10)
    assert np.allclose(dft_matrix(len(x)) @ x, expected, atol=1e-10)
    assert np.allclose(radix2_fft(x), expected, atol=1e-10)
    assert np.allclose(idft_sum(expected), x, atol=1e-10)
