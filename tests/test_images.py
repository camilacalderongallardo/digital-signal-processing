import numpy as np

from dsp.images import block_dct_compress, convolve2d, haar_dwt2, haar_idwt2
from dsp.metrics import psnr


def test_identity_convolution():
    image = np.arange(25).reshape(5, 5)
    kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    assert np.allclose(convolve2d(image, kernel), image)


def test_haar_round_trip():
    rng = np.random.default_rng(3)
    image = rng.standard_normal((16, 16))
    reconstructed = haar_idwt2(haar_dwt2(image))
    assert np.allclose(image, reconstructed, atol=1e-10)


def test_dct_quality_tradeoff():
    image = np.tile(np.linspace(0, 255, 64), (64, 1))
    low, _, low_ratio = block_dct_compress(image, quality=20)
    high, _, high_ratio = block_dct_compress(image, quality=85)

    assert psnr(image, high) >= psnr(image, low)
    assert high_ratio >= low_ratio
