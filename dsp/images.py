"""Two-dimensional filtering, Haar wavelets, and block-DCT compression."""

from __future__ import annotations
import numpy as np
from scipy.fft import dctn, idctn


def convolve2d(image, kernel):
    """Perform zero-padded 2-D convolution using explicit loops."""
    image = np.asarray(image, dtype=float)
    kernel = np.asarray(kernel, dtype=float)

    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(image, ((ph, ph), (pw, pw)), mode="constant")
    output = np.zeros_like(image, dtype=float)
    flipped = np.flipud(np.fliplr(kernel))

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            region = padded[row:row + kh, col:col + kw]
            output[row, col] = np.sum(region * flipped)

    return output


def haar_dwt2(image):
    """Compute a one-level separable orthonormal Haar wavelet transform."""
    image = np.asarray(image, dtype=float)

    if image.shape[0] % 2 or image.shape[1] % 2:
        raise ValueError("Image dimensions must be even")

    low_rows = (image[:, 0::2] + image[:, 1::2]) / np.sqrt(2)
    high_rows = (image[:, 0::2] - image[:, 1::2]) / np.sqrt(2)

    ll = (low_rows[0::2, :] + low_rows[1::2, :]) / np.sqrt(2)
    lh = (low_rows[0::2, :] - low_rows[1::2, :]) / np.sqrt(2)
    hl = (high_rows[0::2, :] + high_rows[1::2, :]) / np.sqrt(2)
    hh = (high_rows[0::2, :] - high_rows[1::2, :]) / np.sqrt(2)

    return ll, lh, hl, hh


def haar_idwt2(coefficients):
    """Reconstruct an image from one-level Haar wavelet coefficients."""
    ll, lh, hl, hh = coefficients

    low_rows = np.zeros((ll.shape[0] * 2, ll.shape[1]))
    high_rows = np.zeros_like(low_rows)

    low_rows[0::2, :] = (ll + lh) / np.sqrt(2)
    low_rows[1::2, :] = (ll - lh) / np.sqrt(2)
    high_rows[0::2, :] = (hl + hh) / np.sqrt(2)
    high_rows[1::2, :] = (hl - hh) / np.sqrt(2)

    image = np.zeros((low_rows.shape[0], low_rows.shape[1] * 2))
    image[:, 0::2] = (low_rows + high_rows) / np.sqrt(2)
    image[:, 1::2] = (low_rows - high_rows) / np.sqrt(2)

    return image


def block_dct_compress(image, quality=50, block_size=8):
    """
    Compress a grayscale image with block DCT and scalar quantization.

    Returns the reconstruction, quantized coefficients, and retained nonzero ratio.
    """
    image = np.asarray(image, dtype=float)

    if image.shape[0] % block_size or image.shape[1] % block_size:
        raise ValueError("Image dimensions must be multiples of block_size")

    quality = int(np.clip(quality, 1, 100))
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality

    base_quantization = np.array([
        [16,11,10,16,24,40,51,61],
        [12,12,14,19,26,58,60,55],
        [14,13,16,24,40,57,69,56],
        [14,17,22,29,51,87,80,62],
        [18,22,37,56,68,109,103,77],
        [24,35,55,64,81,104,113,92],
        [49,64,78,87,103,121,120,101],
        [72,92,95,98,112,100,103,99],
    ], dtype=float)

    q = np.floor((base_quantization * scale + 50) / 100)
    q[q < 1] = 1

    shifted = image - 128
    reconstruction = np.zeros_like(image)
    quantized = np.zeros_like(image)

    for row in range(0, image.shape[0], block_size):
        for col in range(0, image.shape[1], block_size):
            block = shifted[row:row + block_size, col:col + block_size]
            transformed = dctn(block, norm="ortho")
            encoded = np.round(transformed / q)
            decoded = idctn(encoded * q, norm="ortho")

            quantized[row:row + block_size, col:col + block_size] = encoded
            reconstruction[row:row + block_size, col:col + block_size] = decoded + 128

    reconstruction = np.clip(reconstruction, 0, 255)
    nonzero_ratio = np.count_nonzero(quantized) / quantized.size

    return reconstruction, quantized, nonzero_ratio
