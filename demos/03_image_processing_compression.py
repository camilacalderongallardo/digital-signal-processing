from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dsp.images import (
    block_dct_compress,
    convolve2d,
    haar_dwt2,
    haar_idwt2,
)
from dsp.metrics import psnr


def create_test_image(size=256):
    """Create a synthetic image containing edges, gradients, and texture."""
    y, x = np.mgrid[0:size, 0:size]
    image = 40 + 0.55 * x + 0.25 * y

    circle = (x - 75) ** 2 + (y - 85) ** 2 < 38 ** 2
    image[circle] = 225

    image[145:220, 135:225] = 165
    texture = 22 * np.sin(2 * np.pi * x / 13) * np.sin(2 * np.pi * y / 17)
    image[145:220, 135:225] += texture[145:220, 135:225]

    return np.clip(image, 0, 255)


def main():
    output = Path("outputs")
    output.mkdir(exist_ok=True)

    image = create_test_image()

    smoothing_kernel = np.ones((5, 5)) / 25
    laplacian_kernel = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0],
    ])

    smoothed = convolve2d(image, smoothing_kernel)
    edges = convolve2d(image, laplacian_kernel)
    sharpened = np.clip(image + 0.8 * edges, 0, 255)

    ll, lh, hl, hh = haar_dwt2(image)
    threshold = 18
    thresholded = (
        ll,
        np.where(np.abs(lh) >= threshold, lh, 0),
        np.where(np.abs(hl) >= threshold, hl, 0),
        np.where(np.abs(hh) >= threshold, hh, 0),
    )
    wavelet_reconstruction = np.clip(haar_idwt2(thresholded), 0, 255)

    qualities = [20, 50, 85]
    reconstructions = []

    for quality in qualities:
        reconstructed, coefficients, nonzero_ratio = block_dct_compress(
            image, quality=quality
        )
        reconstructions.append((quality, reconstructed, nonzero_ratio))

    plt.figure(figsize=(12, 8))
    items = [
        ("Original", image),
        ("5x5 smoothing", smoothed),
        ("Laplacian edges", np.abs(edges)),
        ("Sharpened", sharpened),
        ("Haar LL subband", ll),
        (
            f"Wavelet thresholding\nPSNR={psnr(image, wavelet_reconstruction):.1f} dB",
            wavelet_reconstruction,
        ),
    ]

    for index, (title, display_image) in enumerate(items, start=1):
        plt.subplot(2, 3, index)
        plt.imshow(display_image, cmap="gray", vmin=0, vmax=255)
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output / "image_filtering_wavelets.png", dpi=180)
    plt.close()

    plt.figure(figsize=(13, 4))
    plt.subplot(1, 4, 1)
    plt.imshow(image, cmap="gray", vmin=0, vmax=255)
    plt.title("Original")
    plt.axis("off")

    for index, (quality, reconstructed, nonzero_ratio) in enumerate(
        reconstructions, start=2
    ):
        plt.subplot(1, 4, index)
        plt.imshow(reconstructed, cmap="gray", vmin=0, vmax=255)
        plt.title(
            f"Quality {quality}\n"
            f"PSNR={psnr(image, reconstructed):.1f} dB\n"
            f"Nonzero={100*nonzero_ratio:.1f}%"
        )
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output / "dct_compression_tradeoff.png", dpi=180)
    plt.close()

    print("Image processing and compression pipeline completed.")
    print("Saved outputs/image_filtering_wavelets.png")
    print("Saved outputs/dct_compression_tradeoff.png")


if __name__ == "__main__":
    main()
