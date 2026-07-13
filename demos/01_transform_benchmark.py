from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from dsp.transforms import dft_matrix, dft_sum, idft_sum, radix2_fft


def main():
    output = Path("outputs")
    output.mkdir(exist_ok=True)

    rng = np.random.default_rng(4)
    sizes = [8, 16, 32, 64, 128, 256]
    direct_times = []
    matrix_times = []
    radix2_times = []
    numpy_times = []

    for N in sizes:
        x = rng.standard_normal(N)

        start = perf_counter()
        X_direct = dft_sum(x)
        direct_times.append(perf_counter() - start)

        start = perf_counter()
        X_matrix = dft_matrix(N) @ x
        matrix_times.append(perf_counter() - start)

        start = perf_counter()
        X_radix2 = radix2_fft(x)
        radix2_times.append(perf_counter() - start)

        start = perf_counter()
        X_numpy = np.fft.fft(x)
        numpy_times.append(perf_counter() - start)

        assert np.allclose(X_direct, X_numpy, atol=1e-9)
        assert np.allclose(X_matrix, X_numpy, atol=1e-9)
        assert np.allclose(X_radix2, X_numpy, atol=1e-9)
        assert np.allclose(idft_sum(X_direct), x, atol=1e-9)

    plt.figure(figsize=(9, 5))
    plt.loglog(sizes, direct_times, "o-", label="Direct DFT")
    plt.loglog(sizes, matrix_times, "o-", label="Matrix DFT")
    plt.loglog(sizes, radix2_times, "o-", label="Recursive radix-2 FFT")
    plt.loglog(sizes, numpy_times, "o-", label="NumPy FFT")
    plt.xlabel("Transform length N")
    plt.ylabel("Execution time (seconds)")
    plt.title("DFT and FFT implementation benchmark")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "transform_benchmark.png", dpi=180)
    plt.close()

    print("Transform algorithms match NumPy FFT.")
    print("Saved outputs/transform_benchmark.png")


if __name__ == "__main__":
    main()
