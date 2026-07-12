from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from src.dsp_tools import (
    apply_fir_filter,
    design_lowpass_fir,
    divide_conquer_dft,
    dft_sum,
    frequency_response,
    generate_test_signal,
    single_sided_spectrum,
)


def main():
    output_directory = Path("outputs")
    output_directory.mkdir(exist_ok=True)

    sample_rate = 2000
    cutoff_frequency = 250

    t, clean, measured = generate_test_signal(sample_rate=sample_rate)
    taps = design_lowpass_fir(
        sample_rate=sample_rate,
        cutoff_frequency=cutoff_frequency,
        number_of_taps=81,
    )
    filtered = apply_fir_filter(measured, taps)

    frequency_before, amplitude_before = single_sided_spectrum(
        measured,
        sample_rate,
    )
    frequency_after, amplitude_after = single_sided_spectrum(
        filtered,
        sample_rate,
    )

    plt.figure(figsize=(10, 5))
    shown = t <= 0.05
    plt.plot(t[shown], measured[shown], label="Measured signal")
    plt.plot(t[shown], clean[shown], label="Desired component", linewidth=2)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title("Synthetic sensor signal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_directory / "time_domain_signal.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.semilogy(
        frequency_before,
        np.maximum(amplitude_before, 1e-8),
        label="Before filtering",
    )
    plt.semilogy(
        frequency_after,
        np.maximum(amplitude_after, 1e-8),
        label="After filtering",
    )
    plt.xlim(0, 800)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Spectrum before and after FIR filtering")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_directory / "spectrum_before_after.png", dpi=160)
    plt.close()

    response_frequency, response = frequency_response(taps, sample_rate)

    plt.figure(figsize=(10, 5))
    response_db = 20 * np.log10(np.maximum(np.abs(response), 1e-8))
    plt.plot(response_frequency, response_db)
    plt.axvline(cutoff_frequency, linestyle="--", label="Cutoff frequency")
    plt.ylim(-100, 5)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Designed FIR low-pass filter")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_directory / "filter_response.png", dpi=160)
    plt.close()

    small_signal = measured[:32]
    direct_result = dft_sum(small_signal)
    split_result = divide_conquer_dft(small_signal)
    numpy_result = np.fft.fft(small_signal)

    print("Maximum DFT error:")
    print("direct vs NumPy =", np.max(np.abs(direct_result - numpy_result)))
    print("split vs NumPy  =", np.max(np.abs(split_result - numpy_result)))
    print()
    print("Saved figures to:", output_directory.resolve())


if __name__ == "__main__":
    main()
