from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dsp.filters import (
    apply_filter,
    design_fir_lowpass,
    design_iir_butterworth,
    frequency_response,
)
from dsp.metrics import rmse
from dsp.sampling import sample_signal, sinc_reconstruct


def analog_signal(t):
    return np.sin(2 * np.pi * 120 * t) + 0.35 * np.sin(2 * np.pi * 420 * t)


def main():
    output = Path("outputs")
    output.mkdir(exist_ok=True)

    duration = 0.05
    reference_time = np.linspace(0, duration, 4000)
    reference = analog_signal(reference_time)

    sample_rates = [600, 1200]
    plt.figure(figsize=(10, 5))

    for sample_rate in sample_rates:
        sample_time, samples = sample_signal(analog_signal, sample_rate, duration)
        reconstructed = sinc_reconstruct(
            sample_time, samples, reference_time, sample_rate
        )
        reconstruction_error = rmse(reference, reconstructed)
        plt.plot(
            reference_time,
            reconstructed,
            label=f"{sample_rate} Hz sampling, RMSE={reconstruction_error:.3f}",
        )

    plt.plot(reference_time, reference, "k--", linewidth=2, label="Reference")
    plt.xlim(0, 0.025)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title("Sampling, aliasing, and sinc reconstruction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "sampling_reconstruction.png", dpi=180)
    plt.close()

    sample_rate = 2000
    t = np.arange(0, 1, 1 / sample_rate)
    rng = np.random.default_rng(10)

    desired = np.sin(2 * np.pi * 120 * t)
    measured = (
        desired
        + 0.55 * np.sin(2 * np.pi * 420 * t)
        + 0.25 * np.sin(2 * np.pi * 650 * t)
        + 0.15 * rng.standard_normal(len(t))
    )

    fir = design_fir_lowpass(sample_rate, 180, 300, attenuation_db=60)
    iir_order, iir = design_iir_butterworth(
        sample_rate, 180, 300,
        passband_ripple_db=1,
        stopband_attenuation_db=50,
    )

    fir_output = apply_filter(measured, fir, mode="fir")
    iir_output = apply_filter(measured, iir, mode="iir")

    fir_frequency, fir_response, fir_phase, fir_delay = frequency_response(
        fir, sample_rate, mode="fir"
    )
    iir_frequency, iir_response, iir_phase, iir_delay = frequency_response(
        iir, sample_rate, mode="iir"
    )

    plt.figure(figsize=(10, 5))
    plt.plot(
        fir_frequency,
        20 * np.log10(np.maximum(np.abs(fir_response), 1e-8)),
        label=f"FIR, {len(fir)} taps",
    )
    plt.plot(
        iir_frequency,
        20 * np.log10(np.maximum(np.abs(iir_response), 1e-8)),
        label=f"Butterworth IIR, order {iir_order}",
    )
    plt.axvspan(0, 180, alpha=0.12, label="Passband")
    plt.axvspan(300, sample_rate / 2, alpha=0.08, label="Stopband")
    plt.ylim(-110, 5)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("FIR and IIR designs from engineering specifications")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "filter_magnitude_comparison.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(fir_frequency, fir_delay, label="FIR group delay")
    plt.plot(iir_frequency, iir_delay, label="IIR group delay")
    plt.xlim(0, 500)
    plt.ylim(-10, max(10, np.nanpercentile(fir_delay, 95) * 1.2))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Group delay (samples)")
    plt.title("Phase behavior and group-delay tradeoff")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "group_delay_comparison.png", dpi=180)
    plt.close()

    valid = slice(len(fir), -len(fir))
    print(f"FIR RMSE against desired component: {rmse(desired[valid], fir_output[valid]):.4f}")
    print(f"IIR RMSE against desired component: {rmse(desired[valid], iir_output[valid]):.4f}")
    print("Saved sampling and filter-design figures.")


if __name__ == "__main__":
    main()
