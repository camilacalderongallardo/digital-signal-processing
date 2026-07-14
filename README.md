# Advanced Digital Signal Processing with Python

A mini electrical engineering project demonstrating DSP
algorithms from course mathematical definitions through implementation, validation,
engineering tradeoff analysis, and application to 1D and 2D signal images.

The repository implements and evaluates:

### Discrete transforms and computational structure

- Direct DFT and inverse DFT from the summation definitions
- Matrix-based DFT
- Recursive radix-2 decimation-in-time FFT
- Numerical validation against NumPy
- Runtime comparison illustrating the difference between DFT
  methods and FFT structure

### Sampling and reconstruction

- Uniform sampling of continuous-time test signals
- Aliasing caused by sampling below the Nyquist rate
- Truncated ideal sinc reconstruction
- Quantitative reconstruction error using RMSE

### Digital filter design

- Kaiser-window FIR design from passband, stopband, and attenuation specifications
- Minimum-order Butterworth IIR design
- Magnitude-response comparison
- Phase and group-delay analysis
- FIR/IIR implementation tradeoffs

### Two-dimensional DSP

- Explicit 2-D convolution
- Image smoothing, edge detection, and sharpening
- One-level separable Haar wavelet transform and inverse transform
- Wavelet coefficient thresholding
- 8x8 block DCT compression with JPEG-style quantization
- PSNR and retained-coefficient measurements
- 

## Run the complete project

python -m venv .venv

Activate the environment and run:

pip install -r requirements.txt
python run_all.py
python -m pytest

## Generated results

- transform_benchmark.png: direct, matrix, radix-2, and library FFT timing
- sampling_reconstruction.png: sampling-rate and aliasing comparison
- filter_magnitude_comparison.png: FIR and IIR magnitude specifications
- group_delay_comparison.png: phase-related implementation tradeoff
- image_filtering_wavelets.png: spatial filtering and Haar decomposition
- dct_compression_tradeoff.png: image quality versus coefficient retention

## Engineering conclusions

- The FFT obtains the same transform as the direct DFT with better
  computational scaling. (this is expected)
- Sampling below twice the highest signal frequency creates irreversible aliasing.
- FIR filters provide linear-phase behavior at the cost of higher order and delay.
- IIR filters meet similar magnitude specifications with lower order but nonlinear
  phase and frequency-dependent group delay.
- Transform-domain image compression trades coefficient sparsity against objective
  reconstruction quality.
- Wavelet and DCT representations emphasize different spatial and frequency
  characteristics of an image.

## Skills

Digital signal processing, Fourier analysis, FFT algorithms, sampling theory,
sinc interpolation, FIR and IIR design, group-delay analysis, two-dimensional
convolution, Haar wavelets, block DCT, quantization, PSNR, NumPy, SciPy,
Matplotlib, numerical validation, modular Python, and unit testing.

## Project origin

This is an independently organized portfolio implementation based on concepts
studied in my undergraduate dsp electrical engineering class. It uses new
experiments, synthetic inputs, original documentation, reusable modules, and
independent validation. It does not include assignment prompts, grading
materials, or instructor-provided solution files.
