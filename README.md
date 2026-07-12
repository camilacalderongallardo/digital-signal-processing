# Digital Signal Processing with Python

DSP portfolio project that demonstrates:

- generation of a noisy multi-tone signal
- direct DFT computation
- divide-and-conquer DFT computation
- FFT-based spectrum analysis
- FIR low-pass filter design
- filtering and before/after spectrum comparison

## Run the project

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
pip install -r requirements.txt
python run_demo.py
```

The demo saves these figures in `outputs/`:

- `time_domain_signal.png`
- `spectrum_before_after.png`
- `filter_response.png`

## Run the tests

```bash
python -m pytest
```

## Main engineering result

The generated signal contains a desired 120 Hz component and unwanted higher-frequency
components. A windowed FIR low-pass filter is designed and applied to retain the desired
component while reducing the higher-frequency content.

## Skills demonstrated

Python, NumPy, SciPy, Matplotlib, DFT, FFT, spectral analysis, FIR filtering,
frequency response analysis, numerical validation, and unit testing.

## Project origin

This portfolio project was independently reorganized from DSP concepts practiced in
electrical engineering university coursework. It uses new synthetic inputs, new documentation, and a standalone
software structure. 
