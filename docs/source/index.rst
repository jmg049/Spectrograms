Spectrograms Documentation
==========================

**Spectrograms** is a fast Python library for FFT-based computations on audio (1D) and image (2D) data, powered by Rust.

Features
--------

**Audio Processing:**

- **Multiple frequency scales**: Linear, Mel, ERB, LogHz, and CQT
- **Flexible amplitude scaling**: Power, magnitude, and decibels
- **Audio features**: MFCC, chromagrams, and raw STFT
- **Streaming support**: Frame-by-frame processing for real-time applications

**Image Processing:**

- **2D FFT operations**: Fast 2D Fourier transforms
- **Spatial filtering**: Low-pass, high-pass, and band-pass filters
- **Convolution**: FFT-based convolution (faster for large kernels)
- **Edge detection**: Frequency-domain edge emphasis
- **Image enhancement**: Sharpening and feature enhancement

**Performance:**

- **High performance**: Rust implementation with FFTW support
- **Batch processing**: Reusable plans for efficient processing (1.5-3x speedup)
- **GIL release**: All functions release Python GIL for parallel processing
- **Type-safe**: Full type stubs for IDE support

Quick Examples
--------------

Audio Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import spectrograms as sg

   # Generate a sine wave
   sr = 16000
   t = np.linspace(0, 1, sr, dtype=np.float64)
   samples = np.sin(2 * np.pi * 440 * t)

   # Compute mel spectrogram
   stft = sg.StftParams(n_fft=512, hop_size=256, window=sg.WindowType.hanning)
   params = sg.SpectrogramParams(stft, sample_rate=sr)
   mel_params = sg.MelParams(n_mels=80, f_min=0.0, f_max=8000.0)

   spec = sg.compute_mel_power_spectrogram(samples, params, mel_params)
   print(f"Shape: {spec.shape}")

Image Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import spectrograms as sg

   # Create or load an image
   image = np.random.randn(256, 256)

   # Compute 2D FFT
   spectrum = sg.fft2d(image)

   # Apply Gaussian blur
   kernel = sg.gaussian_kernel_2d(9, 2.0)
   blurred = sg.convolve_fft(image, kernel)

   # Detect edges
   edges = sg.highpass_filter(image, cutoff=0.1)

Installation
------------

.. code-block:: bash

   pip install spectrograms

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guide/installation
   guide/quickstart
   guide/choosing_parameters
   guide/frequency_scales
   guide/planner_guide
   guide/audio_features
   guide/image_processing
   guide/performance
   guide/examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


