Spectrograms Documentation
==========================

**Spectrograms** is a fast Python library for computing spectrograms and audio features, powered by Rust.

Features
--------

- **Multiple frequency scales**: Linear, Mel, ERB, and CQT
- **Flexible amplitude scaling**: Power, magnitude, and decibels
- **High performance**: Rust implementation with FFTW support
- **Batch processing**: Reusable plans for efficient processing
- **Audio features**: MFCC, chromagrams, and CQT
- **Type-safe**: Full type stubs for IDE support

Quick Example
-------------

.. code-block:: python

   import numpy as np
   import spectrograms as sg

   # Generate a sine wave
   sr = 16000
   t = np.linspace(0, 1, sr, dtype=np.float64)
   samples = np.sin(2 * np.pi * 440 * t)

   # Compute mel spectrogram
   stft = sg.StftParams(n_fft=512, hop_size=256, window="hanning")
   params = sg.SpectrogramParams(stft, sample_rate=sr)
   mel_params = sg.MelParams(n_mels=80, f_min=0.0, f_max=8000.0)

   spec = sg.compute_mel_power_spectrogram(samples, params, mel_params)
   print(f"Shape: {spec.shape}")

Installation
------------

.. code-block:: bash

   pip install spectrograms

For best performance with FFTW:

.. code-block:: bash

   pip install spectrograms[fftw]

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

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


