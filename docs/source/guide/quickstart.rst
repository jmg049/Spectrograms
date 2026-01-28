Quickstart
==========

This guide shows you how to get started with audio and image processing.

Audio Processing
================

Basic Spectrogram
-----------------

Compute a linear power spectrogram from a simple sine wave:

.. code-block:: python

   import numpy as np
   import spectrograms as sg

   # Generate a 440 Hz sine wave (A4 note)
   sample_rate = 16000
   duration = 1.0
   t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float64)
   samples = np.sin(2 * np.pi * 440 * t)

   # Configure STFT parameters
   stft = sg.StftParams(
       n_fft=512,
       hop_size=256,
       window=sg.WindowType.hanning,
       centre=True
   )
   params = sg.SpectrogramParams(stft, sample_rate=sample_rate)

   # Compute spectrogram
   spec = sg.compute_linear_power_spectrogram(samples, params)

   # Access results
   print(f"Shape: {spec.shape}")  # (n_bins, n_frames)
   print(f"Frequency range: {spec.frequency_range()}")
   print(f"Duration: {spec.duration()}")

Understanding the Result
------------------------

The :class:`~spectrograms.Spectrogram` object contains:

- ``data``: 2D NumPy array with shape ``(n_bins, n_frames)``
- ``frequencies``: Frequency values for each bin
- ``times``: Time values for each frame
- ``params``: The parameters used for computation

Mel Spectrogram
---------------

For perceptually-scaled analysis (common in speech and music):

.. code-block:: python

   import spectrograms as sg

   # Configure mel filterbank
   mel_params = sg.MelParams(
       n_mels=80,
       f_min=0.0,
       f_max=8000.0
   )

   # Use decibel scale for visualization
   db_params = sg.LogParams(floor_db=-80.0)

   # Compute mel spectrogram in dB
   mel_spec = sg.compute_mel_db_spectrogram(
       samples, params, mel_params, db_params
   )

Image Processing
================

Basic 2D FFT
------------

Compute the 2D FFT of an image:

.. code-block:: python

   import numpy as np
   import spectrograms as sg

   # Create or load a 256x256 image
   image = np.random.randn(256, 256)

   # Compute 2D FFT
   spectrum = sg.fft2d(image)
   print(f"Spectrum shape: {spectrum.shape}")  # (256, 129)

   # Compute power spectrum
   power = sg.power_spectrum_2d(image)

Image Filtering
---------------

Apply spatial filters to enhance or smooth images:

.. code-block:: python

   import spectrograms as sg

   # Apply Gaussian blur
   kernel = sg.gaussian_kernel_2d(size=9, sigma=2.0)
   blurred = sg.convolve_fft(image, kernel)

   # Detect edges with high-pass filter
   edges = sg.highpass_filter(image, cutoff=0.1)

   # Sharpen image
   sharpened = sg.sharpen_fft(image, amount=1.5)

Understanding Image Results
---------------------------

2D FFT functions return NumPy arrays:

- ``fft2d()``: Complex array with shape ``(nrows, ncols//2 + 1)``
- ``power_spectrum_2d()``: Real array with shape ``(nrows, ncols//2 + 1)``
- Filtering functions: Real array with same shape as input

Next Steps
----------

**Audio:**

- Learn about :doc:`choosing_parameters` for your application
- Optimize batch processing with the :doc:`planner_guide`
- Explore :doc:`audio_features` like MFCC and chromagrams

**Image:**

- Learn about :doc:`image_processing` for 2D FFT operations
- See :doc:`performance` for optimization tips
