Quickstart
==========

This guide shows you how to compute your first spectrogram.

Basic Example
-------------

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
       window="hanning",
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

Next Steps
----------

- Learn about :doc:`choosing_parameters` for your application
- Optimize batch processing with the :doc:`planner_guide`
- Explore :doc:`audio_features` like MFCC and chromagrams
