Convenience Functions
=====================

High-level functions for one-shot computation. For batch processing, use the :doc:`planner` API instead.

Audio Processing Functions
===========================

Linear Spectrograms
-------------------

.. autofunction:: spectrograms.compute_linear_power_spectrogram

.. autofunction:: spectrograms.compute_linear_magnitude_spectrogram

.. autofunction:: spectrograms.compute_linear_db_spectrogram

Mel Spectrograms
----------------

.. autofunction:: spectrograms.compute_mel_power_spectrogram

.. autofunction:: spectrograms.compute_mel_magnitude_spectrogram

.. autofunction:: spectrograms.compute_mel_db_spectrogram

ERB Spectrograms
----------------

.. autofunction:: spectrograms.compute_erb_power_spectrogram

.. autofunction:: spectrograms.compute_erb_magnitude_spectrogram

.. autofunction:: spectrograms.compute_erb_db_spectrogram

LogHz Spectrograms
------------------

.. autofunction:: spectrograms.compute_loghz_power_spectrogram

.. autofunction:: spectrograms.compute_loghz_magnitude_spectrogram

.. autofunction:: spectrograms.compute_loghz_db_spectrogram

Audio Features
--------------

.. autofunction:: spectrograms.compute_cqt

.. autofunction:: spectrograms.compute_chromagram

.. autofunction:: spectrograms.compute_mfcc

Low-Level Audio Functions
--------------------------

.. autofunction:: spectrograms.compute_stft

Image Processing Functions
===========================

2D FFT Operations
-----------------

.. autofunction:: spectrograms.fft2d

.. autofunction:: spectrograms.ifft2d

.. autofunction:: spectrograms.power_spectrum_2d

.. autofunction:: spectrograms.magnitude_spectrum_2d

Frequency Shifting
------------------

.. autofunction:: spectrograms.fftshift

.. autofunction:: spectrograms.ifftshift

Kernels
-------

.. autofunction:: spectrograms.gaussian_kernel_2d

Convolution
-----------

.. autofunction:: spectrograms.convolve_fft

Spatial Filtering
-----------------

.. autofunction:: spectrograms.lowpass_filter

.. autofunction:: spectrograms.highpass_filter

.. autofunction:: spectrograms.bandpass_filter

Feature Enhancement
-------------------

.. autofunction:: spectrograms.detect_edges_fft

.. autofunction:: spectrograms.sharpen_fft
