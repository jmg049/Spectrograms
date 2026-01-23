Convenience Functions
=====================

High-level functions for one-shot spectrogram computation. For batch processing, use the :doc:`planner` API instead.

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

Audio Features
--------------

.. autofunction:: spectrograms.compute_cqt

.. autofunction:: spectrograms.compute_chromagram

.. autofunction:: spectrograms.compute_mfcc

Low-Level Functions
-------------------

.. autofunction:: spectrograms.compute_stft
