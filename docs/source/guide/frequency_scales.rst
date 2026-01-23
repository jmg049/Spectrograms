Frequency Scales
================

The library supports multiple frequency scales, each suited for different applications.

Linear Scale
------------

Standard FFT frequency bins, equally spaced in Hz.

**Use cases:**

- General audio analysis
- Scientific measurements
- Transient detection

.. code-block:: python

   import spectrograms as sg

   params = sg.SpectrogramParams(stft, sample_rate=16000)
   spec = sg.compute_linear_power_spectrogram(samples, params)

**Frequency spacing:** ``sample_rate / n_fft``

Log Frequency Scale
-------------------
Logarithmically spaced frequencies.
.. code-block:: python

   import spectrograms as sg

   params = sg.SpectrogramParams(stft, sample_rate=16000)
   spec = sg.compute_log_power_spectrogram(samples, params)


Mel Scale
---------

Perceptually-motivated scale based on human pitch perception.

**Use cases:**

- Speech recognition
- Audio classification
- Music information retrieval

.. code-block:: python

   mel_params = sg.MelParams(
       n_mels=80,
       f_min=0.0,
       f_max=8000.0
   )
   mel_spec = sg.compute_mel_power_spectrogram(samples, params, mel_params)

**Properties:**

- Approximately linear below 1000 Hz
- Logarithmic above 1000 Hz
- Models human pitch discrimination

**Mel-Hertz conversion:**

$$
m = 2595 \log_{10}\left(1 + \frac{f}{700}\right)
$$

ERB Scale
---------

Equivalent Rectangular Bandwidth models auditory filter bandwidths.

**Use cases:**

- Psychoacoustic modeling
- Hearing research
- Perceptual audio coding

.. code-block:: python

   erb_params = sg.ErbParams(
       n_filters=32,
       f_min=50.0,
       f_max=8000.0
   )
   erb_spec = sg.compute_erb_power_spectrogram(samples, params, erb_params)

**Properties:**

- Based on critical band theory
- More accurate model of auditory perception than Mel
- Filter bandwidth increases with center frequency

Constant-Q Transform
--------------------

Logarithmically-spaced frequencies with constant Q factor.

**Use cases:**

- Music analysis
- Pitch detection
- Harmonic analysis

.. code-block:: python

   cqt_params = sg.CqtParams(
       bins_per_octave=12,
       n_octaves=7,
       f_min=55.0  # A1
   )
   cqt = sg.compute_cqt(samples, 22050, cqt_params, hop_size=512)

**Properties:**

- Frequency resolution matches musical intervals
- Q = constant for all bins
- Variable time resolution (higher frequencies = better time resolution)

Scale Comparison
----------------

+----------+------------------+----------------------+-------------------+
| Scale    | Spacing          | Best For             | Common Uses       |
+==========+==================+======================+===================+
| Linear   | Equal Hz         | General analysis     | FFT, physics      |
+----------+------------------+----------------------+-------------------+
| Mel      | Perceptual       | Speech/audio ML      | ASR, ML models    |
+----------+------------------+----------------------+-------------------+
| ERB      | Auditory filters | Psychoacoustics      | Hearing models    |
+----------+------------------+----------------------+-------------------+
| CQT      | Logarithmic      | Musical pitch        | Music analysis    |
+----------+------------------+----------------------+-------------------+

Choosing a Scale
----------------

**Linear:** When frequency resolution matters more than perceptual modeling

.. code-block:: python

   # Example: Detecting specific frequencies
   linear_spec = sg.compute_linear_power_spectrogram(samples, params)

**Mel:** For machine learning with speech or general audio

.. code-block:: python

   # Example: Speech recognition features
   mel_params = sg.MelParams(n_mels=40, f_min=80.0, f_max=8000.0)
   mel_spec = sg.compute_mel_db_spectrogram(samples, params, mel_params, db_params)

**ERB:** For perceptual modeling or psychoacoustic research

.. code-block:: python

   # Example: Perceptual loudness
   erb_params = sg.ErbParams(n_filters=32, f_min=50.0, f_max=16000.0)
   erb_spec = sg.compute_erb_magnitude_spectrogram(samples, params, erb_params)

**CQT:** For musical analysis or pitch-based applications

.. code-block:: python

   # Example: Music transcription
   cqt_params = sg.CqtParams(bins_per_octave=36, n_octaves=7, f_min=55.0)
   cqt = sg.compute_cqt(samples, 44100, cqt_params, hop_size=512)
