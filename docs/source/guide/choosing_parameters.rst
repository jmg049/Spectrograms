Choosing Parameters
===================

Selecting the right parameters is crucial for spectrogram quality and performance.

STFT Parameters
---------------

FFT Size (n_fft)
~~~~~~~~~~~~~~~~

Controls frequency resolution and time resolution trade-off:

- **Larger values** (2048, 4096): Better frequency resolution, poorer time resolution
- **Smaller values** (512, 256): Better time resolution, poorer frequency resolution

**Recommendations:**

- Speech: 512
- Music: 2048
- General audio: 1024

Hop Size
~~~~~~~~

Number of samples between successive frames:

- **Smaller hop**: Better time resolution, more computation
- **Larger hop**: Faster computation, coarser time resolution

**Common ratios:**

- ``hop_size = n_fft / 4`` (75% overlap) - standard for speech
- ``hop_size = n_fft / 2`` (50% overlap) - good balance

Window Function
~~~~~~~~~~~~~~~

Affects spectral leakage:

- ``"hanning"``: General purpose, good sidelobe suppression
- ``"hamming"``: Similar to Hanning, slightly different characteristics
- ``"blackman"``: Excellent sidelobe suppression, wider main lobe
- ``"kaiser=5.0"``: Adjustable (higher beta = less leakage, wider main lobe)

Centering
~~~~~~~~~

When ``centre=True``, frames are centered by padding:

- First frame centered at ``t=0``
- Last frame centered at end of signal
- Recommended for most applications

When ``False``, no padding is applied (useful for streaming).

Default Configurations
----------------------

The library provides sensible defaults:

Speech Processing
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import spectrograms as sg

   params = sg.SpectrogramParams.speech_default(sample_rate=16000)
   # Uses: n_fft=512, hop_size=160, Hanning window, centre=True

Music Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

   import spectrograms as sg

   params = sg.SpectrogramParams.music_default(sample_rate=44100)
   # Uses: n_fft=2048, hop_size=512, Hanning window, centre=True

Mel Scale Parameters
--------------------

Number of Mel Bands
~~~~~~~~~~~~~~~~~~~

- **Speech recognition**: 40-80 bands
- **Music analysis**: 80-128 bands
- **General audio**: 64 bands

Frequency Range
~~~~~~~~~~~~~~~

Set based on your signal content:

.. code-block:: python

   # Full range (0 Hz to Nyquist)
   mel_params = sg.MelParams(n_mels=80, f_min=0.0, f_max=sample_rate/2)

   # Speech range (common human voice frequencies)
   mel_params = sg.MelParams(n_mels=40, f_min=80.0, f_max=8000.0)

   # Music range
   mel_params = sg.MelParams(n_mels=128, f_min=20.0, f_max=20000.0)

Decibel Conversion
------------------

The floor parameter clips low values:

.. code-block:: python

   # Standard for visualization
   db_params = sg.LogParams(floor_db=-80.0)

   # Higher floor for very quiet signals
   db_params = sg.LogParams(floor_db=-60.0)

ERB Scale
---------

ERB (Equivalent Rectangular Bandwidth) models human auditory perception:

.. code-block:: python

   # Good for psychoacoustic applications
   erb_params = sg.ErbParams(
       n_filters=32,
       f_min=50.0,
       f_max=8000.0
   )

Performance Considerations
--------------------------

Memory Usage
~~~~~~~~~~~~

Memory scales with:

- ``n_fft``: Larger FFT = more memory
- Signal length / ``hop_size``: More frames = more memory

Computation Time
~~~~~~~~~~~~~~~~

Factors affecting speed:

1. FFT size (larger = slower)
2. Number of frames (signal length / hop size)
3. FFT backend (FFTW is fastest)

For batch processing, use the :doc:`planner_guide` to reuse FFT plans.
