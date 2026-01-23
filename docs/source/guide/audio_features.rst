Audio Features
==============

Beyond spectrograms, the library provides common audio features for music and speech analysis.
These are extensions built on top of the core spectrogram computations.

MFCC (Mel-Frequency Cepstral Coefficients)
-------------------------------------------

MFCCs are widely used in speech recognition and audio classification.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import spectrograms as sg
   import numpy as np

   # Generate or load audio
   samples = np.random.randn(16000)  # 1 second at 16 kHz

   # Configure parameters
   stft = sg.StftParams(n_fft=512, hop_size=160, window="hanning")
   mfcc_params = sg.MfccParams(n_mfcc=13)

   # Compute MFCCs
   mfccs = sg.compute_mfcc(
       samples,
       stft_params=stft,
       sample_rate=16000,
       n_mels=40,
       mfcc_params=mfcc_params
   )

   print(f"Shape: {mfccs.shape}")  # (13, n_frames)

Standard Configuration
~~~~~~~~~~~~~~~~~~~~~~

For speech recognition:

.. code-block:: python

   mfcc_params = sg.MfccParams.speech_standard()  # 13 coefficients

Custom Configuration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # More coefficients for detailed analysis
   mfcc_params = sg.MfccParams(n_mfcc=20)

The first coefficient (C0) represents energy. Coefficients 1-12 capture spectral envelope.

Chromagram
----------

Chromagrams represent pitch class energy, useful for music analysis and chord recognition.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import spectrograms as sg

   # Configure chroma parameters
   chroma_params = sg.ChromaParams(
       tuning=440.0,      # A4 reference frequency
       f_min=32.7,        # C1
       f_max=4186.0,      # C8
       norm="l2"          # Normalize each frame
   )

   stft = sg.StftParams(n_fft=4096, hop_size=1024, window="hanning")

   # Compute chromagram
   chroma = sg.compute_chromagram(
       samples,
       stft_params=stft,
       sample_rate=44100,
       chroma_params=chroma_params
   )

   print(f"Shape: {chroma.shape}")  # (12, n_frames)

Output
~~~~~~

Returns a ``(12, n_frames)`` array where each row corresponds to a pitch class:

- Row 0: C
- Row 1: C#/Db
- Row 2: D
- ...
- Row 11: B

Standard Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   chroma_params = sg.ChromaParams.music_standard()

Normalization Options
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # L2 normalization (unit length)
   chroma_params = sg.ChromaParams(norm="l2")

   # L1 normalization (sum to 1)
   chroma_params = sg.ChromaParams(norm="l1")

   # Max normalization (peak at 1)
   chroma_params = sg.ChromaParams(norm="max")

   # No normalization
   chroma_params = sg.ChromaParams(norm=None)

Constant-Q Transform (CQT)
--------------------------

CQT provides logarithmically-spaced frequency bins, matching musical scales.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import spectrograms as sg

   # Configure CQT
   cqt_params = sg.CqtParams(
       bins_per_octave=12,  # Semitone resolution
       n_octaves=7,         # 7 octaves
       f_min=55.0           # A1
   )

   # Compute CQT
   cqt = sg.compute_cqt(
       samples,
       sample_rate=22050,
       cqt_params=cqt_params,
       hop_size=512
   )

   print(f"Shape: {cqt.shape}")  # (num_bins, n_frames)
   print(f"Complex output: {cqt.dtype}")  # complex128

Output
~~~~~~

Returns a complex-valued array. To get magnitude:

.. code-block:: python

   magnitude = np.abs(cqt)
   power = np.abs(cqt) ** 2
   db = 20 * np.log10(np.abs(cqt) + 1e-10)

Configuration
~~~~~~~~~~~~~

.. code-block:: python

   # High frequency resolution (36 bins per octave, third-tones)
   cqt_params = sg.CqtParams(bins_per_octave=36, n_octaves=7, f_min=55.0)

   # Standard (12 bins per octave, semitones)
   cqt_params = sg.CqtParams(bins_per_octave=12, n_octaves=7, f_min=55.0)

The total number of bins is ``bins_per_octave * n_octaves``.

Applications
------------

Speech Recognition
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Standard ASR features
   mfcc_params = sg.MfccParams.speech_standard()
   stft = sg.StftParams(n_fft=512, hop_size=160, window="hamming")

   mfccs = sg.compute_mfcc(samples, stft, 16000, 40, mfcc_params)

Music Analysis
~~~~~~~~~~~~~~

.. code-block:: python

   # Chord recognition
   chroma_params = sg.ChromaParams.music_standard()
   stft = sg.StftParams(n_fft=4096, hop_size=1024, window="hanning")

   chroma = sg.compute_chromagram(samples, stft, 44100, chroma_params)

Audio Classification
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Mel spectrograms for neural networks
   mel_params = sg.MelParams(n_mels=128, f_min=0.0, f_max=8000.0)
   db_params = sg.LogParams(floor_db=-80.0)

   mel_spec = sg.compute_mel_db_spectrogram(samples, params, mel_params, db_params)
