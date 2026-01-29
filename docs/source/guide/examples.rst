Examples
========

This page lists all available example scripts demonstrating various features of the spectrograms library.

All examples are located in the ``python/examples/`` directory of the repository.

Audio Processing Examples
--------------------------

Basic Spectrograms
~~~~~~~~~~~~~~~~~~

**basic_linear.py**
  Compute a simple linear power spectrogram from a sine wave.
  Demonstrates the most basic usage of the library.

  .. code-block:: bash

     python python/examples/basic_linear.py

**mel_spectrogram.py**
  Compute mel spectrograms with different amplitude scales (power, magnitude, dB).
  Shows how to use mel filterbanks for perceptual frequency scaling.

  .. code-block:: bash

     python python/examples/mel_spectrogram.py

Window Functions
~~~~~~~~~~~~~~~~

**compare_windows.py**
  Compare different window functions (Hanning, Hamming, Blackman, Kaiser).
  Visualizes the trade-offs between frequency resolution and spectral leakage.

  .. code-block:: bash

     python python/examples/compare_windows.py

Batch Processing
~~~~~~~~~~~~~~~~

**batch_processing.py**
  Efficient batch processing using the planner API.
  Demonstrates speedup by reusing FFT plans across multiple signals.

  .. code-block:: bash

     python python/examples/batch_processing.py

Streaming
~~~~~~~~~

**streaming.py**
  Frame-by-frame processing for real-time applications.
  Shows how to process audio incrementally with minimal latency.

  .. code-block:: bash

     python python/examples/streaming.py

Audio Features
~~~~~~~~~~~~~~

**mfcc_example.py**
  Compute Mel-Frequency Cepstral Coefficients (MFCCs).
  Common features for speech recognition and audio classification.

  .. code-block:: bash

     python python/examples/mfcc_example.py

**chromagram_example.py**
  Compute chromagrams (pitch class profiles).
  Useful for music analysis and chord recognition.

  .. code-block:: bash

     python python/examples/chromagram_example.py

Machine Learning Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**ml/pytorch_dlpack.py**
  DLPack protocol integration with PyTorch.
  Demonstrates  tensor exchange, chromagram conversion, and neural network pipelines.

  .. code-block:: bash

     python python/examples/ml/pytorch_dlpack.py

Image Processing Examples
--------------------------

2D FFT Basics
~~~~~~~~~~~~~

**fft2d_basic.py**
  Basic 2D FFT operations on images.
  Demonstrates forward/inverse FFT and power spectrum computation.

  .. code-block:: bash

     python python/examples/fft2d_basic.py

Image Filtering
~~~~~~~~~~~~~~~

**image_blur_fft.py**
  Apply Gaussian blur using FFT-based convolution.
  Shows how FFT convolution is faster for large kernels.

  .. code-block:: bash

     python python/examples/image_blur_fft.py

**image_edge_detection.py**
  Edge detection using high-pass filtering in frequency domain.
  Demonstrates spatial filtering techniques.

  .. code-block:: bash

     python python/examples/image_edge_detection.py

Performance Analysis
--------------------

**fft_performance_analysis.py**
  Performance comparison against NumPy and SciPy implementations.
  Measures execution time across different configurations.

  .. code-block:: bash

     python python/examples/fft_performance_analysis.py

**notebook.ipynb**
  Comprehensive Jupyter notebook with interactive benchmarks.
  Includes visualization of performance results and detailed comparisons.

  .. code-block:: bash

     jupyter lab python/examples/notebook.ipynb

Reference Implementations
~~~~~~~~~~~~~~~~~~~~~~~~~

**numpy_impls.py**
  Reference NumPy/SciPy implementations used in benchmarks.
  Useful for understanding the algorithms and comparing approaches.

Running Examples
----------------

All examples are self-contained and can be run directly:

.. code-block:: bash

   # Install the library first
   pip install spectrograms

   # Run any example
   python python/examples/<example_name>.py

Some examples may require additional dependencies for visualization:

.. code-block:: bash

   pip install matplotlib seaborn jupyter

Example Template
----------------

When creating new examples, follow this template:

.. code-block:: python

   #!/usr/bin/env python3
   """
   Brief Title

   Longer description of what this example demonstrates.
   Include any prerequisites or special requirements.
   """

   import numpy as np
   import spectrograms as sg


   def main():
       # Your example code here
       print("Example output")


   if __name__ == "__main__":
       main()

See Also
--------

- :doc:`quickstart` - Quick introduction to basic usage
- :doc:`planner_guide` - Efficient batch processing guide
- :doc:`audio_features` - Audio feature extraction guide
- :doc:`image_processing` - Image processing guide
- :doc:`ml_integration` - Machine learning integration guide
- :doc:`performance` - Performance tips and benchmarks
