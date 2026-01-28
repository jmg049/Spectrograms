2D FFT Functions
=================

Core 2D FFT operations for image processing.

FFT and Inverse FFT
-------------------

.. autofunction:: spectrograms.fft2d

.. autofunction:: spectrograms.ifft2d

Spectral Analysis
-----------------

.. autofunction:: spectrograms.power_spectrum_2d

.. autofunction:: spectrograms.magnitude_spectrum_2d

Frequency Shifting
------------------

.. autofunction:: spectrograms.fftshift

.. autofunction:: spectrograms.ifftshift

2D FFT Planner
--------------

.. autoclass:: spectrograms.Fft2dPlanner
   :members:
   :undoc-members:
   :show-inheritance:

   Planner for efficient batch processing of 2D FFT operations.

   Create a planner once and reuse it for multiple images of the same size
   to avoid repeated FFT plan computation overhead.

   Example
   ~~~~~~~

   .. code-block:: python

      import spectrograms as sg
      import numpy as np

      # Create planner
      planner = sg.Fft2dPlanner()

      # Process multiple images
      images = [np.random.randn(256, 256) for _ in range(10)]
      spectra = [planner.fft2d(img) for img in images]

   Methods
   ~~~~~~~

   .. method:: fft2d(data: numpy.typing.NDArray[numpy.float64]) -> numpy.typing.NDArray[numpy.complex64]

      Compute 2D FFT using cached plan.

      :param data: Input 2D array
      :return: Complex frequency array

   .. method:: ifft2d(spectrum: numpy.typing.NDArray[numpy.complex64], output_ncols: int) -> numpy.typing.NDArray[numpy.float64]

      Compute inverse 2D FFT using cached plan.

      :param spectrum: Complex frequency array
      :param output_ncols: Number of columns in output
      :return: Real 2D array

   .. method:: power_spectrum_2d(data: numpy.typing.NDArray[numpy.float64]) -> numpy.typing.NDArray[numpy.float64]

      Compute power spectrum using cached plan.

      :param data: Input 2D array
      :return: Power spectrum (squared magnitude)

   .. method:: magnitude_spectrum_2d(data: numpy.typing.NDArray[numpy.float64]) -> numpy.typing.NDArray[numpy.float64]

      Compute magnitude spectrum using cached plan.

      :param data: Input 2D array
      :return: Magnitude spectrum
