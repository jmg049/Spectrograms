Image Processing with 2D FFT
=============================

The spectrograms library provides comprehensive 2D FFT operations for image processing, including spatial filtering, convolution, edge detection, and more.

Overview
--------

2D FFT operations allow you to:

- Transform images to frequency domain for analysis
- Apply efficient convolution (faster than spatial convolution for large kernels)
- Perform spatial filtering (low-pass, high-pass, band-pass)
- Detect edges and enhance features
- Sharpen images

All operations work with 2D NumPy arrays and leverage the same high-performance Rust backend used for audio spectrograms.

Basic 2D FFT
------------

Computing the 2D FFT of an image:

.. code-block:: python

   import numpy as np
   import spectrograms as sg

   # Create or load an image (256x256)
   image = np.random.randn(256, 256)

   # Compute 2D FFT
   spectrum = sg.fft2d(image)
   print(f"Spectrum shape: {spectrum.shape}")  # (256, 129)

The output shape is ``(nrows, ncols/2 + 1)`` due to Hermitian symmetry for real-valued input.

Inverse 2D FFT
~~~~~~~~~~~~~~

Reconstruct the image from its frequency representation:

.. code-block:: python

   # Reconstruct original image
   reconstructed = sg.ifft2d(spectrum, output_ncols=256)

   # Verify reconstruction
   assert np.allclose(image, reconstructed)

Power and Magnitude Spectra
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Analyze frequency content:

.. code-block:: python

   # Compute power spectrum (squared magnitude)
   power = sg.power_spectrum_2d(image)

   # Compute magnitude spectrum
   magnitude = sg.magnitude_spectrum_2d(image)

   # Visualize centered spectrum
   power_centered = sg.fftshift(power)

Convolution
-----------

FFT-based convolution is faster than spatial convolution for large kernels.

Gaussian Blur
~~~~~~~~~~~~~

.. code-block:: python

   import spectrograms as sg

   # Create Gaussian kernel
   kernel = sg.gaussian_kernel_2d(size=9, sigma=2.0)

   # Apply blur via FFT convolution
   blurred = sg.convolve_fft(image, kernel)

For small kernels (< 5x5), spatial convolution may be faster. FFT convolution excels with larger kernels.

Custom Kernels
~~~~~~~~~~~~~~

Use any kernel for convolution:

.. code-block:: python

   # Create a simple averaging kernel
   kernel = np.ones((5, 5)) / 25.0

   # Apply convolution
   smoothed = sg.convolve_fft(image, kernel)

Spatial Filtering
-----------------

Spatial filters modify frequency content to enhance or suppress certain features.

Low-Pass Filter
~~~~~~~~~~~~~~~

Removes high-frequency components (smoothing):

.. code-block:: python

   # Keep only low frequencies (0-30% of max frequency)
   smoothed = sg.lowpass_filter(image, cutoff=0.3)

High-Pass Filter
~~~~~~~~~~~~~~~~

Removes low-frequency components (edge enhancement):

.. code-block:: python

   # Remove low frequencies, keep edges
   edges = sg.highpass_filter(image, cutoff=0.1)

Band-Pass Filter
~~~~~~~~~~~~~~~~

Keeps frequencies within a specific range:

.. code-block:: python

   # Keep mid-range frequencies
   filtered = sg.bandpass_filter(image, low_cutoff=0.1, high_cutoff=0.5)

Cutoff values are normalized frequencies in the range [0, 1], where 1.0 represents the Nyquist frequency.

Edge Detection
--------------

FFT-based edge detection emphasizes high-frequency components:

.. code-block:: python

   # Detect edges in frequency domain
   edges = sg.detect_edges_fft(image)

This is equivalent to a high-pass filter optimized for edge detection.

Image Sharpening
----------------

Enhance edges while preserving overall structure:

.. code-block:: python

   # Sharpen image (amount controls strength)
   sharpened = sg.sharpen_fft(image, amount=1.5)

Higher ``amount`` values produce stronger sharpening effects.

Batch Processing with Fft2dPlanner
-----------------------------------

For processing multiple images efficiently, use the planner API to reuse FFT plans:

.. code-block:: python

   import spectrograms as sg
   import numpy as np

   # Create multiple images
   images = [
       np.random.randn(256, 256),
       np.random.randn(256, 256),
       np.random.randn(256, 256),
   ]

   # Create planner once
   planner = sg.Fft2dPlanner()

   # Process all images (reuses FFT plan)
   spectra = []
   for image in images:
       spectrum = planner.fft2d(image)
       spectra.append(spectrum)

The planner caches FFT plans for the given image dimensions, providing 1.5-3x speedup for batch processing.

Performance Considerations
--------------------------

**When to use FFT-based operations:**

- Large kernels (> 7x7) - FFT convolution is much faster
- Multiple operations on the same image - Combine in frequency domain
- Batch processing - Use ``Fft2dPlanner`` for plan reuse

**When spatial methods may be faster:**

- Very small kernels (< 5x5) - Spatial convolution has less overhead
- Single operations - FFT setup cost may dominate

**Memory usage:**

- FFT operations require additional memory for complex frequency arrays
- Input shape ``(M, N)`` produces frequency array of shape ``(M, N//2+1)`` (complex64)

Tips and Best Practices
------------------------

1. **Normalize input**: For best results with filtering, normalize image values to [0, 1] or [-1, 1]

2. **Avoid edge artifacts**: Consider padding images before FFT to reduce edge effects

3. **Choose appropriate cutoffs**: Start with conservative cutoff values (0.1-0.3) and adjust based on results

4. **Combine operations**: Apply multiple filters in frequency domain before inverse FFT to minimize overhead

5. **Use planner for batches**: Always use ``Fft2dPlanner`` when processing multiple images of the same size

See Also
--------

- :doc:`../api/fft2d` - Complete 2D FFT API reference
- :doc:`../api/image_ops` - Image processing functions reference
- :doc:`performance` - Performance benchmarks and optimization tips
