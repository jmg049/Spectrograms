Planner API
===========

The planner API enables efficient batch processing by reusing FFT plans across multiple signals.

Creating Plans
--------------

.. autoclass:: spectrograms.SpectrogramPlanner
   :members:
   :undoc-members:
   :show-inheritance:

Plan Classes
------------

Linear Plans
~~~~~~~~~~~~

.. autoclass:: spectrograms.LinearPowerPlan
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spectrograms.LinearMagnitudePlan
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spectrograms.LinearDbPlan
   :members:
   :undoc-members:
   :show-inheritance:

Mel Plans
~~~~~~~~~

.. autoclass:: spectrograms.MelPowerPlan
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spectrograms.MelMagnitudePlan
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spectrograms.MelDbPlan
   :members:
   :undoc-members:
   :show-inheritance:

ERB Plans
~~~~~~~~~

.. autoclass:: spectrograms.ErbPowerPlan
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spectrograms.ErbMagnitudePlan
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spectrograms.ErbDbPlan
   :members:
   :undoc-members:
   :show-inheritance:

LogHz Plans
~~~~~~~~~~~

.. autoclass:: spectrograms.LogHzPowerPlan
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spectrograms.LogHzMagnitudePlan
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spectrograms.LogHzDbPlan
   :members:
   :undoc-members:
   :show-inheritance:

CQT Plans
~~~~~~~~~

.. autoclass:: spectrograms.CqtPowerPlan
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spectrograms.CqtMagnitudePlan
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spectrograms.CqtDbPlan
   :members:
   :undoc-members:
   :show-inheritance:

2D FFT Planner
--------------

For efficient batch processing of 2D FFT operations on images.

.. autoclass:: spectrograms.Fft2dPlanner
   :members:
   :undoc-members:
   :show-inheritance:

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

      # All methods reuse the cached plan
      power_spectra = [planner.power_spectrum_2d(img) for img in images]
