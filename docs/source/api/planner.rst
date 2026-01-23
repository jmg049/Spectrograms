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
