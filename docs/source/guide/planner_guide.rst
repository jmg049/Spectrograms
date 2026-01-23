Batch Processing
================

When processing multiple signals with the same parameters, the planner API provides significant performance benefits by reusing FFT plans.

Why Use Plans?
--------------

Creating an FFT plan involves:

1. Allocating buffers
2. Planning the FFT algorithm
3. Optimizing for your CPU

This setup cost is amortized over multiple signals when using plans.

Basic Usage
-----------

.. code-block:: python

   import spectrograms as sg
   import numpy as np

   # Generate test signals
   signals = [np.random.randn(16000) for _ in range(100)]

   # Set up parameters
   stft = sg.StftParams(n_fft=512, hop_size=256, window="hanning")
   params = sg.SpectrogramParams(stft, sample_rate=16000)
   mel_params = sg.MelParams(n_mels=80, f_min=0.0, f_max=8000.0)
   db_params = sg.LogParams(floor_db=-80.0)

   # Create planner and plan
   planner = sg.SpectrogramPlanner()
   plan = planner.mel_db_plan(params, mel_params, db_params)

   # Process all signals
   results = [plan.compute(signal) for signal in signals]

Creating Plans
--------------

The :class:`~spectrograms.SpectrogramPlanner` creates reusable plans:

.. code-block:: python

   planner = sg.SpectrogramPlanner()

   # Linear spectrograms
   power_plan = planner.linear_power_plan(params)
   mag_plan = planner.linear_magnitude_plan(params)
   db_plan = planner.linear_db_plan(params, db_params)

   # Mel spectrograms
   mel_power = planner.mel_power_plan(params, mel_params)
   mel_mag = planner.mel_magnitude_plan(params, mel_params)
   mel_db = planner.mel_db_plan(params, mel_params, db_params)

   # ERB spectrograms
   erb_power = planner.erb_power_plan(params, erb_params)
   erb_mag = planner.erb_magnitude_plan(params, erb_params)
   erb_db = planner.erb_db_plan(params, erb_params, db_params)

Computing Spectrograms
----------------------

Full Spectrogram
~~~~~~~~~~~~~~~~

.. code-block:: python

   spec = plan.compute(samples)

Single Frame
~~~~~~~~~~~~

For streaming or real-time processing:

.. code-block:: python

   # Compute only the 10th frame
   frame = plan.compute_frame(samples, frame_idx=10)

Output Shape Prediction
~~~~~~~~~~~~~~~~~~~~~~~

Determine output dimensions before computation:

.. code-block:: python

   signal_length = 16000
   n_bins, n_frames = plan.output_shape(signal_length)

Performance Comparison
----------------------

Without plan reuse:

.. code-block:: python

   import time

   start = time.time()
   for signal in signals:
       spec = sg.compute_mel_db_spectrogram(signal, params, mel_params, db_params)
   elapsed_no_reuse = time.time() - start

With plan reuse:

.. code-block:: python

   planner = sg.SpectrogramPlanner()
   plan = planner.mel_db_plan(params, mel_params, db_params)

   start = time.time()
   for signal in signals:
       spec = plan.compute(signal)
   elapsed_with_reuse = time.time() - start

   speedup = elapsed_no_reuse / elapsed_with_reuse
   print(f"Speedup: {speedup:.2f}x")

When to Use Plans
-----------------

**Use plans when:**

- Processing multiple signals with identical parameters
- Building batch processing pipelines
- Implementing real-time systems
- Performance is critical

**Use convenience functions when:**

- Processing a single signal
- Prototyping or exploration
- Parameters change frequently
- Simplicity is preferred

Memory Considerations
---------------------

Plans hold internal state and buffers. For many different parameter configurations:

.. code-block:: python

   # Create separate plans for different configurations
   plans = {}

   for n_fft in [512, 1024, 2048]:
       stft = sg.StftParams(n_fft=n_fft, hop_size=n_fft//4, window="hanning")
       params = sg.SpectrogramParams(stft, sample_rate=16000)
       planner = sg.SpectrogramPlanner()
       plans[n_fft] = planner.mel_db_plan(params, mel_params, db_params)

   # Use appropriate plan
   spec = plans[512].compute(signal)
