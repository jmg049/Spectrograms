Performance and Benchmarks
===========================

The spectrograms library is designed for high performance, with a Rust core and  Python bindings.

Benchmark Results
-----------------

Benchmarks comparing ``spectrograms`` against NumPy and SciPy implementations are available in the `PYTHON_BENCHMARK.md <https://github.com/jmg049/Spectrograms/blob/main/PYTHON_BENCHMARK.md>`_ file.

Summary
~~~~~~~

Average speedups across all parameter configurations and signal types:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 25 25

   * - Operation
     - Rust (ms)
     - NumPy (ms)
     - SciPy (ms)
     - Avg Speedup
   * - Power
     - 0.126
     - 0.205
     - 0.327
     - 1.6x / 2.6x
   * - Magnitude
     - 0.140
     - 0.198
     - 0.319
     - 1.4x / 2.3x
   * - Decibels
     - 0.257
     - 0.350
     - 0.451
     - 1.4x / 1.8x
   * - Mel
     - 0.180
     - 0.630
     - 0.612
     - **3.5x / 3.4x**
   * - LogHz
     - 0.178
     - 0.547
     - 0.534
     - **3.1x / 3.0x**
   * - ERB
     - 0.601
     - 3.713
     - 3.714
     - **6.2x / 6.2x**

Key Findings
~~~~~~~~~~~~

1. **Filterbank operations** (Mel, ERB, LogHz) show the largest speedups (3-6x) due to:

   - Pre-computed filterbanks cached in plans
   - Sparse matrix operations
   - Minimal memory allocation

2. **Basic operations** (Power, Magnitude, dB) show 1.4-2.6x speedups from:

   - Rust's performance
   -  NumPy integration
   - GIL release during computation

3. **Consistency**: Low standard deviations show reliable, predictable performance

Why spectrograms is Faster
---------------------------

The library achieves superior performance through several optimizations that are applied automatically:

Pre-computed Filterbanks
~~~~~~~~~~~~~~~~~~~~~~~~~

When using the planner API, filterbanks (Mel, ERB, LogHz) are computed once and cached:

.. code-block:: python

   planner = sg.SpectrogramPlanner()
   plan = planner.mel_db_plan(params, mel_params, db_params)

   # Filterbank computed once ↑

   for signal in signals:
       spec = plan.compute(signal)  # Reuses cached filterbank

NumPy/SciPy recompute filterbanks on every call, wasting time on redundant calculations.

Sparse Matrix Operations
~~~~~~~~~~~~~~~~~~~~~~~~~

Filterbanks are stored as sparse matrices and applied using optimized sparse matrix-vector multiplication, avoiding unnecessary computations on zero elements.

Memory Efficiency
~~~~~~~~~~~~~~~~~

The Rust implementation uses:

- Pre-allocated workspace buffers
- Minimal temporary allocations
- Efficient memory layouts

GIL Release
~~~~~~~~~~~

All computation functions release Python's Global Interpreter Lock (GIL), enabling:

- Parallel processing of multiple files across threads
- Concurrent computation with other Python operations

Optimization Tips
-----------------

1. Use the Planner API
~~~~~~~~~~~~~~~~~~~~~~

Always use plans for batch processing:

.. code-block:: python

   # Slow: Creates new plan every iteration
   for signal in signals:
       spec = sg.compute_mel_db_spectrogram(signal, params, mel_params, db_params)

   # Fast: Reuses plan
   planner = sg.SpectrogramPlanner()
   plan = planner.mel_db_plan(params, mel_params, db_params)
   for signal in signals:
       spec = plan.compute(signal)

2. Choose Power-of-2 FFT Sizes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FFT algorithms are optimized for power-of-2 sizes:

.. code-block:: python

   # Fast
   stft = sg.StftParams(n_fft=512, ...)   # 2^9
   stft = sg.StftParams(n_fft=1024, ...)  # 2^10
   stft = sg.StftParams(n_fft=2048, ...)  # 2^11

   # Slower
   stft = sg.StftParams(n_fft=1000, ...)  # Not power-of-2

3. Streaming for Real-Time Applications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For real-time processing, use frame-by-frame computation:

.. code-block:: python

   plan = planner.mel_db_plan(params, mel_params, db_params)

   for frame_idx in range(n_frames):
       frame_data = plan.compute_frame(signal, frame_idx)
       # Process frame immediately

This minimizes latency and memory usage.

4.  DLPack Transfers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When using ML frameworks, leverage the DLPack protocol for  tensor exchange:

.. code-block:: python

   import spectrograms as sg
   import spectrograms.torch as sgt

   # Compute spectrograms on CPU (fast with Rust)
   specs = [
       sg.compute_mel_power_spectrogram(audio, params, mel_params)
       for audio in batch
   ]

   #  transfer to GPU
   gpu_batch = sgt.batch(specs, device='cuda')

**Benefits:**

- No data copying from spectrogram to tensor
- Direct memory sharing between library and framework
- Efficient batch transfer to GPU

**Memory Note:** Keep the original ``Spectrogram`` objects alive while using tensors, as they share memory:

.. code-block:: python

   # ✓ Good: Spectrogram kept in scope
   spec = sg.compute_mel_power_spectrogram(samples, params, mel_params)
   tensor = torch.from_dlpack(spec)
   result = model(tensor)

   # ✗ Bad: Spectrogram may be garbage collected
   tensor = torch.from_dlpack(
       sg.compute_mel_power_spectrogram(samples, params, mel_params)
   )

See :doc:`ml_integration` for complete ML integration guide.

5. Batch Processing with Parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since computation releases the GIL, process multiple files in parallel:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since computation releases the GIL, process multiple files in parallel:

.. code-block:: python

   from concurrent.futures import ThreadPoolExecutor

   planner = sg.SpectrogramPlanner()
   plan = planner.mel_db_plan(params, mel_params, db_params)

   def process_file(signal):
       return plan.compute(signal)

   with ThreadPoolExecutor(max_workers=4) as executor:
       results = list(executor.map(process_file, signals))

Measuring Your Performance
---------------------------

Use the included benchmark notebook to measure performance on your system:

.. code-block:: bash

   # Install development dependencies
   pip install jupyter matplotlib seaborn

   # Run benchmark notebook
   jupyter lab python/examples/notebook.ipynb

This provides detailed timings for your specific hardware and configurations.

See Also
--------

- `PYTHON_BENCHMARK.md <https://github.com/jmg049/Spectrograms/blob/main/PYTHON_BENCHMARK.md>`_ - Full benchmark results
- :doc:`planner_guide` - Efficient batch processing
- ``python/examples/fft_performance_analysis.py`` - Performance analysis example
