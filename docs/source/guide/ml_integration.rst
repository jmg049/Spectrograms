Machine Learning Integration
=============================

The spectrograms library integrates seamlessly with deep learning frameworks through the **DLPack protocol**, enabling  tensor exchange with PyTorch, JAX, TensorFlow, and other ML libraries.

Overview
--------

**Key Features:**

- **zero-copy exchange**: Direct memory sharing without data duplication
- **Framework support**: PyTorch, JAX, and any DLPack-compatible library
- **Convenience wrappers**: High-level ``.to_torch()`` and ``.to_jax()`` methods
- **Metadata preservation**: Optional retention of frequency/time axes and parameters
- **Batching utilities**: Efficient multi-spectrogram batching for training
- **Device flexibility**: CPU and GPU support (framework-dependent)

The library provides two integration approaches:

1. **Direct DLPack** (``torch.from_dlpack()``): Universal,  standard
2. **Convenience modules** (``spectrograms.torch``, ``spectrograms.jax``): Enhanced ergonomics with metadata

DLPack Protocol
---------------

All spectrogram and chromagram objects implement the Python `DLPack protocol <https://dmlc.github.io/dlpack/latest/>`_, which enables  tensor exchange between libraries.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import spectrograms as sg
   import torch

   # Compute a spectrogram
   spec = sg.compute_mel_power_spectrogram(samples, params, mel_params)

   # Convert to PyTorch tensor ()
   tensor = torch.from_dlpack(spec)
   print(tensor.shape)  # (n_mels, n_frames)

This works with **any DLPack-compatible framework**:

.. code-block:: python

   # PyTorch
   import torch
   tensor = torch.from_dlpack(spec)

   # JAX
   import jax.dlpack
   array = jax.dlpack.from_dlpack(spec)

   # TensorFlow (v2.15+)
   import tensorflow as tf
   tensor = tf.experimental.dlpack.from_dlpack(spec)

   # CuPy
   import cupy
   array = cupy.from_dlpack(spec)

Memory Efficiency
~~~~~~~~~~~~~~~~~

DLPack creates a **view** of the spectrogram data without copying:

.. code-block:: python

   spec = sg.compute_mel_db_spectrogram(samples, params, mel_params, db_params)
   tensor = torch.from_dlpack(spec)

   # Both share the same underlying memory
   spec.data[0, 0] = 999.0
   print(tensor[0, 0])  # 999.0

**Important**: Keep the original ``Spectrogram`` object alive while using the tensor, as they share memory.

Supported Types
~~~~~~~~~~~~~~~

The DLPack protocol works with all spectrogram types:

- ``Spectrogram`` (from any ``compute_*`` function)
- ``Chromagram`` (from ``compute_chromagram``)
- ``Fft2dResult`` (2D FFT results)

PyTorch Integration
-------------------

For enhanced ergonomics with PyTorch, import the ``spectrograms.torch`` module:

.. code-block:: python

   import spectrograms as sg
   import spectrograms.torch  # Adds .to_torch() method

Basic Conversion
~~~~~~~~~~~~~~~~

.. code-block:: python

   spec = sg.compute_mel_power_spectrogram(samples, params, mel_params)

   # Simple conversion (returns torch.Tensor)
   tensor = spec.to_torch()

   # GPU conversion
   tensor = spec.to_torch(device='cuda')

   # With specific dtype
   tensor = spec.to_torch(device='cpu', dtype=torch.float32)

With Metadata
~~~~~~~~~~~~~

Preserve frequency/time axes and computation parameters:

.. code-block:: python

   result = spec.to_torch(device='cuda', with_metadata=True)

   # Access tensor and metadata
   result.tensor        # torch.Tensor on GPU
   result.frequencies   # np.ndarray of frequency values
   result.times         # np.ndarray of time values
   result.params        # SpectrogramParams object
   result.shape         # (n_bins, n_frames)
   result.db_range      # (min_db, max_db) if applicable

   # Move to different device
   result_cpu = result.cpu()
   result_gpu = result.cuda(device=0)

Batching for Training
~~~~~~~~~~~~~~~~~~~~~

Create batched tensors from multiple spectrograms:

.. code-block:: python

   import spectrograms.torch as sgt

   # Compute spectrograms for batch of audio samples
   specs = [
       sg.compute_mel_power_spectrogram(audio, params, mel_params)
       for audio in audio_batch
   ]

   # Stack into batch tensor (batch_size, n_mels, n_frames)
   batch_tensor = sgt.batch(specs, device='cuda')

   # With automatic padding for variable lengths
   batch_tensor = sgt.batch(specs, device='cuda', pad=True)

   # Preserve metadata for each sample
   batch_tensor, metadata = sgt.batch_with_metadata(specs, device='cuda')

Neural Network Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~

Complete example for audio classification:

.. code-block:: python

   import torch
   import torch.nn as nn
   import spectrograms as sg
   import spectrograms.torch as sgt

   class AudioClassifier(nn.Module):
       def __init__(self, n_classes=10):
           super().__init__()
           self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
           self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
           self.pool = nn.AdaptiveAvgPool2d((4, 4))
           self.fc = nn.Linear(64 * 4 * 4, n_classes)

       def forward(self, x):
           # x: (batch, n_mels, n_frames)
           x = x.unsqueeze(1)  # Add channel dim: (batch, 1, n_mels, n_frames)
           x = torch.relu(self.conv1(x))
           x = torch.relu(self.conv2(x))
           x = self.pool(x)
           x = x.flatten(1)
           return self.fc(x)

   # Training setup
   model = AudioClassifier(n_classes=10).cuda()
   stft = sg.StftParams(n_fft=512, hop_size=256, window=sg.WindowType.hanning)
   params = sg.SpectrogramParams(stft, sample_rate=16000.0)
   mel_params = sg.MelParams(n_mels=128, f_min=0.0, f_max=8000.0)
   db_params = sg.LogParams(floor_db=-80.0)

   # Training loop
   for audio_batch, labels in dataloader:
       # Compute spectrograms on CPU (fast with Rust backend)
       specs = [
           sg.compute_mel_db_spectrogram(audio, params, mel_params, db_params)
           for audio in audio_batch
       ]

       # Batch and transfer to GPU
       inputs = sgt.batch(specs, device='cuda', dtype=torch.float32)
       labels = labels.cuda()

       # Forward pass
       outputs = model(inputs)
       loss = nn.CrossEntropyLoss()(outputs, labels)

       # Backward and optimize
       loss.backward()
       optimizer.step()

JAX Integration
---------------

For JAX workflows, import the ``spectrograms.jax`` module:

.. code-block:: python

   import spectrograms as sg
   import spectrograms.jax  # Adds .to_jax() method

Basic Conversion
~~~~~~~~~~~~~~~~

.. code-block:: python

   spec = sg.compute_mel_power_spectrogram(samples, params, mel_params)

   # Simple conversion (returns jax.Array)
   array = spec.to_jax()

   # GPU placement
   array = spec.to_jax(device='gpu')

   # CPU placement
   array = spec.to_jax(device='cpu')

With Metadata
~~~~~~~~~~~~~

.. code-block:: python

   result = spec.to_jax(device='gpu', with_metadata=True)

   result.array         # jax.Array on GPU
   result.frequencies   # np.ndarray
   result.times         # np.ndarray
   result.params        # SpectrogramParams
   result.shape         # (n_bins, n_frames)

   # Move between devices
   result_cpu = result.cpu()
   result_gpu = result.gpu()

Batching
~~~~~~~~

.. code-block:: python

   import spectrograms.jax as sgj

   specs = [
       sg.compute_mel_power_spectrogram(audio, params, mel_params)
       for audio in audio_batch
   ]

   # Stack into batch array
   batch_array = sgj.batch(specs, device='gpu')

JAX Training Example
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import spectrograms as sg
   import spectrograms.jax as sgj
   from flax import linen as nn

   class AudioCNN(nn.Module):
       n_classes: int = 10

       @nn.compact
       def __call__(self, x):
           # x: (batch, n_mels, n_frames)
           x = x[..., jnp.newaxis]  # Add channel: (batch, n_mels, n_frames, 1)
           x = nn.Conv(features=32, kernel_size=(3, 3))(x)
           x = nn.relu(x)
           x = nn.Conv(features=64, kernel_size=(3, 3))(x)
           x = nn.relu(x)
           x = jnp.mean(x, axis=(1, 2))  # Global average pooling
           return nn.Dense(self.n_classes)(x)

   # Initialize model
   model = AudioCNN(n_classes=10)
   rng = jax.random.PRNGKey(0)
   params_model = model.init(rng, jnp.ones((1, 128, 100)))

   # Training step
   @jax.jit
   def train_step(params_model, inputs, labels):
       def loss_fn(params_model):
           logits = model.apply(params_model, inputs)
           return jnp.mean(optax.softmax_cross_entropy(logits, labels))

       loss, grads = jax.value_and_grad(loss_fn)(params_model)
       return loss, grads

   # Compute spectrograms and convert to JAX
   stft = sg.StftParams(n_fft=512, hop_size=256, window=sg.WindowType.hanning)
   spec_params = sg.SpectrogramParams(stft, sample_rate=16000.0)
   mel_params = sg.MelParams(n_mels=128, f_min=0.0, f_max=8000.0)

   for audio_batch, labels in dataloader:
       specs = [
           sg.compute_mel_power_spectrogram(audio, spec_params, mel_params)
           for audio in audio_batch
       ]

       inputs = sgj.batch(specs, device='gpu')
       loss, grads = train_step(params_model, inputs, labels)

API Reference
-------------

Spectrogram Methods
~~~~~~~~~~~~~~~~~~~

All ``Spectrogram`` and ``Chromagram`` objects provide:

**DLPack Protocol:**

.. code-block:: python

   __dlpack__(*, stream=None, max_version=None, dl_device=None, copy=None)
   __dlpack_device__() -> tuple[int, int]

**PyTorch** (when ``spectrograms.torch`` is imported):

.. code-block:: python

   .to_torch(device='cpu', with_metadata=False, dtype=None)
       -> torch.Tensor | TorchSpectrogram

**JAX** (when ``spectrograms.jax`` is imported):

.. code-block:: python

   .to_jax(device='cpu', with_metadata=False)
       -> jax.Array | JaxSpectrogram

Module Functions
~~~~~~~~~~~~~~~~

**spectrograms.torch:**

- ``batch(specs, device='cpu', dtype=None, pad=False) -> torch.Tensor``
- ``batch_with_metadata(specs, device='cpu', dtype=None, pad=False) -> tuple[torch.Tensor, list[dict]]``

**spectrograms.jax:**

- ``batch(specs, device='cpu', pad=False) -> jax.Array``
- ``batch_with_metadata(specs, device='cpu', pad=False) -> tuple[jax.Array, list[dict]]``

Best Practices
--------------

Memory Management
~~~~~~~~~~~~~~~~~

When using DLPack, the tensor shares memory with the original spectrogram:

.. code-block:: python

   # ✓ Good: Keep spectrogram alive
   spec = sg.compute_mel_power_spectrogram(samples, params, mel_params)
   tensor = torch.from_dlpack(spec)
   process(tensor)  # spec is still in scope

   # ✗ Bad: Spectrogram may be garbage collected
   tensor = torch.from_dlpack(
       sg.compute_mel_power_spectrogram(samples, params, mel_params)
   )
   # Original data may be freed!

If you need independent memory, use ``copy=True``:

.. code-block:: python

   spec = sg.compute_mel_power_spectrogram(samples, params, mel_params)
   tensor = torch.from_dlpack(spec.__dlpack__(copy=True))

Performance Tips
~~~~~~~~~~~~~~~~

1. **Batch on CPU, transfer once**: Compute spectrograms with CPU threads, then batch transfer to GPU

   .. code-block:: python

      # Parallel CPU computation (GIL-free)
      specs = [compute_spec(audio) for audio in batch]

      # Single GPU transfer
      gpu_batch = sgt.batch(specs, device='cuda')

2. **Reuse plans for batches**: Use planner for consistent parameters

   .. code-block:: python

      planner = sg.SpectrogramPlanner()
      plan = planner.mel_db_plan(params, mel_params, db_params)

      for batch in dataloader:
          specs = [plan.compute(audio) for audio in batch]
          gpu_batch = sgt.batch(specs, device='cuda')

3. **Choose appropriate dtype**: Use ``float32`` for ML to reduce memory

   .. code-block:: python

      tensor = spec.to_torch(device='cuda', dtype=torch.float32)

Device Support
~~~~~~~~~~~~~~

**Current limitations:**

- Spectrograms are computed on **CPU only** (using FFTW)
- DLPack transfers to GPU happen **after** computation
- GPU FFT computation is not yet supported

This is typically fine because:

- CPU spectrogram computation is very fast (Rust + FFTW)
- GPU is used for model training/inference where it's most beneficial
- Data transfer is a one-time cost per batch

See Also
--------

- :doc:`performance` - General performance optimization
- :doc:`planner_guide` - Batch processing with reusable plans
- `DLPack Specification <https://dmlc.github.io/dlpack/latest/>`_
- `PyTorch DLPack docs <https://pytorch.org/docs/stable/dlpack.html>`_
- `JAX DLPack docs <https://jax.readthedocs.io/en/latest/_autosummary/jax.dlpack.html>`_
