Results
=======

Spectrogram
-----------

.. autoclass:: spectrograms.Spectrogram
   :members:
   :undoc-members:
   :show-inheritance:

DLPack Protocol
---------------

All spectrogram objects (``Spectrogram``, ``Chromagram``, ``Fft2dResult``) implement the
`DLPack protocol <https://dmlc.github.io/dlpack/latest/>`_ for  tensor exchange
with deep learning frameworks.

.. py:method:: Spectrogram.__dlpack__(*, stream=None, max_version=None, dl_device=None, copy=None)

   Export the spectrogram data as a DLPack capsule for  tensor exchange.

   This method implements the DLPack protocol, enabling efficient data sharing with
   deep learning frameworks like PyTorch, JAX, and TensorFlow without copying data.

   :param stream: Must be None for CPU tensors (reserved for future GPU support)
   :type stream: Optional[int]
   :param max_version: Maximum DLPack version supported by the consumer
   :type max_version: Optional[tuple[int, int]]
   :param dl_device: Target device (device_type, device_id). Must be (1, 0) for CPU
   :type dl_device: Optional[tuple[int, int]]
   :param copy: If True, create a copy of the data. If None or False, returns a view
   :type copy: Optional[bool]
   :return: A DLPack capsule containing the tensor data
   :raises BufferError: If parameters are invalid for CPU tensors

   **Example:**

   .. code-block:: python

      import spectrograms as sg
      import torch

      spec = sg.compute_mel_power_spectrogram(samples, params, mel_params)

      # Convert to PyTorch tensor ()
      tensor = torch.from_dlpack(spec)

      # With explicit copy
      tensor_copy = torch.from_dlpack(spec.__dlpack__(copy=True))

.. py:method:: Spectrogram.__dlpack_device__()

   Return the device type and device ID for DLPack protocol.

   :return: Tuple of (device_type, device_id). Always (1, 0) for CPU device
   :rtype: tuple[int, int]

   **Device Type Constants:**

   - ``1`` (kDLCPU): CPU device
   - ``2`` (kDLCUDA): CUDA GPU
   - ``10`` (kDLROCm): ROCm GPU

   **Example:**

   .. code-block:: python

      spec = sg.compute_mel_power_spectrogram(samples, params, mel_params)
      device_type, device_id = spec.__dlpack_device__()
      print(f"Device: type={device_type}, id={device_id}")  # Device: type=1, id=0

See :doc:`../guide/ml_integration` for complete usage examples with PyTorch and JAX.
