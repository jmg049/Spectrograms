Installation
============

Requirements
------------

- Python 3.9 or higher
- NumPy 1.26 or higher

Install from PyPI
-----------------

.. code-block:: bash

   pip install spectrograms

Optional Dependencies
---------------------

For Machine Learning Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To use the spectrograms library with deep learning frameworks, install the optional ML dependencies:

**PyTorch:**

.. code-block:: bash

   pip install torch

Visit `PyTorch's official website <https://pytorch.org/>`_ for GPU-specific installation instructions.

AND/OR for JAX

**JAX:**

.. code-block:: bash

   pip install jax jaxlib

For GPU support:

.. code-block:: bash

   # NVIDIA GPU (CUDA)
   pip install jax[cuda12]

   # Or for CUDA 11
   pip install jax[cuda11]

Visit `JAX's installation guide <https://jax.readthedocs.io/en/latest/installation.html>`_ for detailed instructions.

AND/OR for TensorFlow

**TensorFlow:**

.. code-block:: bash

   pip install tensorflow

The DLPack protocol is supported in TensorFlow 2.15+.

**Note:** You only need to install the frameworks you plan to use. The core spectrograms library works independently and will use the DLPack protocol automatically when these frameworks are available.
The library can be installed with the optional dependencies (PyTorch and JAX so far) for machine learning integration, or without them for basic spectrogram and image processing functionality.

Install from Source
-------------------

.. code-block:: bash

   git clone https://github.com/jmg/Spectrograms
   cd spectrograms
   pip install .

