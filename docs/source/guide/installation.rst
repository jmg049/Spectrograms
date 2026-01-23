Installation
============

Requirements
------------

- Python 3.8 or higher
- NumPy

Install from PyPI
-----------------

.. code-block:: bash

   pip install spectrograms

Install from Source
-------------------

.. code-block:: bash

   git clone https://github.com/jmg/spectrograms
   cd spectrograms
   pip install .

.. Optional: FFTW Backend
.. ----------------------

.. For best performance, install with FFTW support:

.. .. code-block:: bash

..    pip install spectrograms[fftw]

.. This requires FFTW3 to be installed on your system. If not available, the library falls back to the pure-Rust RealFFT backend.
