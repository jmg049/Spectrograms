Adding Examples
===============

Good documentation includes practical examples. Here's how to add them effectively.

Creating Example Files
-----------------------

Example files should be standalone and runnable:

.. code-block:: python

   #!/usr/bin/env python3
   """
   Brief Title

   Longer description of what this example demonstrates.
   """

   import numpy as np
   import spectrograms as sg


   def main():
       # Your example code here
       pass


   if __name__ == "__main__":
       main()

Place examples in ``python/examples/`` directory.

Inline Examples
---------------

For small code snippets in documentation:

.. code-block:: rst

   .. code-block:: python

      import spectrograms as sg

      # Brief, focused example
      spec = sg.compute_linear_power_spectrogram(samples, params)

Best Practices
--------------

1. **Start simple**: Begin with the most basic usage
2. **Build complexity**: Add advanced features progressively
3. **Explain why**: Don't just show what, explain when and why
4. **Show output**: Include expected results or shapes
5. **Handle errors**: Show proper error handling when relevant

Linking to API
--------------

Reference API elements using Sphinx roles:

.. code-block:: rst

   Use :class:`~spectrograms.SpectrogramParams` to configure...
   Call :func:`~spectrograms.compute_mel_power_spectrogram`...
   See :doc:`../guide/quickstart` for more...

The ``~`` prefix shows only the last component of the name.

Adding to Documentation
-----------------------

1. Create or update RST file in appropriate section
2. Add to toctree in parent index
3. Build and verify: ``python build_docs.py --serve``
4. Check for warnings and broken links
