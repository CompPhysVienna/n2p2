Additional information for nnp-sfclust example
==============================================

The tool ``nnp-sfclust`` and the Jupyter notebook ``analyze-descriptors.ipynb``
belong together and allow to analyze the quality of descriptors (see ???link).

The data set provided here is a subset of the |Cu2S| `data set
<https://doi.org/10.5281/zenodo.2634098>`__. Every tenth configuration of fhe
subset ``input.data.6`` was extracted using the the command ``nnp-select
interval 10``.

Atomic environment files can then be created with the following settings:

.. code-block:: bash

   mpirun -np 4 nnp-sfclust 500 12 6 3 6

Finally, run the Jupyter notebook, e.g. like this:

.. code-block:: bash

   jupyter notebook ./analyze-descriptors.ipynb

.. |Cu2S| replace:: Cu\ :sub:`2`\ S
