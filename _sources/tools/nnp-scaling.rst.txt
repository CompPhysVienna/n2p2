.. _nnp-scaling:

nnp-scaling
===========

.. warning::
   Documentation under construction...

This tool calculates all symmetry functions for a given dataset (\ ``input.data``\ ),
stores scaling information and computes neighbor and symmetry function
histograms. It is a prerequisite for training with ``nnp-train``. ``nnp-scaling``
can be called with MPI parallelization and requires an additional command line
argument, e.g.

.. code-block:: none

   mpirun -np 4 nnp-scaling 500

This will randomly distribute the given structures to 4 cores (for load
balancing). The scaling parameters (minimum, maximum, mean and sigma) of
symmetry functions are written to ``scaling.data``. Histograms showing the
symmetry function value distributions are provided in separate files (e.g.
``sf.008.0003.histo`` will contain the histogram for symmetry function 3 for
oxygen atoms). In the above example, the command line parameter ``nbin`` is set
to 500 which determines the number of bins for symmetry function histograms.
In addition, a neighbor histogram is written to ``neighbors.histo``. The screen
output contains a useful section about memory requirements during training,
e.g.

.. code-block:: none

   *** MEMORY USAGE ESTIMATION ***************************************************

   Estimated memory usage for training (keyword "memorize_symfunc_results":
   Valid for training of energies and forces.
   Memory for local structures  :     12459839800 bytes (11882.63 MiB = 11.60 GiB).
   Memory for all structures    :     49526892170 bytes (47232.53 MiB = 46.13 GiB).
   Average memory per structure :         6839786 bytes (6.52 MiB).
   *******************************************************************************

While ``nnp-scaling`` itself will not use a lot of RAM memory, training speed with
``nnp-train`` can be significantly increased if intermediate symmetry function
results can be stored and reused. This will usually require a large amount of
memory and the above lines present a rough estimate of the minimum usage. In
practice at least 10% more memory should be expected.
