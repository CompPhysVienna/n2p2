.. _nnp-prune:

nnp-prune
=========

The tool ``nnp-prune`` is used to identify and remove symmetry functions that have
little effect on the training performance. There are two modes of operation:

Prune before training
^^^^^^^^^^^^^^^^^^^^^

Here, symmetry functions are discarded if their range (i.e. maximum - minimum
value over the entire data set) is below a provided threshold. The first step
is to calculate all symmetry functions with :ref:`nnp-scaling`:

.. code-block:: bash

   mpirun -np 4 nnp-scaling 500

This will produce the file ``scaling.data`` where minimum, maximum, mean and sigma
values are stored. Then, in the same directory call for example

.. code-block:: bash

   nnp-prune range 1.0E-4

to eliminate all symmetry functions with range below :math:`10^{-4}`. The resulting
settings file is called ``output-prune-range.nn``. It is an exact copy of
``input.nn`` but all lines with pruned symmetry functions are commented out. If
the result is reasonable, rename the file to ``input.nn`` and start over with the
training process (note: do not forget to calculate ``scaling.data`` again with
:ref:`nnp-scaling`).

Prune after training
^^^^^^^^^^^^^^^^^^^^

After a neural network potential has been fitted with ``nnp-train`` it is possible
to identify unnecessary symmetry functions with a so-called sensitivity
analysis. Here, the derivatives of the output neuron with respect to the input
layer neurons, i.e. the symmetry functions, are averaged over the whole data.
This analysis is automatically performed by ``nnp-dataset`` and the results are
stored in files like ``sensitivity.008.out`` (for the oxygen neural network):

.. code-block:: bash

   mpirun -np 4 nnp-dataset 1

With the sensitivity data ready, use the pruning tool to automatically remove
symmetry functions with a sensitivity below the threshold:

.. code-block:: bash

   nnp-prune sensitivity 0.5

Here, the threshold is set to 0.5% where 100% is equal to the total sum of all
sensitivities. The resulting settings file is called
``output-prune-sensitivity.nn`` and is again an exact copy of ``input.nn``,
only with commented pruned symmetry function lines.
