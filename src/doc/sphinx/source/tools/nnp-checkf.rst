.. _nnp-checkf:

nnp-checkf
==========

This tool is useful for debugging as it computes numeric forces via the
symmetric difference quotient (central difference) and compares it to the
analytic forces obtained directly from n2p2's prediction. Parallelization via
MPI is supported and it can process multiple configurations in the input file.

Requirements:
-------------

The HDNNP setup, symmetry function scaling data and weight parameters are
required in the usual files. The data file ``input.data`` may contain multiple
configurations.

* ``input.data``
* ``input.nn``
* ``scaling.data``
* ``weights.???.data``

Usage:
------

.. code-block:: none

   mpirun -np <np> nnp-checkf <<delta>>

where

.. code-block:: none

   <np> ........ Number of MPI processes to use.
   <<delta>> ... (optional) Displacement for central difference (default: 1.0e-4).

The ``<<delta>>`` parameter determines the position displacement of atoms used
for the central difference approximation of the forces. If no value is provided
the default is :math:`\delta = 10^{-4}`.

Sample screen output:
---------------------

.. code-block:: none

   *** ANALYTIC/NUMERIC FORCES CHECK *********************************************
   
   Delta for symmetric difference quotient:   1.000E-04
   Individual analytic/numeric forces will be written to "checkf-forces.out"
   Per-structure summary of analytic/numeric force comparison will be
   written to "checkf-summary.out"
   Found 3 configurations in data file: input.data.
   Starting loop over 3 configurations...
   
                          numForces meanAbsError  maxAbsError  verdict
   -------------------------------------------------------------------
   Configuration      1:        144    7.168E-10    2.984E-09  OK.
   Configuration      2:        144    8.215E-10    3.212E-09  OK.
   Configuration      3:        144    7.254E-10    2.459E-09  OK.
   *******************************************************************************

File output:
------------

*  ``checkf-forces.out``: Comparison of analytic/numeric values, each force one
   line.

*  ``checkf-summary.out``: Per-structure accumulated comparison (same as screen
   output).

Examples:
---------

* Run on 4 cores and override default :math:`\delta` value:

  .. code-block:: none

     mpirun -np 4 nnp-checkf 1.0E-3

