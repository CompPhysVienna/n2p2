.. _nnp-predict:

nnp-predict
===========

This is a simple tool to predict energies and forces of a single configuration,
given an existing HDNNP parameterization.

Requirements:
-------------
The HDNNP setup, symmetry function scaling data and weight parameters are
required in the usual files:

* ``input.data``
* ``input.nn``
* ``scaling.data``
* ``weights.???.data``

Usage:
------

.. code-block:: none

   nnp-predict <info>
               <info> ... Write structure information for debugging to "structure.out" (0/1)

The tool will output the resulting potential energy and atomic forces on the
screen and in files (see sections below). Enabling the additional debugging
output to is usually not necessary. The file "structure.out" contains basically
the entire content of the ``Structure`` and ``Atom`` instances, i.e. the full
neighbor lists and symmetry function derivatives of all atoms.

.. warning::

   Depending on the configuration the "structure.out" debugging file can be
   huge.

Sample screen output:
---------------------

.. code-block:: none

   *** PREDICTION ****************************************************************
   
   Reading structure file...
   Structure contains 144 atoms (2 elements).
   Calculating NNP prediction...
   
   -------------------------------------------------------------------------------
   NNP energy:  -5.73656032E+02
   
   NNP forces:
            1  S  -1.40007861E-01   2.74032613E-02  -5.96150479E-03
            2  S   1.39972905E-01   2.73894285E-02   5.93808974E-03
            3  S   1.39988541E-01  -2.73969011E-02   6.02337848E-03
            4  S  -1.39985315E-01  -2.73976741E-02  -6.01454763E-03
            5  S  -1.34194641E-01  -3.09539812E-02   3.93006430E-02
            .
            .
            .
          140 Cu   4.22986438E-02  -1.77295621E-02   3.63706424E-02
          141 Cu  -3.57169850E-02   7.57362417E-02   1.68472680E-03
          142 Cu   3.57606444E-02   7.57328191E-02  -1.65808098E-03
          143 Cu   3.57294618E-02  -7.57352222E-02  -1.67173728E-03
          144 Cu  -3.57158597E-02  -7.57244307E-02   1.68265914E-03
   -------------------------------------------------------------------------------
   Writing output files...
    - energy.out
    - nnatoms.out
    - nnforces.out
   Writing structure with NNP prediction to "output.data".
   Finished.
   *******************************************************************************

File output:
------------

* ``energy.out``: Contains the potential energy predicted by the NNP.

* ``nnforces.out``: Contains the atomic forces predicted by the NNP.

* ``nnatoms.out``: Contains the atomic energy contributions to the total
  potential energy.

* ``output.data``: Contains the configuration just like in the ``input.data``
  file but with NNP energy and force predictions inserted.

Examples:
---------

* Write out additional structure information file:

  .. code-block:: none

     nnp-predict 1
