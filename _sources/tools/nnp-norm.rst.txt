.. _nnp-norm:

nnp-norm
========

This is a helper tool for the :ref:`training procedure <training>` which enables
an optional data set normalization. Please have a look at this :ref:`explanation
<units>` why normalization may help to achieve consistent training results.
A full description of the procedure is provided in section 3.1 in [1]_. In
short, the normalization is achieved by subtracting a mean energy per atom and
converting energy and length units in such way that the energies per atom and the
forces have unit standard deviation. In practice, this requires three numbers
which are  computed from statistical characteristics of the data set:

1. mean energy per atom (keyword ``mean_energy``)

2. energy unit conversion factor (keyword ``conv_energy``)

3. length unit conversion factor (keyword ``conv_length``)
   
However, this tool will not actually apply the normalization to the data set but
rather store the above numbers as keyword-value pairs in an additional header of
the settings file. Other n2p2 tools will read them in, confirm the activation of
the normalization in the log file and automatically apply unit conversion
on-the-fly. No additional intervention by the user is required!

Requirements:
-------------

A data set with multiple configurations and a basic settings file are required:

* ``input.data``
* ``input.nn``

A working symmetry function setup in the settings file is not required, only
basic information about elements is needed.

Usage:
------

.. code-block:: none

   nnp-norm

Sample screen output:
---------------------

.. code-block:: none

   *** DATA SET NORMALIZATION ****************************************************
   
   Writing energy/atom vs. volume/atom data to "evsv.dat".
   
   Total number of structures: 20
   Total number of atoms     : 2064
   Mean/sigma energy per atom:  -2.33538240E-01 +/-   1.17415619E-03
   Mean/sigma force          :  -8.55943153E-12 +/-   2.67687948E-02
   Conversion factor energy  :   8.5167544626563699E+02
   Conversion factor length  :   2.2798325235535355E+01
   
   Writing converted data file to "output.data".
   WARNING: This data set is provided for debugging purposes only and is NOT intended for training.
   
   Writing backup of original settings file to "input.nn.bak".
   
   Writing extended settings file to "input.nn".
   
   Use this settings file for normalized training.
   *******************************************************************************

File output:
------------

* ``input.nn``: The settings file with additional normalization header.

* ``input.nn.bak``: A backup of the original settings file without normalization keywords.

* ``evsv.dat``: Energies, volumes and atom numbers of all configurations in the
  data set.

* ``output.data``: The data set with normalization applied, only for debugging
  purposes.

.. [1] Singraber, A.; Morawietz, T.; Behler, J.; Dellago, C. Parallel
   Multistream Training of High-Dimensional Neural Network Potentials. J. Chem.
   Theory Comput. 2019, 15 (5), 3075–3092. https://doi.org/10.1021/acs.jctc.8b01092
