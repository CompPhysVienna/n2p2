.. _units:

Units
=====

In principle the library and all applications are agnostic to a specific system
of physical units. This means that numeric values in input files are processed
unaltered. Hence, it is the user's responsibility to provide data in a
consistent way. For instance, the same length units must be used in training
configurations (see :ref:`configuration file format<cfg_file>`) and in the
:ref:`definition of symmetry functions<descriptors>` (mind that some parameters,
e.g. :math:`\eta, r_c`, need to be given in length units).

.. warning::

   Note that unit conversion may be required if an existing neural network
   potential is used to drive an MD simulation in LAMMPS. If the LAMMPS units
   (see command `units <https://lammps.sandia.gov/doc/units.html>`__) are not
   matching with those used during NNP training, appropriate conversion factors
   need to be provided. See the :ref:`pair_style nnp reference <pair_nnp>` for
   further details. 

Normalizing the data set: "internal" units
------------------------------------------

Processing data sets unaltered potentially introduces a dependence of training
results on the chosen unit system, i.e. if the same data set would be set up
with different physical unit systems, it is unclear whether the training would
converge to comparable errors. To avoid this problem an additional
pre-processing of the training data can be performed with the tool
:ref:`nnp-norm`. This tool will determine conversion factors for a reduced
"internal" unit system and add them to the settings file. Other tools will
recognize the corresponding :ref:`keywords <keywords>` and automatically
perform the conversion to "internal" units. No additional intervention of the
user is necessary and quantities are usually converted back to physical units
for screen or file output.

.. note::

   Sometimes quantities are provided also in internal units (for debugging
   purposes). If this is the case it will be explicitly mentioned in the screen
   output or in the file header. The default output of all tools is given in the
   original physical unit system.

For further details see the :ref:`tool description <nnp-norm>` and a recent
publication [1]_.

.. important::

   The described data set normalization step is **optional**, you are not
   required to perform it. It is also not guaranteed that the quality of
   training will increase in all cases.

.. [1] Singraber, A.; Morawietz, T.; Behler, J.; Dellago, C. Parallel
   Multistream Training of High-Dimensional Neural Network Potentials. J. Chem.
   Theory Comput. 2019, 15 (5), 3075–3092. https://doi.org/10.1021/acs.jctc.8b01092
