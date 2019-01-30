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
