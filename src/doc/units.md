Units
=====

In principle the library and all applications are agnostic to a specific system
of physical units. This means that numeric values in input files are processed
unaltered. Hence, it is the user's responsibility to provide data in a
consistent way. For instance, the same length units must be used in training
configurations (see [configuration file format](cfg_file.md)) and in the
[definition of symmetry functions](descriptors.md) (mind that some parameters,
e.g. @f$\eta, r_c@f$, need to be given in length units).
