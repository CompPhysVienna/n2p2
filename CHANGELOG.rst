Changelog
=========

All notable changes to this project will be documented in this file.

Version 2.3.0 - 2024-11-??
--------------------------

New features
^^^^^^^^^^^^

* Allow manual definition of train/test set (add ``set=train/test`` after ``begin`` in
  input.data).
* Additional symmetry function log output, extra column ``spread``.
* Structure ``remap()`` function is now available in pynnp.

Maintenance
^^^^^^^^^^^

* Fixed CI: Cython updated to version 3.0+, added ``__rmul__`` to ``Vec3D`` in
  pynnp.

Important changes
^^^^^^^^^^^^^^^^^

* Synchronized LAMMPS developer build with LAMMPS-distributed source: Finally
  switched to pair style ``hdnnp`` everywhere.

Documentation
^^^^^^^^^^^^^

* Removed old ``nnp`` pair style description page.
* Updated LAMMPS interface build instructions page.
* Added train/test set split documentation.

Version 2.2.0 - 2022-05-23
--------------------------

New features
^^^^^^^^^^^^

* Normalization procedure on-the-fly via nnp-train, keyword ``normalize_data_set``.
* New normalization scheme based on force predictions.
* New keyword ``force_energy_ratio`` for nnp-train.
* Additional neighbor statistics output from nnp-scaling.
* New tool nnp-checkdw (for testing purposes).

Important changes
^^^^^^^^^^^^^^^^^

* New CI via Github Actions.
* Use self-compiled doxygen for API docs, use version 1.9.4.

Bugfixes
^^^^^^^^

* Fixed missing atomic energy offsets for LAMMPS interface.
* Fixed NaNs from nnp-dist when running mixed data sets.
* Fixed bugs in pynnp examples.

Version 2.1.4 - 2021-05-11
--------------------------

New features
^^^^^^^^^^^^

* Changed build behavior for LAMMPS interface to conform with PR (lammps/lammps#2626)


Version 2.1.3 - 2021-04-26
--------------------------

Bugfixes
^^^^^^^^

* Fixed bug in nnp-scaling which could cause segfaults (MPI only).


Version 2.1.2 - 2021-04-12
--------------------------

New features
^^^^^^^^^^^^

* New tool to check numeric vs. analytic forces

Bugfixes
^^^^^^^^

* Fixed critical bug for large cutoff values (large compared to lattice vectors)


Version 2.1.1 - 2021-01-15
--------------------------

New features
^^^^^^^^^^^^

* Now compatible with latest LAMMPS stable release 29Oct2020
* Improved EW warnings in LAMMPS
* Citation header in screen output
* Restructured training to prepare for multi-network HDNNPs (e.g. 4G-HDNNP)
* New training "properties" can be added more easily now

Bugfixes
^^^^^^^^

* Fixed bugs in makefiles
* Fixed bug when creating scaling data


Version 2.1.0 - 2020-12-21
--------------------------

New features
^^^^^^^^^^^^

* Polynomial symmetry functions (Martin P. Bircher and Andreas Singraber)
* Better symmetry function cache mechanism

Important changes
^^^^^^^^^^^^^^^^^

* Documentation updates (API docs in Sphinx via Breathe)


Version 2.0.3 - 2020-11-30
--------------------------

New features
^^^^^^^^^^^^

* Tool nnp-scaling puts out rejected configurations into a separate file.


Version 2.0.2 - 2020-11-25
--------------------------

Bugfixes
^^^^^^^^

* Bugfix in CabanaMD interface: SF group initialization


Version 2.0.1 - 2020-10-05
--------------------------

Bugfixes
^^^^^^^^

* Fix bug in makefile which sometimes causes a failed build due to parallel execution


Version 2.0.0 - 2020-10-05
--------------------------

New features
^^^^^^^^^^^^

* Training library (Multi-stream Kalman filter training).
* Tools for HDNNP data set handling, etc.
* Python interface (basic functionality).
* Sphinx documentation (+ Doxygen API reference).
* CabanaMD interface (by Saaketh Desai and Sam Reeve)

Important changes
^^^^^^^^^^^^^^^^^

* License change from MPL 2.0 to GPL v3 or later.


Version 1.0.0 - 2018-08-13
--------------------------

New features
^^^^^^^^^^^^

* Core library (NN, symmetry functions, ...).
* LAMMPS interface.
* Documentation (in parts) via doxygen.
