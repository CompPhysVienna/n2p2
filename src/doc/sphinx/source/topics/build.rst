.. _build:

Build instructions
==================

Code structure
--------------

This package contains multiple components with varying interdependencies and
dependencies on third-party libraries. You may not need to build all
components, this depends on the intended use. The following table lists all
components and their respective requirements (follow the links for more
information).

+---------------------------------+------------------------------+------------------------------------------------------+
| Component                       | Requirements                 | Function                                             |
+=================================+==============================+======================================================+
| :ref:`libnnp <libnnp>`          | C++11 compiler (icpc, g++)   | NNP core library (NN, SF, Structure, ...)            |
+---------------------------------+------------------------------+------------------------------------------------------+
| libnnpif                        | libnnp, MPI                  | Interfaces to other software (LAMMPS, ...)           |
+---------------------------------+------------------------------+------------------------------------------------------+
| libnnptrain                     | libnnp, MPI, GSL, Eigen 3.3+ | Dataset and training routines (Kalman, ...).         |
+---------------------------------+------------------------------+------------------------------------------------------+
| :ref:`nnp-convert`              | libnnp                       | Convert between structure file formats.              |
+---------------------------------+------------------------------+------------------------------------------------------+
| nnp-cutoff                      | libnnp                       | Test speed of different cutoff functions.            |
+---------------------------------+------------------------------+------------------------------------------------------+
| nnp-dist                        | libnnp                       | Calculate radial and angular distribution functions. |
+---------------------------------+------------------------------+------------------------------------------------------+
| :ref:`nnp-predict`              | libnnp                       | Predict energy and forces for one structure.         |
+---------------------------------+------------------------------+------------------------------------------------------+
| :ref:`nnp-prune`                | libnnp                       | Prune symmetry functions.                            |
+---------------------------------+------------------------------+------------------------------------------------------+
| :ref:`nnp-select`               | libnnp                       | Select subset from data set.                         |
+---------------------------------+------------------------------+------------------------------------------------------+
| nnp-symfunc                     | libnnp                       | Symmetry function shape from settings file.          |
+---------------------------------+------------------------------+------------------------------------------------------+
| nnp-comp2                       | libnnptrain                  | Compare prediction of 2 NNPs for data set.           |
+---------------------------------+------------------------------+------------------------------------------------------+
| nnp-dataset                     | libnnptrain                  | Calculate energies and forces for a whole data set.  |
+---------------------------------+------------------------------+------------------------------------------------------+
| :ref:`nnp-norm`                 | libnnptrain                  | Calculate normalization factors for data set.        |
+---------------------------------+------------------------------+------------------------------------------------------+
| :ref:`nnp-scaling`              | libnnptrain                  | Calculate symmetry function values for data set.     |
+---------------------------------+------------------------------+------------------------------------------------------+
| nnp-train                       | libnnptrain                  | Train a neural network potential.                    |
+---------------------------------+------------------------------+------------------------------------------------------+
| :ref:`lammps-nnp <if_lammps>`   | libnnpif                     | Pair style `nnp` for LAMMPS                          |
+---------------------------------+------------------------------+------------------------------------------------------+
| pynnp                           | libnnp, python, cython       | Python interface to NNP library.                     |
+---------------------------------+------------------------------+------------------------------------------------------+
| doc                             | Sphinx, Doxygen, Breathe     | Documentation.                                       |
+---------------------------------+------------------------------+------------------------------------------------------+

The master makefile
-------------------

A master makefile is provided in the ``src`` directory which provides targets
for all individual components.  For instance, compiling the interface library
``libnnpif`` requires only to type

.. code-block:: bash

   make libnnpif

in the ``src`` directory. Similarly, to build the application ``nnp-predict``
run

.. code-block:: bash

   make nnp-predict

If an application depends on libraries, these will be built in advance
automatically. Compiled binaries will be copied to the ``bin`` path (relative to
the root directory), whereas libraries can be found in the ``lib`` folder.  To
clean up individual components use

.. code-block:: bash

   make clean-<component>

or to clean everything (except documentation) use

.. code-block:: bash

   make clean

By default, all libraries and applications will be built for static linking,
i.e ``.a`` versions of libraries and statically built versions of executables
are created. If dynamic linking is preferred use the ``MODE=shared`` switch as
additional argument of the make command:

.. code-block:: bash

   make MODE=shared nnp-predict

This will build ``.so`` versions of libraries and executables which require
dynamic linking at runtime. Do not forget to point your linker to the ``lib``
directory, e.g. correctly set the environment variable ``LD_LIBRARY_PATH``.

There are three different choices for the ``MODE`` switch: 

   * ``static`` (*default*): This is the default which is used when no mode is
     explicitly set at the command line. Static build of libraries and
     applications.

   * ``shared``: Use for dynamic linking, creates ``.so`` versions of libraries.

   * ``test``: Special builds for CI tests and coverage reports.

Currently the build process has been tested with two different compilers, the
GNU compiler g++ 5.4 (``gnu``) and the Intel compiler 17 (``intel``). It is
possible to switch between them via the ``COMP`` variable, e.g.

.. code-block:: bash

   make libnnp COMP=intel

If you need to change compiler variables and paths have a look at the
corresponding makefiles containing global build parameters:

.. code-block:: bash

   src/makefile.gnu
   src/makefile.intel

You can also create new parameter makefiles based on the above and change the
file name suffix according to your target:

.. code-block:: bash

   src/makefile.<target>
   make libnnp COMP=<target>

.. note::

   In contrast to earlier versions it is now safe to use the `-j` switch to
   enable parallel compilation. By default only a single processor is used. For
   instance, in order to use 4 processors to build all components type:

   .. code-block:: bash
   
      make -j 4

Individual component makefiles
------------------------------

It is also possible to invoke individual makefiles for each component manually.
Just switch to the corresponding folder and use ``make MODE=<mode>
COMP=<target>``. The global build parameters will be used from the
``src/makefile.<target>`` file.

Project-wide compilation options
--------------------------------

Each of the build parameter makefiles ``src/makefile.<target>`` contains a
section at the end which allows to enable/disable certain options at compile
time:

Symmetry function groups
^^^^^^^^^^^^^^^^^^^^^^^^

**Flag:** ``-DNNP_NO_SF_GROUPS`` (default: *disabled*)

If this flag is set the symmetry function group feature will be disabled
everywhere. This will result in a much worse performance but may be useful for
debugging and development purposes. Note that disabling symmetry function groups
will not change results, please see details in this publication [1]_.

Improved symmetry function derivative memory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Flag:** ``-DNNP_FULL_SFD_MEMORY`` (default: *disabled*)

By default *n2p2* reduces the memory usage when multiple elements are present by
eliminating storage for symmetry function derivatives which are zero by
definition. This happens whenever a symmetry function is only sensitive to
neighbors of certain (and not all) elements. Then, there is no space required
for derivatives with respect to neighbors of all other elements and hence a
significant amount of memory allocation can be avoided. The actual benefit
depends on the symmetry function setup, as a rough estimate expect about 30 to
50% reduction. This feature is particularly useful for training of large data
sets when symmetry function derivatives are stored in memory (keyword
``memorize_symfunc_results``).

However, for debugging and development purposes (see e.g. `this
discussion <https://github.com/CompPhysVienna/n2p2/issues/68>`__) it can be
helpful to keep the naive, full symmetry function derivative memory allocation.
This can be achieved by enabling the flag ``-DNNP_FULL_SFD_MEMORY``. Only in
this case there is a one-to-one correspondance between the list of symmetry
functions in the :ref:`libnnp <libnnp>` output and the symmetry function
derivative vectors in :cpp:member:`nnp::Atom::Neighbor::dGdr`.

Normally, i.e. when ``-DNNP_FULL_SFD_MEMORY`` is **disabled**, an additional
section in the :ref:`libnnp <libnnp>` output will displayed after the ``SETUP:
SYMMETRY FUNCTIONS`` section, which indicates the amount of still required
memory for symmetry function derivatives. Here is how the output looks like for
the RPBE-D3 water example (``examples/nnp-predict/H2O_RPBE-D3``):

.. code-block:: none

   *** SETUP: SYMMETRY FUNCTION MEMORY *******************************************

   Symmetry function derivatives memory table for element  H :
   -------------------------------------------------------------------------------
   Relevant symmetry functions for neighbors with element:
   -  H:   15 of   27 ( 55.6 %)
   -  O:   19 of   27 ( 70.4 %)
   -------------------------------------------------------------------------------
   Symmetry function derivatives memory table for element  O :
   -------------------------------------------------------------------------------
   Relevant symmetry functions for neighbors with element:
   -  H:   18 of   30 ( 60.0 %)
   -  O:   16 of   30 ( 53.3 %)
   -------------------------------------------------------------------------------
   *******************************************************************************

Benchmarking the training program and the LAMMPS interface with the same
system gives the following results: 

+---------------------------------+-------------+------------+------------+
| ``-DNNP_FULL_SFD_MEMORY``       | *enabled*   | *disabled* | difference |
+=================================+=============+============+============+
| Training (memory)               | 55.2 GB     | 37.8 GB    | -31.5 %    |
+---------------------------------+-------------+------------+------------+
| MD with LAMMPS (memory)         | 725.6 MB    | 500.0 MB   | -31.1 %    |
+---------------------------------+-------------+------------+------------+
| MD with LAMMPS (speed)          | 33.82 s     | 34.14 s    |  +0.9 %    |
+---------------------------------+-------------+------------+------------+

Given the significant reduction in memory and the negligible impact on speed
the improved memory layout is used by default (``-DNNP_FULL_SFD_MEMORY``
disabled).

.. [1] Singraber, A.; Behler, J.; Dellago, C. Library-Based LAMMPS
   Implementation of High-Dimensional Neural Network Potentials. J. Chem. Theory
   Comput. 2019, 15 (3), 1827–1840. https://doi.org/10.1021/acs.jctc.8b00770
