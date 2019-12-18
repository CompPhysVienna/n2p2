.. n2p2 - A neural network potential package documentation master file, created by
   sphinx-quickstart on Tue Jan 29 14:37:49 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the documentation of n2p2!
=====================================

This repository (obtain source code `here
<https://github.com/CompPhysVienna/n2p2>`__) provides ready-to-use software for
high-dimensional neural network potentials in computational physics and
chemistry.  The methodology behind the Behler-Parinello neural network
potentials was first described here:

`J. Behler and M. Parrinello, Phys. Rev. Lett. 98, 146401 (2007)
<https://doi.org/10.1103/PhysRevLett.98.146401>`__

This package contains software that will allow you to use existing neural
network potential parameterizations to predict energies and forces (with
standalone tools but also in conjunction with the MD software `LAMMPS
<http://lammps.sandia.gov>`__). In addition it is possible to train new neural
network potentials with the provided training tools.

Documentation
=============

.. danger::

   The build process has changed recently, please have a look at the compilation
   chapter below!

.. warning::

   Unfortunately many parts of the documentation are still unfinished and will
   be completed little by little. If you have specific questions, consider to
   ask on GitHub (file an issue) and I will update the corresponding docs as
   quickly as possible.

This package uses automatic documentation generation via `Doxygen
<http://www.doxygen.nl>`__, `Sphinx <http://www.sphinx-doc.org>`__
and `Exhale <https://github.com/svenevs/exhale>`__. An online version of the
documentation which is automatically updated with the main repository can be
found `here <https://compphysvienna.github.io/n2p2>`__.

API documentation
-----------------

Most parts of the C++ code are documented in the header files via Doxygen
annotations. The information written in the source files is automatically
extracted by `Exhale` (which uses `Doxygen`) and integrated into this
documentation (see `API` section on the left). However, because
this documentation and also `Exhale` is still under development some things may
not work as expected. As a fallback option the unaltered Doxygen API
documentation is also available `here <doxygen/index.html>`__.

Purpose
=======

This repository provides applications for multiple purposes. Depending on your
task at hand you may only need individual parts and do not need to compile all
components (see components table below). As a new user you may find yourself in
one of these three possible scenarios:

Prediction with existing neural network potential
-------------------------------------------------

If you have a working neural network potential setup (i.e. a settings file with
network and symmetry function parameters, weight files and a scaling file)
ready and want to predict energies and forces for a single structure you only
need these components:

* :ref:`libnnp <libnnp>`
* :ref:`nnp-predict`

Molecular dynamics simulation
-----------------------------

Similarly, if you have a working neural network potential setup and would like
to run an MD simulation with an external MD software (so far only LAMMPS is
supported), these components are required:

* :ref:`libnnp <libnnp>`
* `libnnpif`
* :ref:`pair_style nnp <if_lammps>`

Training a new neural network potential
---------------------------------------

To train a completely new neural network potential the following parts are required:

* :ref:`libnnp <libnnp>`
* `libnnptrain`
* :ref:`nnp-scaling`
* `nnp-train`

Additional, though not strictly required tools, are also quite useful:

* `nnp-comp2`
* :ref:`nnp-convert`
* `nnp-dataset`
* `nnp-dist`
* :ref:`nnp-norm`
* :ref:`nnp-prune`
* :ref:`nnp-select`
* `nnp-symfunc`

Rough guidelines for NNP training are provided [here](training.md).


Build process
=============

Code structure
--------------

This package contains multiple components with varying interdependencies and
dependencies on third-party libraries. You may not need to build all
components, this depends on the intended use. The following table lists all
components and their respective requirements (follow the links for more
information).

+---------------------------------+----------------------------+------------------------------------------------------+
| Component                       | Requirements               | Function                                             |
+=================================+============================+======================================================+
| :ref:`libnnp <libnnp>`          | C++11 compiler (icpc, g++) | NNP core library (NN, SF, Structure, ...)            |
+---------------------------------+----------------------------+------------------------------------------------------+
| libnnpif                        | libnnp, MPI                | Interfaces to other software (LAMMPS, ...)           |
+---------------------------------+----------------------------+------------------------------------------------------+
| libnnptrain                     | libnnp, MPI, GSL, Eigen    | Dataset and training routines (Kalman, ...).         |
+---------------------------------+----------------------------+------------------------------------------------------+
| :ref:`nnp-convert`              | libnnp                     | Convert between structure file formats.              |
+---------------------------------+----------------------------+------------------------------------------------------+
| nnp-cutoff                      | libnnp                     | Test speed of different cutoff functions.            |
+---------------------------------+----------------------------+------------------------------------------------------+
| nnp-dist                        | libnnp                     | Calculate radial and angular distribution functions. |
+---------------------------------+----------------------------+------------------------------------------------------+
| :ref:`nnp-predict`              | libnnp                     | Predict energy and forces for one structure.         |
+---------------------------------+----------------------------+------------------------------------------------------+
| :ref:`nnp-prune`                | libnnp                     | Prune symmetry functions.                            |
+---------------------------------+----------------------------+------------------------------------------------------+
| :ref:`nnp-select`               | libnnp                     | Select subset from data set.                         |
+---------------------------------+----------------------------+------------------------------------------------------+
| nnp-symfunc                     | libnnp                     | Symmetry function shape from settings file.          |
+---------------------------------+----------------------------+------------------------------------------------------+
| nnp-comp2                       | libnnptrain                | Compare prediction of 2 NNPs for data set.           |
+---------------------------------+----------------------------+------------------------------------------------------+
| nnp-dataset                     | libnnptrain                | Calculate energies and forces for a whole data set.  |
+---------------------------------+----------------------------+------------------------------------------------------+
| :ref:`nnp-norm`                 | libnnptrain                | Calculate normalization factors for data set.        |
+---------------------------------+----------------------------+------------------------------------------------------+
| :ref:`nnp-scaling`              | libnnptrain                | Calculate symmetry function values for data set.     |
+---------------------------------+----------------------------+------------------------------------------------------+
| nnp-train                       | libnnptrain                | Train a neural network potential.                    |
+---------------------------------+----------------------------+------------------------------------------------------+
| pair_style nnp                  | libnnpif                   | Pair style `nnp` for LAMMPS                          |
+---------------------------------+----------------------------+------------------------------------------------------+
| pynnp                           | libnnp, python, cython     | Python interface to NNP library.                     |
+---------------------------------+----------------------------+------------------------------------------------------+
| doc                             | Sphinx, Doxygen, Exhale    | Documentation.                                       |
+---------------------------------+----------------------------+------------------------------------------------------+

The simple way using the master makefile
----------------------------------------

A master makefile is provided in the ``src`` directory which provides targets for
all individual components (except for the LAMMPS :ref:`pair_style  nnp <if_lammps>`).
For instance, compiling the interface library ``libnnpif`` requires only to type

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

By default the makefile will use 4 processors for compiling multiple files at
once, this behaviour can be overriden with the CORES switch, e.g. to use 8
cores:

.. code-block:: bash

   make libnnp CORES=8

.. warning::

   Do not use the `-j` switch for the master makefile, this may mess up
   the compilation order.

Individual component makefiles
------------------------------

It is also possible to invoke individual makefiles for each component manually.
Just switch to the corresponding folder and use ``make MODE=<mode>
COMP=<target>``. The global build parameters will be used from the
``src/makefile.<target>`` file.

Examples
========

Applications
------------

Minimal working examples for each application can be found in the corresponding
subdirectory in ``examples``, e.g. ``examples/nnp-train``.

Training data sets and single configurations (for testing)
----------------------------------------------------------

Small data sets for testing purposes can be found in
``examples/configuration-sets`` and single configurations are provided in
``configuration-single``.

Training data sets (full size)
------------------------------
Actual full size data sets may be rather large and are therefore hosted elsewhere:

+-------------+--------------+
| System      | Link         |
+=============+==============+
| |H2O| [1]_  | |h2o_data|_  |
+-------------+--------------+
| |Cu2S| [2]_ | |cu2s_data|_ |
+-------------+--------------+

NNP potentials ready for use
----------------------------

Working pre-trained NNP potentials are located in ``examples/potentials``.


Keywords
========

The setup of a neural network potential (network topology, symmetry function
parameters,...) is stored in a simple text file with keyword-argument pairs. A
list of keywords is provided :ref:`here <keywords>`.


.. toctree::
   :hidden:

   Overview<self>

.. toctree::
   :hidden:
   :caption: Topics

   Topics/descriptors
   Topics/keywords
   Topics/cfg_file
   Topics/if_lammps
   Topics/pair_nnp
   Topics/training
   Topics/units

.. toctree::
   :hidden:
   :caption: Tools

   Tools/libnnp
   Tools/nnp-convert
   Tools/nnp-norm
   Tools/nnp-predict
   Tools/nnp-prune
   Tools/nnp-select
   Tools/nnp-scaling

.. toctree::
   :hidden:
   :caption: API

   doc-exhale/root

.. toctree::
   :hidden:
   :caption: About

   About/authors
   About/license
   About/changelog

.. |H2O| replace:: H\ :sub:`2`\ O

.. |Cu2S| replace:: Cu\ :sub:`2`\ S

.. |h2o_data| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.2634098.svg
.. _h2o_data: https://doi.org/10.5281/zenodo.2634098

.. |cu2s_data| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.2603918.svg
.. _cu2s_data: https://doi.org/10.5281/zenodo.2603918

.. [1] Morawietz, T.; Singraber, A.; Dellago, C.; Behler, J. How van Der Waals
   Interactions Determine the Unique Properties of Water. Proc. Natl. Acad. Sci.
   U. S. A. 2016, 113 (30), 8368–8373. `https://doi.org/10.1073/pnas.1602375113. <https://doi.org/10.1073/pnas.1602375113>`__

.. [2] Singraber, A.; Morawietz, T.; Behler, J.; Dellago, C. Parallel
   Multistream Training of High-Dimensional Neural Network Potentials. J. Chem.
   Theory Comput. 2019, 15 (5), 3075–3092. `https://doi.org/10.1021/acs.jctc.8b01092. <https://doi.org/10.1021/acs.jctc.8b01092>`__
