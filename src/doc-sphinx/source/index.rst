.. n2p2 - A neural network potential package documentation master file, created by
   sphinx-quickstart on Tue Jan 29 14:37:49 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the documentation of n2p2!
=====================================

This repository (obtain source code `here <https://github.com/CompPhysVienna/n2p2>`__) provides ready-to-use
software for high-dimensional neural network potentials in materials science.
The methodology behind the Behler-Parinello neural network potentials was first
described here:

`J. Behler and M. Parrinello, Phys. Rev. Lett. 98, 146401 (2007) <https://doi.org/10.1103/PhysRevLett.98.146401>`_

This package contains software that will allow you to use existing neural
network potential parameterizations to predict energies and forces (with
standalone tools but also in conjunction with the MD software
`LAMMPS <http://lammps.sandia.gov>`_). In addition it is possible to train new
neural network potentials with the provided training tools.

Documentation
=============

.. warning::

   Unfortunately many parts of the documentation are still unfinished and will
   be completed little by little.

This package uses automatic documentation generation via `doxygen
<http://www.stack.nl/~dimitri/doxygen>`_, `sphinx <http://www.sphinx-doc.org>`_
and `exhale <https://github.com/svenevs/exhale>`_. An online version of the
documentation which is automatically updated with the main repository can be
found `here <http://compphysvienna.github.io/n2p2>`__.

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
* `nnp-predict`

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

* [nnp-comp2](nnp-comp2.md)
* [nnp-convert](nnp-convert.md)
* [nnp-dataset](nnp-dataset.md)
* [nnp-dist](nnp-dist.md)
* [nnp-norm](nnp-norm.md)
* [nnp-prune](nnp-prune.md)
* :ref:`nnp-select`
* [nnp-symfunc](nnp-symfunc.md)

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

+---------------------------------+---------------------------------+----------------------------+------------------------------------------------------+
| Component                       | Location                        | Requirements               | Function                                             |
+=================================+=================================+============================+======================================================+
| [libnnp](libnnp.md)             | `src`                           | C++98 compiler (icpc, g++) | NNP core library (NN, SF, Structure, ...)            |
+---------------------------------+---------------------------------+----------------------------+------------------------------------------------------+
| libnnpif                        | `src`                           | `libnnp`, MPI              | Interfaces to other software (LAMMPS, ...)           |
+---------------------------------+---------------------------------+----------------------------+------------------------------------------------------+
| libnnptrain                     | `src`                           | `libnnp`, MPI, GSL, Eigen  | Dataset and training routines (Kalman, ...).         |
+---------------------------------+---------------------------------+----------------------------+------------------------------------------------------+
| [nnp-convert](nnp-convert.md)   | `src/application`               | `libnnp`                   | Convert between structure file formats.              |
+---------------------------------+---------------------------------+----------------------------+------------------------------------------------------+
| [nnp-cutoff](nnp-cutoff.md)     | `src/application`               | `libnnp`                   | Test speed of different cutoff functions.            |
+---------------------------------+---------------------------------+----------------------------+------------------------------------------------------+
| [nnp-dist](nnp-dist.md)         | `src/application`               | `libnnp`                   | Calculate radial and angular distribution functions. |
+---------------------------------+---------------------------------+----------------------------+------------------------------------------------------+
| [nnp-predict](nnp-predict.md)   | `src/application`               | `libnnp`                   | Predict energy and forces for one structure.         |
+---------------------------------+---------------------------------+----------------------------+------------------------------------------------------+
| [nnp-prune](nnp-prune.md)       | `src/application`               | `libnnp`                   | Prune symmetry functions.                            |
+---------------------------------+---------------------------------+----------------------------+------------------------------------------------------+
| [nnp-select](nnp-select.md)     | `src/application`               | `libnnp`                   | Select subset from data set.                         |
+---------------------------------+---------------------------------+----------------------------+------------------------------------------------------+
| [nnp-symfunc](nnp-symfunc.md)   | `src/application`               | `libnnp`                   | Symmetry function shape from settings file.          |
+---------------------------------+---------------------------------+----------------------------+------------------------------------------------------+
| [nnp-comp2](nnp-comp2.md)       | `src/application`               | `libnnptrain`              | Compare prediction of 2 NNPs for data set.           |
+---------------------------------+---------------------------------+----------------------------+------------------------------------------------------+
| [nnp-dataset](nnp-dataset.md)   | `src/application`               | `libnnptrain`              | Calculate energies and forces for a whole data set.  |
+---------------------------------+---------------------------------+----------------------------+------------------------------------------------------+
| [nnp-norm](nnp-norm.md)         | `src/application`               | `libnnptrain`              | Calculate normalization factors for data set.        |
+---------------------------------+---------------------------------+----------------------------+------------------------------------------------------+
| [nnp-scaling](nnp-scaling.md)   | `src/application`               | `libnnptrain`              | Calculate symmetry function values for data set.     |
+---------------------------------+---------------------------------+----------------------------+------------------------------------------------------+
| [nnp-train](nnp-train.md)       | `src/application`               | `libnnptrain`              | Train a neural network potential.                    |
+---------------------------------+---------------------------------+----------------------------+------------------------------------------------------+
| [pair_style  nnp](if_lammps.md) | `src/interface/LAMMPS`          | `libnnpif`                 | %Pair style `nnp` for LAMMPS                         |
+---------------------------------+---------------------------------+----------------------------+------------------------------------------------------+
| [pynnp](pynnp.md)               | `src/`                          | `libnnp`, python, cython   | Python interface to NNP library.                     |
+---------------------------------+---------------------------------+----------------------------+------------------------------------------------------+
| doc                             | `src/doc`                       | doxygen, graphviz          | Doxygen documentation.                               |
+---------------------------------+---------------------------------+----------------------------+------------------------------------------------------+

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
   Topics/units

.. toctree::
   :hidden:
   :caption: Tools

   Tools/libnnp
   Tools/nnp-select
   Tools/nnp-scaling

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
