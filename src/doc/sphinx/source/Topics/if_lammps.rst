.. _if_lammps:

LAMMPS NNP interface
====================

Purpose
-------

The LAMMPS interface adds the neural network potential method in LAMMPS. Hence,
one can use a previously fitted NNP to predict energies and forces and use
LAMMPS to propagate an MD simulation. LAMMPS parallelization via MPI is
fully supported.

Build instructions
------------------

The LAMMPS interface has been tested with version ``16Mar18``. To build LAMMPS
with support for neural network potentials follow these steps: First, build the
required libraries:

.. code-block:: none

   cd src
   make libnnpif

For dynamic linking add the argument ``MODE=shared``.

.. note::

   If dynamic linking (\ ``make libnnpif MODE=shared``\ ) is used, you need to make the NNP
   libraries visibile in your system, e.g. add this line in your ``.bashrc``\ :

.. code-block:: none

   export LD_LIBRARY_PATH=<path-to-n2p2>/lib:${LD_LIBRARY_PATH}

Then change to the LAMMPS root directory and link the repository root folder to
the ``lib`` subdirectory:

.. code-block:: bash

   cd <path-to-LAMMPS>/
   ln -s <path-to-n2p2> lib/nnp

.. danger::

   The link should be named ``nnp``\ , NOT ``n2p2``\ !

Next, copy the USER-NNP package to the LAMMPS source directory:

.. code-block:: bash

   cp -r src/interface/LAMMPS/src/USER-NNP <path-to-LAMMPS>/src

Finally activate the NNP package in LAMMPS:

.. code-block:: bash

   cd <path-to-LAMMPS>/src
   make yes-user-nnp

Now, you can compile LAMMPS for your target as usual:

.. code-block:: bash

   make <target>

.. note::

   If you want to compile a serial version of LAMMPS with neural network potential
   support, the use of MPI needs to be deactivated for ``libnnpif``. Just enable the
   ``-DNOMPI`` option in the settings makefile of your choice, e.g. ``makefile.gnu``.

Usage
-----

The neural network potential method is introduced in the context of a pair style
named ``nnp``. LAMMPS comes with a large collection of these pair styles, e.g. for
a LJ or Tersoff potential, look
`here <http://lammps.sandia.gov/doc/pair_style.html>`_ for more information. The
setup of a ``nnp`` pair style is done by issuing two commands: ``pair_style`` and
``pair_coeff``. See :ref:`this page <pair_nnp>` for a detailed
description.
