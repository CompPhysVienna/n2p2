.. _if_lammps:

LAMMPS interface
================

Purpose
-------

The LAMMPS interface adds the neural network potential method in LAMMPS. Hence,
one can use a previously fitted NNP to predict energies and forces and use
LAMMPS to propagate an MD simulation. LAMMPS parallelization via MPI is
fully supported.

Build instructions
------------------

The recommended way to build LAMMPS together with n2p2 is using the LAMMPS
package management and build system. If you are new to LAMMPS, please start with
the basics of the LAMMPS build system which are described
`here <https://docs.lammps.org/Build.html>`__. Specific instructions for the
LAMMPS + n2p2 build can be found here:

* `LAMMPS Packages with extra build options: ML-HDNNP <https://docs.lammps.org/Build_extras.html#ml-hdnnp>`__
* `LAMMPS Packages details: ML-HDNNP <https://docs.lammps.org/Packages_details.html#pkg-ml-hdnnp>`__

For example, these commands will build LAMMPS with n2p2 support following the
traditional makefile method:

.. code-block:: shell

   cd /path/to/lammps/src

   # Enable ML-HDNNP package:
   make yes-ml-hdnnp

   # Automatically download and build n2p2:
   make lib-hdnnp args="-b"
   # Alternatively, use existing n2p2 installation:
   # (follow instructions in <path-to-lammps>/lib/hdnnp/README)
   # make lib-hdnnp args="-p path/to/n2p2

   # Build LAMMPS for desired target, for example:
   make mpi -j

Alternatively, the CMake build commands may look like this:

.. code-block:: shell

   cd path/to/lammps
   mkdir build; cd build
   cmake ../cmake -D PKG_ML-HDNNP=yes -D DOWNLOAD_N2P2=yes
   cmake --build . -j

Usage
-----

The neural network potential method is introduced in the context of a pair style
named ``hdnnp``. LAMMPS comes with a large collection of these pair styles, e.g. for
a LJ or Tersoff potential, look
`here <https://docs.lammps.org/Commands_pair.html>`__ for more information. The
setup of a ``hdnnp`` pair style is done by issuing two commands: ``pair_style`` and
``pair_coeff``. See the `pair style hdnnp
<https://docs.lammps.org/pair_hdnnp.html>`__ documentation page for a detailed
description.

Committees
^^^^^^^^^^

A committee of 2G-HDNNPs (keywords ``committee_mode`` and ``committee_data`` in
``input.nn``, see :ref:`keywords`) is used by ``pair_style hdnnp`` like a single
HDNNP: energies and forces are those of member 0 (validation mode) or the
committee averages (prediction mode). The committee disagreement is checked with
these additional ``pair_style hdnnp`` keywords (in the ``hdnnp`` pair style files
of n2p2, i.e. with the development build described below):

* ``calcd <integer>``: check every this many time steps (default 1).
* ``maxcdf <float>``: stop the run if the largest component of the force
  disagreement of any atom exceeds this value (LAMMPS force units). The force
  disagreement is only calculated if this value is positive (default 0).
* ``wricdf <float>``: write the atoms whose force disagreement exceeds this
  value, together with their neighbors, to files
  ``input<timestep>-<atom ID>.xyz`` (default 1.0).
* ``maxcde <float>``: stop the run if the energy disagreement per atom
  exceeds this value (LAMMPS energy units). The energy disagreement is only
  calculated if this value is positive (default 0).
* ``wricde <float>``: append time step, energy, energy disagreement and
  disagreement per atom to the file ``committee-energy.txt`` if the energy
  disagreement per atom exceeds this value (default 1.0).

The forces of all members are communicated like the forces, so the check costs
about as much as the force calculation of the members. An example is
``examples/interface-LAMMPS/H2O_RPBE-D3_Committee``.

Energy conservation with 4G-HDNNPs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With a 4G-HDNNP (``pair_style hdnnp/4g`` together with ``kspace_style hdnnp`` and
``fix hdnnp``) the real-space part of the Ewald sum is truncated at the cutoff
radius of the neural network without shifting, so the energy jumps slightly
whenever a pair of atoms crosses the cutoff. Two settings are needed for good
energy conservation in MD:

* A nonzero neighbor skin with a check in every step, e.g. (``units metal``):

  .. code-block:: none

     neighbor 1.0 bin
     neigh_modify every 1 delay 0 check yes

  With ``neighbor 0.0 bin`` and a rebuild delay, pairs that move inside the
  cutoff between two neighbor list rebuilds are missing until the next rebuild,
  and their energy is added in a jump that no force corresponds to. In tests
  with liquid water models (630 atoms, NVE, time step 0.5 fs) this caused
  energy drifts of 0.003 to 0.1 meV/atom/ps, growing with the size of the atomic
  charges. With the settings above the drift was below 1e-4 meV/atom/ps, the
  same as for a 2G-HDNNP.

* A tight relative accuracy in ``kspace_style hdnnp``, e.g. ``1.0e-5`` or
  smaller, in particular for large atomic charges. The accuracy sets the Ewald
  splitting parameter and with it the size of the real-space term left at the
  cutoff.

The 4G-HDNNP examples in ``examples/interface-LAMMPS`` use these settings. It is
good practice to check the energy conservation of every new model with a short
NVE run.

Development build
-----------------

.. danger::

   This is not the recommended way to build LAMMPS + n2p2, please consider using
   the LAMMPS package and build system as described above! The development build
   usually does not offer any benefit. Instead, it may include unstable or
   untested additions, or may even not compile at all.

For the development build the main makefile provides the compilation target
``lammps-hdnnp`` which will automatically download LAMMPS (from the `GitHub
releases page <https://github.com/lammps/lammps/releases>`__) into the
``interface`` directory, unpack it to ``lammps-hdnnp``, add the necessary n2p2
files to it and compile the LAMMPS ``mpi`` target. Finally, the binary
``lmp_mpi`` will be copied to the n2p2 ``bin`` directory. Hence, compiling
LAMMPS with NNP support is as easy as typing

The LAMMPS version which will be downloaded is determined in
``src/interface/makefile``. Close to the top there is a variable ``LAMMPS_VERSION``
which contains the LAMMPS version string, e.g. ``stable_29Aug2024``. This
default value can be changed in the makefile or also on the command-line if
desired:

.. code-block:: none

   make LAMMPS_VERSION=stable_2Aug2023 lammps-hdnnp -j

The development build will perform the following actions:

1. If not present, download the LAMMPS source tarball from GitHub.
2. Unpack the LAMMPS source code.
3. Link the necessary folders and files to the ``lammps-hdnnp/lib/hdnnp``
   directory.
4. Copy the contents of n2p2's ``src/LAMMPS/src/ML-HDNNP`` folder to the LAMMPS
   ML-HDNNP package folder.
5. Modify the LAMMPS ``mpi`` target makefile to contain compiler flags from
   n2p2.
6. Enable the ML-HDNNP package with ``make yes-ml-hdnnp``.
7. Compile the LAMMPS ``mpi`` target.
8. Copy the executable ``lmp_mpi`` to n2p2's ``bin`` directory.

Since in step 4 the LAMMPS-distributed ``hdnnp`` pair style files are replaced
with the ones located in n2p2, this build may not be identical to the
recommended way of installing LAMMPS + n2p2 described above.

.. note::

   The automatic compilation will only work on Unix-like systems as it relies on
   tools such as ``wget``, ``sed`` and ``tar``.

To remove the developer build run ``make clean-lammps-hdnnp``. However, this
will not delete the downloaded LAMMPS tarballs in the ``src/interface``
directory so they can be reused for the next build. Please delete them
manually if desired.

.. important::

   During development be aware that the **copied** files in the LAMMPS ``src``
   or ``src/ML-HDNNP`` directories are not managed by git. Be sure to transfer
   modifications back to n2p2's ``src/interface/LAMMPS/src/ML-HDNNP`` directory
   before cleaning!

Note on older development builds
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Previous versions of n2p2 (before 2.3.0) had a similar automatic development
build with **older** pair style source code files (pre-dating the `merge request
<https://github.com/lammps/lammps/pull/2626>`__ for the official LAMMPS repo).
The most obvious difference is that the pair style name was ``nnp`` and not
``hdnnp``. This version of the LAMMPS interface is not supported anymore. For
future installations please always try to use the recommended build method
described at the top of this page.

LAMMPS script files using the ``nnp`` pair style require three minor modifications
to be converted to the official ``hdnpp`` version:

1. Replace the pair style name.
2. The element mapping previously entered via the ``emap`` keyword is now listed
   in the ``pair_coeff`` line in the usual LAMMPS style.
3. The cutoff radius is moved from the ``pair_coeff`` line to the first
   (mandatory) argument of the ``pair_style`` line.

Here is an example:

.. code-block:: shell

   # Old nnp pair style lines:
   pair_style nnp showew no showewsum 100 maxew 1000 resetew yes emap "1:H,2:O"
   pair_coeff * * 6.01

   # Corresponding hdnnp pair style lines:
   pair_style hdnnp 6.01 showew no showewsum 100 maxew 1000 resetew yes
   pair_coeff * * H O

Please consult the `pair style hdnnp
<https://docs.lammps.org/pair_hdnnp.html>`__ documentation page for details. The
documentation for the old ``nnp`` pair style is kept in the n2p2 repository
here: ``src/doc/sphinx/source/old/pair_nnp.rst``.
