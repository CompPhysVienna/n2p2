.. _pair_nnp:

LAMMPS NNP pair style
=====================

pair_style nnp command
----------------------

Syntax
^^^^^^

.. code-block:: none

   pair_style nnp keyword value ...

* zero or more keyword/value pairs may be appended

* keyword = *dir* or *showew* or *showewsum* or *maxew* or *resetew* or *cflength* or *cfenergy*

* value depends on the preceding keyword:

  *  *dir* value = directory

      directory = Path to NNP configuration files

  *   *showew* value = *yes* or *no*

  *   *showewsum* value = summary

       summary = Write EW summary every this many timesteps (*0* turns summary off)

  *   *maxew* value = threshold

       threshold = Maximum number of EWs allowed

  *   *resetew* value = *yes* or *no*

  *   *cflength* value = length

       length = Length unit conversion factor

  *   *cfenergy* value = energy

       energy = Energy unit conversion factor

Examples
^^^^^^^^

.. code-block:: none

   pair_style nnp showew yes showewsum 100 maxew 1000 resetew yes cflength 1.8897261328 cfenergy 0.0367493254

   pair_style nnp dir "./" showewsum 10000

   pair_coeff * * 6.01

.. warning::

   Only use a single `pair_style nnp` line in your LAMMPS script.

Description
^^^^^^^^^^^

This pair style adds an interaction based on the high-dimensional neural network
potential method [1]_. These potentials must
be carefully trained to reproduce the potential energy surface in the desired
phase-space region prior to their usage in an MD simulation. This pair style
uses an interface to the NNP library [2]_ [3]_, see the documentation
there for more information.

The maximum cutoff radius of all symmetry functions is the only argument of the
*pair_coeff* command which should be invoked with asterisk wild-cards only:

.. code-block:: none

   pair_coeff * * cutoff

.. note::

   The cutoff must be given in LAMMPS length units, even if the neural network
   potential has been trained using a different unit system (see remarks about the
   *cflength* and *cfenergy* keywords below for details).

The numeric value may be slightly larger than the actual maximum symmetry
function cutoff radius (to account for rounding errors when converting units),
but must not be smaller.

.. important::

   The atom type specifications in configuration files used to
   start MD simulations must be consistent with the ordering of elements in the NNP
   library. Thus, atom types must be sorted in order of ascending atomic number,
   e.g. the only correct mapping for a configuration containing hydrogen, oxygen
   and zinc atoms is as follows:
   
   +---------+---------------+-------------+
   | Element | Atomic number | LAMMPS type |
   +=========+===============+=============+
   |       H |             1 |           1 |
   +---------+---------------+-------------+
   |       O |             8 |           2 |
   +---------+---------------+-------------+
   |      Zn |            30 |           3 |
   +---------+---------------+-------------+

----

Use the *dir* keyword to specify the directory containing the NNP configuration
files. The directory must contain "input.nn" with neural network
and symmetry function setup, "scaling.data" with symmetry function scaling data
and "weights.???.data" with weight parameters for each element.

The keyword *showew* can be used to turn on/off the display of extrapolation
warnings (EWs) which are issued whenever a symmetry function value is out of
bounds defined by minimum/maximum values in "scaling.data". An extrapolation
warning may look like this:

.. code-block:: none

   ### NNP EXTRAPOLATION WARNING ### STRUCTURE:      2 ATOM:     36 SYMFUNC:   14 VALUE:  8.978E-02 MIN:  3.900E-08 MAX:  8.888E-02

stating that the value 8.978E-02 of symmetry function 14 was out of bounds
(maximum in "scaling.data" is 8.888E-02) for atom 36. Here, the structure index
refers to the MPI rank.

.. note::

   The *showew* keyword should only be set to *yes* for debugging purposes.
   Extrapolation warnings may add lots of overhead as they are communicated each
   timestep. Also, if the simulation is run in a phase-space region where the NNP
   was not correctly trained, lots of extrapolation warnings may clog log files and
   the console. In a production run use *showewsum* instead.

The keyword *showewsum* can be used to get an overview of extrapolation warnings
occurring during an MD simulation. The argument specifies the interval at which
extrapolation warning summaries are displayed and logged. An EW summary may look
like this:

.. code-block:: none

   ### NNP EW SUMMARY ### TS:        100 EW         11 EWPERSTEP  1.100E-01

Here, at timestep 100 the occurrence of 11 extrapolation warnings since the last
summary is reported, which corresponds to an EW rate of 0.11 per timestep.
Setting *showewsum* to 0 deactivates the EW summaries.

A maximum number of allowed extrapolation warnings can be specified with the
*maxew* keyword. If the number of EWs exceeds the *maxew* argument the
simulation is stopped. Note however that this is merely an approximate threshold
since the check is only performed at the end of each timestep and each MPI
process counts individually to minimize communication overhead.

The keyword *resetew* alters the behavior of the above mentioned *maxew*
threshold. If *resetew* is set to *yes* the threshold is applied on a
per-timestep basis and the internal EW counters are reset at the beginning of
each timestep. With *resetew* set to *no* the counters accumulate EWs along the
whole trajectory.

If the training of a neural network potential has been performed with different
physical units for length and energy than those set in LAMMPS, it is still
possible to use the potential when the unit conversion factors are provided via
the *cflength* and *cfenergy* keywords. If for example, the NNP was
parameterized with Bohr and Hartree training data and symmetry function
parameters (i.e. distances and energies in "input.nn" are given in Bohr and
Hartree) but LAMMPS is set to use *metal* units (Angstrom and eV) the correct
conversion factors are:

.. code-block:: none

   cflength 1.8897261328

   cfenergy 0.0367493254

Thus, arguments of *cflength* and *cfenergy* are the multiplicative factors
required to convert lengths and energies given in LAMMPS units to respective
quantities in native NNP units (1 Angstrom = 1.8897261328 Bohr, 1 eV =
0.0367493254 Hartree).

----

Restrictions
^^^^^^^^^^^^

Currently it is unclear whether this pair style can work in conjunction with
other interactions (`pair_hybrid <https://lammps.sandia.gov/doc/pair_hybrid.html>`_).

Related commands
^^^^^^^^^^^^^^^^

`pair_coeff <https://lammps.sandia.gov/doc/pair_coeff.html>`_


`units <https://lammps.sandia.gov/doc/units.html>`_

Default
^^^^^^^


The default options are *dir* = "nnp/", *showew* = yes, *showewsum* = 0, *maxew* = 0, *resetew* = no,
*cflength* = 1.0, *cfenergy* = 1.0.

----

.. [1] Behler, J.; Parrinello, M. Generalized Neural-Network Representation of
   High-Dimensional Potential-Energy Surfaces. Phys. Rev. Lett. 2007, 98 (14),
   146401. https://doi.org/10.1103/PhysRevLett.98.146401

.. [2] https://github.com/CompPhysVienna/n2p2

.. [3] Singraber, A.; Morawietz, T.; Behler, J.; Dellago, C. Parallel
   Multistream Training of High-Dimensional Neural Network Potentials. J. Chem.
   Theory Comput. 2019, 15 (5), 3075–3092. https://doi.org/10.1021/acs.jctc.8b01092
