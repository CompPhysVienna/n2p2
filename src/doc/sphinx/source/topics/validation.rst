.. _validation:

Validation of 4G-HDNNPs
=======================

Besides regression tests against stored results, the test suite checks the
4G-HDNNP implementation against results that do not come from n2p2 itself. This
page lists these checks and how to repeat them for your own model.

Checks in the test suite
------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Check
     - What it compares
     - Test
   * - Independent charge equilibration reference
     - Charges and electrostatic energy of a Na-Cl pair (closed form) and of a
       periodic 64-atom disordered rock-salt frame, computed by an independent
       script (``tools/misc/qeq-reference.py``); agreement about 2e-11 e
     - ``test_StructureElectrostatics``
   * - Forces against finite differences
     - Analytical forces of 2G and 4G water models against central finite
       differences of the energy, including cells shorter than the cutoff
     - ``test_ForcesFiniteDifference``
   * - LAMMPS against ``nnp-predict``
     - Energy and forces of the same structure and model in LAMMPS and in
       ``nnp-predict``: liquid water (``H2O_SPCFw_4G``, forces to 1e-5 eV/Å) and
       a non-periodic carbon chain (``Carbon_chain_4G``, forces to 1e-6 eV/Å);
       energy of a metal oxide slab (``AlAuMgO_4G``)
     - ``test_lammps_hdnnp``
   * - LAMMPS forces against finite differences
     - Forces of ``H2O_SPCFw_4G`` against finite differences of the LAMMPS
       energy (``md-fd.lmp``), tolerance 3e-5 eV/Å
     - ``test_lammps_hdnnp``
   * - Units
     - The same model in normalized units and in the units of the data gives
       the same energy, charges and forces (``H2O_SPCFw_4G`` and
       ``H2O_SPCFw_4G_metal``)
     - ``test_Prediction``, ``test_lammps_hdnnp``

Energy conservation
-------------------

In NVE runs with the settings recommended in :ref:`if_lammps` (neighbor skin
1 Å with a check in every step, relative ``kspace_style hdnnp`` accuracy 1e-5),
4G-HDNNPs of liquid water conserve energy as well as a 2G-HDNNP: the drift over
50 ps at a 0.5 fs time step is below 1e-4 meV/atom/ps, and the fluctuation is
about 0.013 meV/atom for 630 atoms.

Checking your own model
-----------------------

* **Charges:** ``nnp-dataset`` writes predicted and reference charges to
  ``charges.comp``; with the charges-only mode it runs stage 1 alone.
* **LAMMPS against** ``nnp-predict``: run one structure as a single point in both
  and compare energy, charges and forces. Differences of the order of 1e-6 eV/Å
  in the forces come from the different Ewald splitting of the two codes and
  shrink with a tighter ``kspace_style hdnnp`` accuracy.
* **Forces against finite differences in LAMMPS:** follow
  ``examples/interface-LAMMPS/H2O_SPCFw_4G/md-fd.lmp``, which displaces atoms by
  ±1e-3 Å and prints the energies.
* **Energy conservation:** a short NVE run with the settings above; a drift of
  more than about 1e-3 meV/atom/ps points to the neighbor settings or the
  ``kspace_style hdnnp`` accuracy.
