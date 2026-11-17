.. _changes_4g:

Changes to the 4G-HDNNP implementation
======================================

This page lists the defects of the 4G-HDNNP implementation that were fixed
after the merge of the 4G prediction branch (pull request #77, commit
``29b9c9f1``, March 2025), and the main speed-ups. Each entry names the pull
request with the full description and the tests added with it.

Do I need to retrain or rerun?
------------------------------

* **Models trained with normalized units** (``nnp-norm``, ``normalize_data_set``):
  retrain. Their Coulomb interaction was practically absent during training
  (#222).
* **4G stage 2 trained with** ``memorize_symfunc_results`` **off** (the default):
  the force updates and training errors missed the :math:`\partial\chi/\partial
  r` term (#211, #227). Retraining is recommended.
* **Training sets with structures of different volume or number of atoms:**
  the Ewald sums used the parameters of the wrong structure (#229).
* **LAMMPS results with** ``pair_style hdnnp/4g``: rerun. Before #212, #222 and
  #228, energies, charges and forces could differ from ``nnp-predict``.
  Non-periodic systems did not work before #235.
* **NVE runs** with ``neighbor 0.0`` and ``neigh_modify delay 10`` (the input of
  the earlier water examples): rerun with the settings in
  :ref:`if_lammps` (#230).

2G-HDNNPs are affected only by the entries marked "2G and 4G".

Fixed defects
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 45 10 15

   * - Defect
     - Effect
     - Affected
     - Fixed in
   * - Stage-2 force updates missed the :math:`\partial\chi/\partial r` term
       when the charge equilibration was reused (``memorize_symfunc_results``
       off).
     - Force derivatives wrong by 7e-3 to 9e-3 eV/Å, up to about 20 % of
       :math:`|F|` (H2O_RPBE-D3_4G).
     - training, stage 2
     - #211
   * - The same term was missing in the training errors when an epoch ended
       inside a group of force updates.
     - Training-set force RMSE too large, which changed the update candidates
       of the next epoch.
     - training, stage 2
     - #227
   * - The 4G charge equilibration used 4 pi epsilon = 1 instead of the
       ``four_pi_epsilon`` setting, also for normalized models.
     - Correct only for atomic units without normalization; for normalized
       models the Coulomb interaction was weaker by about 6.5e4.
     - training, prediction
     - #222
   * - The Ewald parameters (eta, cutoffs) of the last structure set up were
       used for all structures.
     - Wrongly truncated Ewald sums for training sets with different volumes;
       division by zero for small cells.
     - training
     - #229
   * - An element missing in ``initial_hardness`` silently kept a hardness of
       zero; a restarted stage 1 reset the hardness; ``nnp-scaling`` skipped the
       normalization setup.
     - The hardness of such an element could never be trained; restarts did not
       continue from the trained parameters.
     - training, stage 1
     - #216
   * - ``normalize_data_set`` in stage 2 was accepted although the
       normalization is not consistent with the electrostatics.
     - Abort in an Eigen assertion, or inconsistent files. Now a clear error.
     - training, stage 2
     - #217
   * - Periodic self-images were skipped in the short-range force assembly
       (cells shorter than the cutoff).
     - Forces not the gradient of the energy (1e-3 relative for a water
       molecule in a 10 bohr box).
     - 2G and 4G, prediction
     - #221
   * - The training update counter was not initialized; an MPI process without
       training data crashed.
     - Arbitrary counter in ``train-log.out``; uncaught ``std::out_of_range``.
       Now a clear error.
     - 2G and 4G, training
     - #224, #226
   * - LAMMPS: the real-space electrostatics were truncated at the pair cutoff.
     - Charges off by up to 0.19 e, energy by 0.16 eV, forces by 2.4e-2 eV/Å
       (AlAuMgO_4G).
     - LAMMPS
     - #212
   * - LAMMPS: the reciprocal-space energy was not added to the total energy;
       the charge and lambda minimizers kept iterating without progress.
     - Wrong total energy; iterations without progress up to ``maxit`` (85 of
       100 for AlAuMgO_4G), 4 times slower MD.
     - LAMMPS
     - #212
   * - LAMMPS: normalized models, unit conversion (``four_pi_epsilon``) and
       element mappings not in the order of atomic numbers were handled wrongly.
     - Wrong energies (1.7 eV for a water box), parameters assigned to the wrong
       elements, or a segmentation fault.
     - LAMMPS
     - #222
   * - LAMMPS: charges and force lambda were solved by a minimizer that stopped
       at about 1e-8 relative gradient.
     - Noisy potential energy (up to 1e-3 eV for 630 atoms), failing
       finite-difference force checks. Now solved by conjugate gradients.
     - LAMMPS
     - #228
   * - LAMMPS: non-periodic systems (``boundary f f f``).
     - Crash in the first step; no charge-response forces; energy off by
       199 eV for a 12-atom chain; heap corruption.
     - LAMMPS
     - #235
   * - LAMMPS: the pair style's charge equilibration arrays were freed after
       the pair style itself was deleted.
     - Use-after-free at the end of every run (usually unnoticed).
     - LAMMPS
     - #236
   * - The water examples used a zero neighbor skin with rebuilds every 10
       steps.
     - Energy drift of 0.003 to 0.1 meV/atom/ps in NVE.
     - LAMMPS examples
     - #212, #230
   * - The 4G stage-2 inputs of ``examples/nnp-train/Carbon_chain_4G`` did not
       match its ``input.nn``.
     - ``nnp-train`` stopped with a weight count error.
     - example
     - #219

Speed-ups
---------

.. list-table::
   :header-rows: 1
   :widths: 55 30 15

   * - Change
     - Part
     - Pull request
   * - Ewald reciprocal part of the charge equilibration matrix as two rank
       updates; the matrix factorized once per structure; charge error
       derivatives summed once per atom; repeated force and charge calculations
       skipped
     - training
     - #218
   * - Symmetry functions kept between the force update trials of a structure
     - training, stage 2
     - #225
   * - Neural network input derivatives in one pass; pair forces collected from
       the neighbor side; faster electrostatic forces
     - training, prediction
     - #220
   * - O(N²) scaling recovered in the electrostatic forces and derivatives
     - LAMMPS
     - #212
   * - Structure factors reduced with one ``MPI_Allreduce``; Ewald tables read
       row by row; conjugate gradients instead of a minimizer
     - LAMMPS
     - #223, #233, #228

On one CPU node, 4G training is 24 to 32 times faster per epoch in stage 1 and
6.5 to 11 times faster in stage 2 than at ``29b9c9f1`` (SPC/Fw water,
1254 structures, three machines).

New tests
---------

The test suite checks the 4G implementation against independent results, see
:ref:`validation`. Most fixes above add a test that fails without them.
