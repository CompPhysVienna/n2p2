.. _training:

NNP training procedure
======================

.. warning::

   This part of the documentation is incomplete!

Creating a new high-dimensional neural network potential from scratch is
usually a multi-step iterative procedure which includes

* data set generation/augmentation,
* selection/refinement/pruning of symmetry function parameters,
* experimenting with neural network topology/settings,
* the actual neural network training, i.e. optimizing the weight parameters.

Tools are ready-made for most of these tasks and can be combined to generate a
new NNP from scratch. Here is a rough guideline of the individual steps from a
set of configurations to an initial NN potential. However, be aware that
creating a reliable NNP usually requires repeated data set refining and adequate
testing before it is ready for production!

Step 1: Data set
""""""""""""""""

Prepare a data set (name the file ``input.data``) in the file format described
:ref:`here <cfg_file>`. Unfortunately there's no simple recipe how to create a
"good" set of configurations, you will find more hints in the literature. As a
starting point you could try with some (100+) configurations taken from an *ab
initio* MD simulation.

Step 2: NN and symmetry function setup
""""""""""""""""""""""""""""""""""""""

Prepare a settings file (name the file ``input.nn``): Use the recommended file
`here
<https://github.com/CompPhysVienna/n2p2/blob/master/examples/input.nn.recommended>`__
and change the settings in the "GENERAL NNP SETTINGS" section according to
your system. Change the symmetry function definitions in the "SYMMETRY
FUNCTIONS" section.. again this not a trivial task, please find more information
in the literature [1]_ [2]_ [3]_. See also the description of the
``symfunction_short`` keyword :ref:`here <keywords>`.

.. note::

   There is a very useful standalone Python tool written by Florian Buchner (see
   his `pull request <https://github.com/CompPhysVienna/n2p2/pull/15>`__) which
   allows to create sets of symmetry function lines following the guidelines
   given in [2]_ and [3]_. To use it, just copy the file `sfparamgen.py
   <https://github.com/flobuch/n2p2/blob/symfunc_paramgen/tools/python/symfunc_paramgen/src/sfparamgen.py>`__
   to a local directory and follow the instructions given in this `Jupyter
   notebook
   <https://github.com/flobuch/n2p2/blob/symfunc_paramgen/tools/python/symfunc_paramgen/examples/example.ipynb>`__.


Step 3: Compute symmetry function statistics
""""""""""""""""""""""""""""""""""""""""""""

With the files ``input.data`` and ``input.nn`` ready in the same directoy, run
the tool ``nnp-scaling`` (supports MPI parallelization). This will compute all
symmetry functions for all atoms once and store statistics about them in a
third file (``scaling.data``) required for training.

Step 4: NNP Training
""""""""""""""""""""

Run the actual training program ``nnp-train``, preferably in parallel via:
``mpirun -np 16 nnp-train`` Be aware of the memory footprint which is estimated
in the previous step (see end of log file or screen output).

Step 5: Collect weight files
""""""""""""""""""""""""""""
Upon training weight files are created for each epoch ``weights.???.<epoch>``.
Select an epoch with satisfying RMSE and rename the corresponding weight
files to ``weights.???.data`` (``???`` is the atomic number of elements
occurring).

Step 6: Prediction and MD simulation
""""""""""""""""""""""""""""""""""""

Try the potential by predicting energies and forces for a new configuration:
Collect the files from training (``input.nn``, ``scaling.data`` and
``weights.???``) in a folder together with a single configuration (named again
``input.data``) and run the tool ``nnp-predict``. Alternatively, try to run a MD
simulation with LAMMPS (see setup instructions :ref:`here <if_lammps>` and
:ref:`here <pair_nnp>`).

Please also have a look at the ``examples`` directory which provides working
example setups for each tool. If there are problems don't hesitate to ask
again...

.. [1] Behler, J. Atom-Centered Symmetry Functions for Constructing
   High-Dimensional Neural Network Potentials. J. Chem. Phys. 2011, 134 (7),
   074106. https://doi.org/10.1063/1.3553717

.. [2] Imbalzano, G.; Anelli, A.; Giofré, D.; Klees, S.; Behler, J.; Ceriotti,
   M. Automatic Selection of Atomic Fingerprints and Reference Configurations for
   Machine-Learning Potentials. J. Chem. Phys. 2018, 148 (24), 241730.
   https://doi.org/10.1063/1.5024611

.. [3] Gastegger, M.; Schwiedrzik, L.; Bittermann, M.; Berzsenyi, F.;
   Marquetand, P. WACSF—Weighted Atom-Centered Symmetry Functions as Descriptors in
   Machine Learning Potentials. J. Chem. Phys. 2018, 148 (24), 241709.
   https://doi.org/10.1063/1.5019667
