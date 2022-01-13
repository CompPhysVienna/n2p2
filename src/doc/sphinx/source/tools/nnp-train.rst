.. _nnp-train:

nnp-train
=========

.. warning::

   Documentation in progress...

This tool implements the actual training procedure for a given data set. It is
able to train both 2G and 4G [1]_ neural networks (NN). In the latter case the
training procedure consists of two training stages. Stage 1 needs to be done
first. It is the training of the charge NNs. After this is finished one can go
to stage 2 which consists of training the short-ranged NNs by fitting the data
to energy and forces.

Requirements:
-------------
* ``input.data``
* ``input.nn``
* ``scaling.data``

Additionally for stage 2 in the 4G case:

* ``hardness.???.data``
* ``weightse.???.data``

Usage:
-----------

When training a 2G NN the following command will train a NN with the given
topology in ``input.nn`` for the given data set with 4 MPI tasks

.. code-block:: none

   mpirun -np 4 nnp-train

If one has specified a 4G NN the command is

.. code-block:: none

   mpirun -np 4 nnp-train <n>

where ``<n>`` is the stage (1 or 2). After finishing stage 1 one has to choose
the NN architecture of the preferred training epoch. Usually one picks the epoch
with the lowest RMSE in the training set but there may be reasons to deviate
from that rule. After deciding for epoch ``<m>`` one has to rename the files
``hardness.???.<m>.out`` and ``weightse.???.<m>.out`` to ``hardness.???.data``
and ``weightse.???.data``, respectively.
When the training is finished (after stage 2 with the 4G NN or after the
training with the 2G NN) it is again necessary to pick an epoch ``<m>`` of this
run and rename the files ``weights.???.<m>.out`` to ``weights.???.data``.

Sample screen output:
---------------------

A typical stage 1 training with a 4G network looks like this:

.. code-block:: none

   .
   .
   .
   *** TRAINING LOOP *************************************************************

   The training loop output covers different errors, update and
   timing information. The following quantities are organized
   according to the matrix scheme below:
   -------------------------------------------------------------------
   ep ........ Epoch.
   Q_count ... Number of charge updates.
   Q_train ... RMSE of training charges.
   Q_test .... RMSE of test     charges.
   Q_pt ...... Percentage of time for charge updates w.r.t. to t_train.
   count ..... Total number of updates.
   train ..... Percentage of time for training.
   error ..... Percentage of time for error calculation.
   other ..... Percentage of time for other purposes.
   epoch ..... Total time for this epoch (seconds).
   total ..... Total time for all epochs (seconds).
   -------------------------------------------------------------------
   charge     ep  Q_count       Q_train        Q_test    Q_pt
   timing     ep    count  train  error  other      epoch      total
   -------------------------------------------------------------------
   CHARGE      0        0   2.30301E-01   2.75350E-01     0.0
   TIMING      0        0    0.0   58.3   41.7       0.09       0.09
   ------
   CHARGE      1        4   1.64420E-02   1.25118E-02   100.0
   TIMING      1        4   92.8    3.8    3.3       0.41       0.49
   ------
   CHARGE      2        4   8.13293E-03   4.64616E-03   100.0
   TIMING      2        4   91.7    4.5    3.9       0.34       0.83
   ------
   .
   .
   .
   ------
   CHARGE     10        4   3.41430E-03   2.22138E-03   100.0
   TIMING     10        4   90.0    5.0    5.0       0.38       3.65
   -------------------------------------------------------------------------------
   TIMING Training loop finished: 3.65 seconds.
   *******************************************************************************

Whereas 2G NN training or stage 2 training with 4G NN produces something similar
to this:

.. code-block:: none

   .
   .
   .
   *** TRAINING LOOP *************************************************************
   
   The training loop output covers different errors, update and
   timing information. The following quantities are organized
   according to the matrix scheme below:
   -------------------------------------------------------------------
   ep ........ Epoch.
   E_count ... Number of energy updates.
   E_train ... RMSE of training energies per atom.
   E_test .... RMSE of test     energies per atom.
   E_pt ...... Percentage of time for energy updates w.r.t. to t_train.
   F_count ... Number of force updates.
   F_train ... RMSE of training forces.
   F_test .... RMSE of test     forces.
   F_pt ...... Percentage of time for force updates w.r.t. to t_train.
   count ..... Total number of updates.
   train ..... Percentage of time for training.
   error ..... Percentage of time for error calculation.
   other ..... Percentage of time for other purposes.
   epoch ..... Total time for this epoch (seconds).
   total ..... Total time for all epochs (seconds).
   -------------------------------------------------------------------
   energy     ep  E_count       E_train        E_test    E_pt
   force      ep  F_count       F_train        F_test    F_pt
   timing     ep    count  train  error  other      epoch      total
   -------------------------------------------------------------------
   ENERGY      0        0   1.80089E-02   1.72559E-02     0.0
   FORCE       0        0   1.76247E-01   1.93256E-01     0.0
   TIMING      0        0    0.0   82.9   17.1       0.12       0.12
   ------
   ENERGY      1        4   5.48098E-05   2.99658E-05    15.5
   FORCE       1       16   3.97965E-03   3.93252E-03    84.5
   TIMING      1       20   95.4    3.5    1.1       1.21       1.33
   ------
   ENERGY      2        4   1.62363E-05   8.82677E-06    14.1
   FORCE       2       16   3.15635E-03   2.18593E-03    85.9
   TIMING      2       20   95.2    3.6    1.1       1.23       2.56
   ------
   .
   .
   .
   ------
   ENERGY     10        4   2.47602E-05   8.54473E-06    14.2
   FORCE      10       16   8.73691E-03   1.41630E-02    85.8
   TIMING     10       20   94.5    4.0    1.5       1.23      12.35
   -------------------------------------------------------------------------------
   TIMING Training loop finished: 12.35 seconds.
   *******************************************************************************

File output:
------------

Always generated:
^^^^^^^^^^^^^^^^^

In the following ``[...]`` is a part of the filename that only exists in 4G training.

* ``learning-curve.out[.stage-<n>]``: Contains the errors of the NN after each
  epoch for all quantities that are used for this training.

* ``test.data``: Contains the data that is only used for testing but not for
  training (formatted like ``input.data``).

* ``train.data``: Contains the data that is only used for training but not for
  testing (formatted like ``input.data``).

* ``updater.???.out[.stage-<n>]``: Contains informations about the optimization
  algorithm that was used for training the NN.

* ``timing.out[.stage-<n>]``: Contains information about the time needed for
  individual tasks in the training procedure (e.g. update and error
  calculation).

Optional:
^^^^^^^^^

In 4G stage 1 (if ``write_weights_epoch`` is set non-zero):

* ``hardness.???.??????.out``
* ``weightse.???.??????.out``

In 2G and 4G stage 2 (if ``write_weights_epoch`` is set non-zero):

* ``weights.???.??????.out``

In 4G stage 1 (if ``write_traincharges`` is set non-zero):

* ``traincharges.??????.out``: Contains a comparison between the reference charges
  and the predicted charges for the data used in the training after the epoch
  denoted by ``??????``.

* ``testcharges.??????.out``: Contains a comparison between the reference charges
  and the predicted charges for the data used for testing after the epoch
  denoted by ``??????``.


In 2G or 4G stage 2 (if ``write_trainpoints`` is set non-zero):

* ``trainpoints.??????.out``: Contains a comparison between the reference energies
  and the predicted energies for the data used in the training after the epoch
  denoted by ``??????``.

* ``testpoints.??????.out``: Contains a comparison between the reference
  energies and the predicted energies for the data used for testing after the epoch
  denoted by ``??????``.

In 2G or 4G stage 2 (if ``write_trainforces`` is set non-zero):

* ``trainforces.??????.out``: Contains a comparison between the reference forces
  and the predicted forces for the data used in the training after the epoch
  denoted by ``??????``.

* ``testforces.??????.out``: Contains a comparison between the reference
  forces and the predicted forces for the data used for testing after the epoch
  denoted by ``??????``.


.. [1] Ko, T. W.; Finkler, J. A.; Goedecker, S.; Behler, J. A Fourth-Generation High-Dimensional Neural Network Potential with Accurate Electrostatics Including Non-Local Charge Transfer. Nature Communications 2021, 12 (1), 398. https://doi.org/10.1038/s41467-020-20427-2.
