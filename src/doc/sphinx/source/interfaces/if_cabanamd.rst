.. _if_cabanamd:

CabanaMD interface
==================

Purpose
-------
The CabanaMD interface adds the neural network potential method in a proxy
application which uses the Cabana particle library. `Cabana
<https://github.com/ECP-copa/Cabana>`_ uses the `Kokkos
<https://github.com/kokkos/kokkos>`_ programming model to run on multi-core CPUs
and GPUs; `CabanaMD <https://github.com/ECP-copa/CabanaMD>`_ provides a simple
MD code to explore performance with Cabana and Kokkos. The Cabana version of
n2p2 reimplements a small part of the neural network potential to enable
simulations on the GPU [1]_.


Build instructions
------------------

Go to the ``src`` directory and compile with the CabanaMD interface enabled:

.. code-block:: none

   cd src
   make libnnpif INTERFACES=CabanaMD

Alternatively, set the ``INTERFACES`` variable in the master makefile
(``src/makefile``) to ``CabanaMD`` and just run in the ``src`` directory:

.. code-block:: none

   make libnnpif

For dynamic linking add the argument ``MODE=shared``.

.. note::
   If dynamic linking (\ ``make libnnpif MODE=shared``\ ) is used, you need to make the NNP
   libraries visibile in your system, e.g. add this line in your ``.bashrc``\ :

   .. code-block:: none

      export LD_LIBRARY_PATH=<path-to-n2p2>/lib:${LD_LIBRARY_PATH}

This completes the necessary steps on the n2p2 side, further instructions on how
to build CabandaMD together with this interface are provided `here
<https://github.com/ECP-copa/CabanaMD/wiki>`__.


Usage
-----

See `here <https://github.com/ECP-copa/CabanaMD/wiki>`__ for building and running
CabanaMD after building n2p2 (with the CabanaMD interface).

.. [1] Desai, S.; Reeve, S. T.; Belak, J. F. Implementing a Neural Network
   Interatomic Model with Performance Portability for Emerging Exascale
   Architectures. `arXiv:2002.00054 <https://arxiv.org/abs/2002.00054>`__
   [cond-mat, physics:physics] 2020
