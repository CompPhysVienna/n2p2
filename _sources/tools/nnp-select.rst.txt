.. _nnp-select:

nnp-select
==========

The ``nnp-select`` tool extracts a small subset from a larger training data set,
e.g. for testing purposes. The initial set ``input.data`` is expected to be
located in the current working directory. Two different modes of operation are
supported: selecting structures

#. randomly with given seed, or
#. at regular intervals.

Usage:
------

.. code-block:: none

   nnp-select <mode> <arg1 <arg2>>
              <mode> ... Choose selection mode (random/interval).
              Arguments for mode "random":
                <arg1> ... Ratio of selected structures (1.0 equals 100 %).
                <arg2> ... Seed for random number generator (integer).
              Arguments for mode "interval":
                <arg1> ... Select structures in this interval (integer).
              Execute in directory with these NNP files present:
              - input.data (structure file)

Sample screen output:
---------------------

.. code-block:: none

   *** NNP-SELECT ****************************************************************

   Selecting every 3 structure.
   *******************************************************************************
   Structure       1 selected.
   Structure       4 selected.
   Structure       7 selected.
   Structure      10 selected.
   Structure      13 selected.
   Structure      16 selected.
   *******************************************************************************
   Total    structures           :      16
   Selected structures           :       6
   Selected structures percentage: 37.500 %
   *******************************************************************************

File output:
------------

* ``output.data``\ : The requested subset of training structures.
* ``reject.data``\ : The rejected configurations, i.e. all data minus ``output.data``.
* ``nnp-select.log`` : Log file (copy of screen output).

Examples:
---------

* 
  Select randomly 10% of the original set with random seed 123:

  .. code-block:: none

     nnp-select random 0.1 123

* 
  Select every 20th structure from original set (starting with structure 1):

  .. code-block:: none

     nnp-select interval 20

