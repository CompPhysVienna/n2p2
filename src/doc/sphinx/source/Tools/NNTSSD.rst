.. _NNTSSD:

NNTSSD
==========

The purpose of the ``NNTSSD`` tool is to analyse the training set size dependence of residual error of neural network potenials. It is specifically written to be used with ``n2p2``.

Structure:
----------

``NNTSSD`` consists of four main methods:

#. ``create_training_datasets`` creates training datasets of different size from a given original dataset using the tool :ref:`nnp-select`.
#. ``training_neural_network`` trains the neural network with different existing datasets using the program :ref:`training`.
#. ``analyse_learning_curves`` prepares anaylse data from the learning curves obtained in training.
#. ``plot_size_dependence`` plots the energy and forces RMSE versus training set size.


Build instructions:
-------------------

The code is written in 


Depending on which parts shall be executed, there are some files and folders expected to be located in the current working directory. Please see the code documentation for further information.

.. code-block:: none

   Creates training datasets from a given original dataset using the program nnp-select.
        
        Requirements
        ----------
        'input.data' : file
            Contains original set of trainingdata.
        ../../../../bin/nnp-select : executable program
            Performs random selection of sets according to given ratio.
            
        Outputs
        ----------
        'Output' : folder
            It contains all of the following outputs.
        'Output/ratio*' : folders
            Its name tells the ratio * of current from original dataset.
        'Output/ratio*/ratio*_**' : subfolders of the previous
            Its name in addition tells the sample number ** of its ratio *.
        'Output/ratio*/ratio*_**/input.data' :  file
            Contains new training dataset of specified size ratio.
        'Output/ratio*/ratio*_**/nnp-select.log' : file
            Log file created by running nnp-select.



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
* ``nnp-select.log`` : Log file (copy of screen output).

.. _python3: https://www.python.org/
