.. _NNTSSD:

NNTSSD
==========

The purpose of the ``NNTSSD`` tool is to analyse the training set size dependence of residual error of neural network potenials. It is specifically written to be used with ``n2p2``.


Structure:
----------

``NNTSSD`` provides a class consisting of four main methods:

#. ``create_training_datasets`` creates training datasets of different size from a given original dataset using the tool :ref:`nnp-select`.
#. ``training_neural_network`` trains the neural network with different existing datasets using :ref:`training`.
#. ``analyse_learning_curves`` prepares anaylse data from the learning curves obtained in training.
#. ``plot_size_dependence`` plots the energy and forces RMSE versus training set size.

There are some additional functionalities that might be useful.

* ``perform_NNTSSD`` performs NNTSSD methods according to user-given specifications.
  It uses the module ``file_input`` or ``interactive_input`` to read user-given parameters.


Build instructions:
-------------------

To use the ``NNTSSD`` tool, please first build ``n2p2`` according to the build instructions on the home page of this documentation.

The code for the ``NNTSSD`` tool is written in python3_ and uses the packages os_, sys_, shutil_, numpy_ and matplotlib_.

Depending on which methods shall be executed, there are some files and folders expected to be located in the current working directory. Please see the code documentation for further information.


Usage:
------

*
   The four main methods listed above can of course easily be used by calling them, eg. in a python3 console, or can be runned automatically in a short script.


*
   However, for some applications it might be convenient to use the function ``perform_NNTSSD`` which performs NNTSSD methods according to user-given specifications. Firstly, it reads the NNTSSD parameters either from the file ``NNTSSD_input.dat`` using the module ``file_input.dat`` or, if not successful, from interactive user input. Secondly, it performs a user-given selection of the main four NNTSSD methods listed above.
   For reading the input file, the method ``file_input.py`` is used. It requires a file ``NNTSSD_input.dat`` of the following self-explaining form:

   .. code-block:: none

       # INPUT SPECIFICATIONS FOR RUNNING NNTSSD:
       ################################################################################
       # Specify which NNTSSD.Tools steps shall be performed:
       ################################################################################
       # Create datasets/training/analyse learning curves/plot size dependence?
       # (y/n, separated by blank space):
       y y y y
       ################################################################################
       # Specify parameters for NNTSSD.Tools.create_training_datatsets:
       ################################################################################
       # Give a list of the desired set size ratios, separated by blank space:

       # Alternatively: Give minimum, maximum and step size of desired ratios,
       # separated by blank space (used if above is empty):
       0.8 0.9 0.1
       # Give the number of sample datasets per training size:
       2
       # Do you wish to fix the random generator seed to a specific value? (y/n)
       y
       # If yes, give the random generator seed (optional, default is 123):
       123
       ################################################################################
       # Specify parameters for NNTSSD.Tools.training_neural_network:
       ################################################################################
       # Give the desired number of training epochs:
       4
       # Give the number of cores you want to use for mpirun:
       4
       # Write a VSC submission script (if not, training is performed on your machine)?
       n
       # If yes, give the maximum time required for exectuing job (hh:mm:ss, optional):
       00:10:00
       # Do you wish to fix the random generator seed to a specific value? (y/n)
       y
       # If yes, give the random generator seed (optional, default is 123):
       789

Example:
---------

There is a short example prepared. In the directory ``NNTSSD/example/``, run the following command

  .. code-block:: none

     python3 ../source/NNTSSD.py


Sample screen output:
---------------------

.. code-block:: none
   
    **********************************************************************
    NNTSSD - Tools for Neural Network Training Set Size Dependence
    **********************************************************************
    Performing the following NNTSSD steps:
       True 	 Create training datasets
       True 	 Training neural network
       True 	 Analyse learning curves
       True 	 Plot size dependence

    ***CREATING TRAINING DATASETS**********************************************
    number of samples per training set size =  2
    number of different training set sizes =  2
    We are working with ratio 0.80
    ../../../../bin/nnp-select random 0.80 123
    ...
    ../../../../bin/nnp-select random 0.80 123
    ...
    We are working with ratio 0.90
    ../../../../bin/nnp-select random 0.90 123
    ...
    ../../../../bin/nnp-select random 0.90 123
    ...
    FINISHED creating datasets.

    ***TRAINING NEURAL NETWORK*************************************************
    We are working with ratio 0.80
     mpirun -np 4 ../../../../../../../bin/nnp-train
     ...
     mpirun -np 4 ../../../../../../../bin/nnp-train
     ...
    We are working with ratio 0.90
     mpirun -np 4 ../../../../../../../bin/nnp-train
     ...
     mpirun -np 4 ../../../../../../../bin/nnp-train
     ...
    FINISHED training with  2  different ratios.

    ***ANALYSING LEARNING CURVES***********************************************
       Analysing data at epoch of minimum energy
    We are working with ratio 0.80
    We are working with ratio 0.90
       Analysing data at epoch of minimum force
    We are working with ratio 0.80
    We are working with ratio 0.90
    FINISHED analysing learning curves.

    ***PLOTTING SIZE DEPENDENCE************************************************
       Plotting size dependence.
    FINISHED plotting size dependence.

Where the three dots refer to screen outputs from :ref:`nnp-select` or :ref:`training`.


File output:
------------

``NNTSSD`` creates a directory ``Output`` containing a 2-layered directory structure and various files. See the code documentation for explanation of the specific files each method creates.
The final results are plotted and saved as *.png files

* ``Output/Energy_RMSE.png``\ : Shows train and test energy RMSE (and its standard deviation) versus training set size.
* ``Output/Forces_RMSE.png`` : Shows train and test forces RMSE (and its standard deviation) versus training set size.

.. _python3: https://www.python.org/
.. _os: https://docs.python.org/3/library/os.html
.. _sys: https://docs.python.org/3/library/sys.html
.. _shutil: https://docs.python.org/3/library/shutil.html
.. _numpy: https://www.numpy.org/
.. _matplotlib: https://matplotlib.org/
