.. _NNTSSD:

NNTSSD
==========

The purpose of the ``NNTSSD`` tool is to analyse the training set size dependence of residual error of neural network potenials. It is specifically written to be used with ``n2p2``.


Structure:
----------

``NNTSSD`` provides a three classes consisting of two to four main methods:

#. ``NNTSSD.Tools`` class:
	* ``create_training_datasets`` creates training datasets of different size from a given original dataset using the tool :ref:`nnp-select`.
	* ``training_neural_network`` trains the neural network with different existing datasets using :ref:`training`.
	* ``analyse_learning_curves`` prepares anaylse data from the learning curves obtained in training.
	* ``plot_size_dependence`` plots the energy and forces RMSE versus training set size.
#. ``NNTSSD.Validation`` class:
	* ``predict_validation_data`` predicts the energies and forces of a validation dataset for all trained neural networks.
	* ``plot_validation_data`` plots the energy and force RMSE versus training set size.
#. ``NNTSSD.External_Testdata`` class:
	* ``predict_test_data`` predicts the energies and forces of a test dataset for all trained neural networks.
	* ``analyse_learning_curves`` prepares analyse data from the learning curves obtained in training neural network and predicting testdata.
	* ``plot_test_size_dependence`` plots the energy and force RMSE versus training set size.

* ``perform_NNTSSD`` performs NNTSSD methods according to user-given specifications.
  It uses the module ``file_input`` to read user-given parameters.

For more detailed information about the classes and methods, please see the code documentation for ``NNTSSD``.

There are some additional functionalities provided in the following programs that might be useful.

* ``split_data.py`` splits the file input.data in two files with respect to given fraction. It may be used to create an external testset, validation dataset, or both.
* ``file_input.py`` contains tools for reading the NNTSSD parameters from file ``NNTSSD_input.dat`` (see ``Usage`` for further information).
* ``learning_curves.py`` provides tools for analysing learning curves and plotting training performance. By default, learning curves obtained from external testdata are used. If they are not available, learning curves obtained from internal testdata are used.


Build instructions:
-------------------

To use the ``NNTSSD`` tool, please first build ``n2p2`` according to the build instructions on the home page of this documentation.

The code for the ``NNTSSD`` tool is written in python3_ and uses the packages os_, sys_, shutil_, numpy_ and matplotlib_.

The code documentation for NNTSSD can be built by executing

  .. code-block:: none

     make html

in the directory ~/n2p2/tools/python/NNTSSD/docs/.

Depending on which methods shall be executed, there are some files and folders expected to be located in the current working directory. Please see the code documentation for further information.


Guidance:
---------

A rough guidance of how ``NNTSSD`` is designed to be used. We assume there exists one large original dataset that shall be used for analysing the training set size dependence of residual error of ``n2p2``:

#. (If validation dataset and/or external testdata shall be used) Use ``split_data.py`` to split the original dataset in two/three datasets: the ``training dataset`` (which automatically includes the internal testdata), the ``validation dataset`` and/or the ``external testdata``.
#. Copy the ``training dataset`` renamed as ``input.data`` into a directory of desired name that must be located in ``~/n2p2/tools/python/NNTSSD/`` (in the following, this directory will be refered to as ``working directory``). In the ``working directory``, run the method ``Tools.create_training_datasets``. (A user friendly way how to "run" ``NNTSSD`` methods is described in the next section ``Usage``. Basically, just fill in the file ``NNTSSD_input.dat`` as desired and run ``NNTSSD.py`` in the ``working directory``)
#. Copy the ``n2p2`` files ``input.nn`` and ``scaling.data`` into the ``working directory`` and run the method ``Tools.training_neural_network``.
#. (If desired) Run the methods ``Tools.analyse_learning_curves`` and ``Tools.plot_size_dependence``. The analysis is done using internal testsets only (no external testset, no validation dataset).
#. (If an external testset shall be used) Create a directory ``predict_test_data`` inside the ``working directory``. Into it, copy the ``external testdata`` renamed as ``input.data`` and the files ``input.nn``and ``scaling.data``. In the ``working directory``, run the methods ``External_Testdata.predict_test_data``, ``External_Testdata.analyse_learning_curves`` and ``External_Testdata.plot_test_size_dependence``.
#. (If a validation dataset shall be used) Create a directory ``validation_data`` inside the ``working directory``. Into it, copy the ``validation dataset`` renamed as ``input.data``, as well as the files ``input.nn`` and ``scaling.data``. In the ``working directory``, run the methods ``Validation.predict_validation_data``, ``Validation.plot_validation_data``.


Usage:
------

*
   The four main methods listed above can of course easily be used by calling them, eg. in a python3_ console, or can be runned automatically in a short script.


*
   However, for some applications it might be convenient to use the function ``perform_NNTSSD`` which performs NNTSSD methods according to user-given specifications. Firstly, it reads the NNTSSD parameters from the file ``NNTSSD_input.dat`` using the module ``file_input.dat``. Secondly, it performs a user-given selection of the main NNTSSD methods listed above.
   For reading the input file, the method ``file_input.py`` is used. It requires a file ``NNTSSD_input.dat`` of the following self-explaining form:

   .. code-block:: none

       # INPUT SPECIFICATIONS FOR RUNNING NNTSSD:
       ################################################################################
       # Specify which NNTSSD.Tools steps shall be performed:
       ################################################################################
       # Create datasets/training/analyse learning curves/plot size dependence?
       # (y/n, separated by blank space):
       y y y y
       # In case of external testdata: predict data/analyse learning curves/plot SSD?
       y y y
       # Predict validation data/plot size dependence?
       y y
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
       n
       # If yes, give the random generator seed (optional, default is 123):
       123
       ################################################################################
       # Specify parameters for NNTSSD.Tools.training_neural_network:
       ################################################################################
       # Give the desired number of training epochs:
       4
       # Give the fraction of data you want to keep for testing (nnp-internal):
       0.1
       # Give the number of cores you want to use for mpirun (nnp-train):
       4
       # Write a VSC submission script (if not, training is performed on your machine)?
       n
       # If yes, give the maximum time required for exectuing job (hh:mm:ss, optional):
       00:10:00
       # Do you wish to fix the random generator seed to a specific value? (y/n)
       n
       # If yes, give the random generator seed (optional, default is 123):
       789
       ################################################################################
       # Specify parameters for NNTSSD.External_Testdata.predict_test_data:
       ################################################################################
       # Give the number of cores you want to use for mpirun (nnp-dataset):
       4
       # Do you wish to fix the random generator seed to a specific value? (y/n)
       n
       # If yes, give the random generator seed (optional, default is 123):
       789
       ################################################################################
       # Specify parameters for NNTSSD.Validation.predict_validation_data:
       ################################################################################
       # Give the number of cores you want to use for mpirun (nnp-dataset):
       4
       # Do you wish to fix the random generator seed to a specific value? (y/n)
       n
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
	   True 	Tools	 Create training datasets
	   True 	Tools	 Training neural network
	   True 	Tools	 Analyse learning curves
	   True 	Tools	 Plot size dependence
	   True 	External Testset	 Predicting external testset
	   True 	External Testset	 Analyse learning curves wrt external testset
	   True 	External Testset	 Plot size dependence wrt external testset
	   True 	Validation	 Predicting and analysing validation dataset
	   True 	Validation	 Plot size dependence wrt validation dataset

	***CREATING TRAINING DATASETS***************************************************
	   number of samples per training set size =  2
	   number of different training set sizes =  2
	   We are working with ratio 0.80
	    ../../../../bin/nnp-select random 0.80 128 >/dev/null
	    ../../../../bin/nnp-select random 0.80 978 >/dev/null
	   We are working with ratio 0.90
	    ../../../../bin/nnp-select random 0.90 918 >/dev/null
	    ../../../../bin/nnp-select random 0.90 215 >/dev/null
	   INFO: Removed old 'Output' folder.
	FINISHED creating datasets.

	***TRAINING NEURAL NETWORK*****************************************************
	   We are working with ratio 0.80
	    mpirun -np 4 ../../../../../../../bin/nnp-train >/dev/null
	    mpirun -np 4 ../../../../../../../bin/nnp-train >/dev/null
	   We are working with ratio 0.90
	    mpirun -np 4 ../../../../../../../bin/nnp-train >/dev/null
	    mpirun -np 4 ../../../../../../../bin/nnp-train >/dev/null
	FINISHED training with  2  different ratios.

	***ANALYSING LEARNING CURVES***************************************************
	   Analysing data at epoch of minimum energy
	    We are working with ratio 0.80
	    We are working with ratio 0.90
	   Analysing data at epoch of minimum force
	    We are working with ratio 0.80
	    We are working with ratio 0.90
	   Analysing data at epoch of minimum comb
	    We are working with ratio 0.80
	    We are working with ratio 0.90
	FINISHED analysing learning curves.

	***PLOTTING SIZE DEPENDENCE****************************************************
	   Plotting size dependence.
	FINISHED plotting size dependence.

	***PREDICTING EXTERNAL TESTDATA************************************************
	   We are working with ratio 0.80
	   We are working with ratio 0.90
	FINISHED predicting test data.

	***ANALYSING LEARNING CURVES W/ EXTERNAL TESTDATA******************************
	   Analysing data at epoch of minimum energy
	    We are working with ratio 0.80
	    We are working with ratio 0.90
	   Analysing data at epoch of minimum force
	    We are working with ratio 0.80
	    We are working with ratio 0.90
	   Analysing data at epoch of minimum comb
	    We are working with ratio 0.80
	    We are working with ratio 0.90
	FINISHED analysing learning curves.

	***PLOTTING TEST SIZE DEPENDENCE W/ EXTERNAL TESTDATA**************************
	   Plotting test size dependence.
	FINISHED plotting test size dependence.

	***PREDICTING VALIDATION DATA************************************************
	   Analysing data at epoch of minimum energy
	    We are working with ratio 0.80
	    We are working with ratio 0.90
	   Analysing data at epoch of minimum force
	    We are working with ratio 0.80
	    We are working with ratio 0.90
	   Analysing data at epoch of minimum comb
	    We are working with ratio 0.80
	    We are working with ratio 0.90
	FINISHED predicting validation data.

	***PLOTTING TEST SIZE DEPENDENCE W/ VALIDATION DATA****************************
	   Plotting test size dependence.
	FINISHED plotting test size dependence.


To see the ``n2p2`` screen outputs from :ref:`nnp-select`, :ref:`training` and ``nnp-dataset``, simply remove the addition ``>/dev/null`` in the ``NNTSSD`` code after ``n2p2`` commands. (The user documentation for ``nnp-dataset`` is not yet available, but will be linked here. It works very similar to :ref:`nnp-predict`, but for datasets instead of single configurations).


File output:
------------

``NNTSSD`` creates a directory ``Output`` containing a 2-layered directory structure (3-layered in case of external testdata or a validation dataset) and various files. See the code documentation for explanation of the specific files each method creates.
The final results are plotted and saved as *.png files. If no external testdata or validation dataset is used, the *.png files are

* ``Output/int_Energy_RMSE_epoch_comparison.png``\ : Shows train and test energy RMSE (and its standard deviation) versus training set size, for the case of internal testdata and comparing three epoch optimization approaches.
* ``Output/int_Forces_RMSE_epoch_comparison.png`` : Shows train and test forces RMSE (and its standard deviation) versus training set size, for the case of internal testdata and comparing three epoch optimization approaches.

For a more detailed description of which *.png files are saved if external testdata or validation datasets are used, please see the code documentation of ``NNTSSD``.


Tests:
------

Tests are prepared and can be used to check whether ``NNTSSD`` works and creates the correct outputs.
In the directory ``NNTSSD/tests/``, tun the following command

  .. code-block:: none

     python3 -m pytest

.. _python3: https://www.python.org/
.. _os: https://docs.python.org/3/library/os.html
.. _sys: https://docs.python.org/3/library/sys.html
.. _shutil: https://docs.python.org/3/library/shutil.html
.. _numpy: https://www.numpy.org/
.. _matplotlib: https://matplotlib.org/
