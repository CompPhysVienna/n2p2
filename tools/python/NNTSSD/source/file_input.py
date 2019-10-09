#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tools for reading the NNTSSD parameters from file 'NNTSSD_input.dat'.
"""

import numpy as np
import sys

def convert_yes_no_to_logical(yn_input):
    """This function converts 'y' and 'n' into the logicals True or False.
        
    Parameters
    ----------
    yn_input : string
        Should contain 'y' or 'n'.
    
    Returns
    -------
    logical : logical
        True if the input string has been 'y'.
        False if the input string has been 'n'.
        
    Raises
    ------
    ValueError
        If input string does not contain either 'y' or 'n'.
    """
    if yn_input == "y":
        logical = True
    elif yn_input == "n":
        logical = False
    else:
        raise ValueError("Please input 'y' or 'n' for a (y/n) question.")
#        sys.exit("ERROR: Please input 'y' or 'n' for a (y/n) question.")
    return logical

def read_parameters_from_file():
    """This function reads the NNTSSD parameters from the file 'NNTSSD_input.dat'.
    
    Requirements
    ------------
    'NNTSSD_input.dat' : file
        Contains user-given specifications on training set size parameters and NNTSSD steps.
        
    Returns
    -------
    create_logical : logical
        True if the function NNTSSD.Tools.create_training_datasets() shall be performed. False otherwise.
    train_logical : logical
        True if the function NNTSSD.Tools.training_neural_network() shall be performed. False otherwise.
    analyse_logical : logical
        True if the function NNTSSD.Tools.analyse_learning_curves() shall be performed. False otherwise.
    plot_logical : logical
        True if the function NNTSSD.Tools.plot_size_dependence() shall be performed. False otherwise.
    set_size_ratios : numpy.ndarray
        One dimensional array; contains a list of ratios of the original training set size that are examined.
    n_sets_per_size : int
        Value, specifies how many sample sets per training size are considered.
    fix_random_seed_create : logical
        True if random seed for nnp-select shall be fixed, False otherwise.
    random_seed_create : integer, optional
        User-given fixed random number generator seed. Default is 123.
    write_submission_script_logical : logical
        If True, a job submission script for VSC is written in each of the folders 'Output/ratio*/ratio*_**/'.
        If False, the command nnp_train is executed right away.
    fix_random_seed_train : logical
        True if random seed for nnp-train shall be fixed, False otherwise.
    random_seed_train : integer, optional
        User given fixed random number generator seed. Default is 123.
    maximum_time: string, optional
        User given maximum time required for executing the VSC job. Default is None.
    mpirun_cores : integer
        Number of cores that shall be used for executing the training.
    n_epochs : integer
        Number of training epochs.
    """
    content = []
    user_input  = open("NNTSSD_input.dat")
    for line in user_input:
        if not line.strip().startswith("#"):
            content.append(line.rstrip().split())
    create_logical = convert_yes_no_to_logical(content[0][0])
    train_logical = convert_yes_no_to_logical(content[0][1])
    analyse_logical = convert_yes_no_to_logical(content[0][2])
    plot_logical = convert_yes_no_to_logical(content[0][3])
    External_test_logical = convert_yes_no_to_logical(content[1][0])
    External_analyse_logical = convert_yes_no_to_logical(content[1][1])
    External_plot_logical = convert_yes_no_to_logical(content[1][2])
    Validation_predict_logical = convert_yes_no_to_logical(content[2][0])
    Validation_plot_logical = convert_yes_no_to_logical(content[2][1])
    if create_logical:
        set_size_ratios = np.array([float(x) for x in content[3]])
        if len(set_size_ratios) == 0:
            set_size_ratios = np.arange(float(content[4][0]),float(content[4][1])+float(content[4][2]),float(content[4][2]))
        n_sets_per_size = abs(int(content[5][0]))
        fix_random_seed_create = convert_yes_no_to_logical(content[6][0])
        if fix_random_seed_create:
            try:
                random_seed_create = abs(int(content[7][0]))
            except:
                random_seed_create = None
        else:
            random_seed_create = None
    else:
        set_size_ratios = None
        n_sets_per_size = None
        fix_random_seed_create = None
        random_seed_create = None
    if train_logical:
        n_epochs = abs(int(content[8][0]))
        test_fraction = float(content[9][0])
        mpirun_cores = abs(int(content[10][0]))
        write_submission_script_logical = convert_yes_no_to_logical(content[11][0])
        if write_submission_script_logical:
            try:
                maximum_time = str(content[12][0])
            except:
                maximum_time = None
        else:
            maximum_time = None
        fix_random_seed_train = convert_yes_no_to_logical(content[13][0])
        if fix_random_seed_train:
            try:
                random_seed_train = abs(int(content[14][0]))
#                print(random_seed_train)
            except:
                random_seed_train = None
        else:
            random_seed_train = None
    else:
        n_epochs = None
        test_fraction = None
        mpirun_cores = None
        write_submission_script_logical = None
        maximum_time = None
        fix_random_seed_train = None
        random_seed_train = None
    if External_test_logical:
        dataset_cores = abs(int(content[15][0]))
        fix_random_seed_predict = convert_yes_no_to_logical(content[16][0])
        if fix_random_seed_predict:
            try:
                random_seed_predict = abs(int(content[17][0]))
            except:
                random_seed_predict = None
        else:
            random_seed_predict = None
    else:
        dataset_cores = None
        fix_random_seed_predict = None
        random_seed_predict = None
    if Validation_predict_logical:
        validation_dataset_cores = abs(int(content[18][0]))
        fix_random_seed_validation = convert_yes_no_to_logical(content[19][0])
        if fix_random_seed_validation:
            try:
                random_seed_validation = abs(int(content[20][0]))
            except:
                random_seed_validation = None
        else:
            random_seed_validation = None
    else:
        validation_dataset_cores = None
        fix_random_seed_validation = None
        random_seed_validation = None
    user_input.close()
    return create_logical,train_logical,analyse_logical,plot_logical,\
    External_test_logical,External_analyse_logical,External_plot_logical,\
    Validation_predict_logical,Validation_plot_logical,\
    set_size_ratios,n_sets_per_size,fix_random_seed_create,random_seed_create,\
    n_epochs,test_fraction,mpirun_cores,write_submission_script_logical,maximum_time,fix_random_seed_train,random_seed_train,\
    dataset_cores,fix_random_seed_predict,random_seed_predict,\
    validation_dataset_cores,fix_random_seed_validation,random_seed_validation
