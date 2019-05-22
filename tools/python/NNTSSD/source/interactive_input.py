#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tools for reading the NNTSSD parameters by interactive user input.
"""
import numpy as np

def user_input_yes_no(question):
    read = ""
    while True:
        read = input(question+" (y/n) ")
        if read == "y":
            logical = True
            break
        elif read == "n":
            logical = False
            break
        else:
            print("Please select (y/n).")
            continue
    return logical

def input_parameters_by_user():
    """This function reads the NNTSSD parameters from interactive user input.
    
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
    print("Specify which NNTSSD steps shall be performed:")
    create_logical = user_input_yes_no("  Shall training datasets be created?")
    train_logical = user_input_yes_no("  Shall neural network be trained?")
    analyse_logical = user_input_yes_no("  Shall learning curves be analysed?")
    plot_logical = user_input_yes_no("  Shall size dependence be plotted?")
    
    if create_logical:
        print("\n***INPUT for create_training_datasets*******************************")
        ratios_minmax_logical = user_input_yes_no("Do you want to give the desired ratios in the form 'min max step'\n(otherwise, you can give them as a list)?")
        ratios_loop_logical = False
        while not ratios_loop_logical:
            if ratios_minmax_logical:
                set_size_input = np.array([float(x) for x in input("  Give the minimum, maximum and step size of desired ratios, separated by blank space:\n    ").split()])
                set_size_ratios = np.arange(set_size_input[0],set_size_input[1]+set_size_input[2],set_size_input[2])
                if (set_size_ratios > 0.).all() and (set_size_ratios <= 1.).all():
                    print("    The given ratios are ",set_size_ratios)
                    ratios_loop_logical = user_input_yes_no("  Continue?")
                else:
                    print("    Make sure your ratios are in interval (0,1].")
            else:
                set_size_ratios = np.array([float(x) for x in input("  Give the desired set size ratios, separated by blank space:\n    ").split()])
                if (set_size_ratios > 0.).all() and (set_size_ratios <= 1.).all():
                    print("    The given ratios are ",set_size_ratios)
                    ratios_loop_logical = user_input_yes_no("    Continue?")
                else:
                    print("    Make sure your ratios are in interval (0,1].")
        n_sets_loop_logical = False
        while not n_sets_loop_logical:
            n_sets_per_size = int(input("Give the number of sample datasets per training size: "))
            if (n_sets_per_size >= 1):
                print("  The selected sample number per training size is ",n_sets_per_size)
                n_sets_loop_logical = user_input_yes_no("  Continue?")
            else:
                print("  Please use a positive value!")
        fix_random_seed_create = user_input_yes_no("Do you wish to fix the random generator seed to a specific value?")
        if fix_random_seed_create:
            random_seed_create = int(input("  Give the random generator seed: "))
        else:
            random_seed_create = None
    if train_logical:
        print("\n***INPUT for training_neural_network********************************")
        fix_random_seed_train = user_input_yes_no("Do you wish to fix the random generator seed to a specific value?")
        if fix_random_seed_train:
            random_seed_train = int(input("  Give the random generator seed: "))
        else:
            random_seed_train = None
        cores_loop_logical = False
        while not cores_loop_logical:
            mpirun_cores = int(input("Give the number of cores you want to use for mpirun: "))
            if (mpirun_cores >= 1):
                print(" The given number of cores is ",mpirun_cores)
                cores_loop_logical = user_input_yes_no(  "Continue?")
            else:
                print("  Please use a positive value!")
        epochs_loop_logical = False
        while not epochs_loop_logical:
            n_epochs = int(input("Give the number of training epochs: "))
            if (n_epochs >= 1):
                print(" The given number of epochs is ",n_epochs)
                epochs_loop_logical = user_input_yes_no(  "Continue?")
            else:
                print("  Please use a positive value!")
        write_submission_script_logical = user_input_yes_no("Write a VSC submission script (if not, training is performed on your machine)?")
        if write_submission_script_logical:
            time_logical = user_input_yes_no("  Do you want to give a maximum time required for executing the job?")
            if time_logical:
                maximum_time = input(" Give the maximum time required for executing the job (hh:mm:ss):")
            else:
                maximum_time = None
            
    return create_logical,train_logical,analyse_logical,plot_logical,\
    set_size_ratios,n_sets_per_size,fix_random_seed_create,random_seed_create,\
    n_epochs,mpirun_cores,write_submission_script_logical,maximum_time,fix_random_seed_train,random_seed_train