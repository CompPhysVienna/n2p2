#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 13:48:11 2019

@author: mr

Tests for reading the NNTSSD parameters from file 'NNTSSD_input.dat'.
"""

import numpy as np
import sys
sys.path.append("../source")

import file_input

def test_read_parameters_from_file():
    """This function tests the function read_parameters_from_file().
    
    It fails if input file 'NNTSSD_input.dat' is read incorrectly.
    """
    fails = []
    create_logical,train_logical,analyse_logical,plot_logical,\
    set_size_ratios,n_sets_per_size,fix_random_seed_create,random_seed_create,\
    mpirun_cores,write_submission_script_logical,maximum_time,fix_random_seed_train,random_seed_train\
    = file_input.read_parameters_from_file()
    
    if not create_logical and not train_logical and analyse_logical and plot_logical:
        fails.append("The specification on which NNTSSD.Tools steps failed.")
    if not np.array_equal(set_size_ratios,np.array([0.8,0.9])):
        fails.append("The set_size_ratios were read incorrectly from the input file.")
    if not n_sets_per_size == 2:
        fails.append("The number of samples per set size was read incorrectly from the input file.")
    if not fix_random_seed_create == True:
        fails.append("The random seed logical for creating datasets is incorrect.")
    if not random_seed_create == 456:
        fails.append("The random seed for creating datasets is incorrect.")
    if not mpirun_cores == 4:
        fails.append("Fail: mpirun_cores read incorrectly.")
    if not write_submission_script_logical == False:
        fails.append("The logical for writing a VSC submission script is incorrect.")
    if not maximum_time=="08:00:00":
        fails.append("The maximum time required for exectuing the job is incorrect.")
    if not fix_random_seed_train == True:
        fails.append("The random seed logical for training datasets is incorrect.")
    if not random_seed_train == 789:
        fails.append("The random seed for training datasets is incorrect.")
    
    assert not fails, "Fails occured:\n{}".format("\n".join(fails))