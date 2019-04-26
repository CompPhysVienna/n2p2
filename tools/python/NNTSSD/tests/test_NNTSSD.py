# -*- coding: utf-8 -*-
"""
17.04.2019
@author: mr

PYTHON 3

Tests for NNTSSD: Neural Network Training Set Size Dependence
"""
import numpy as np
import sys
sys.path.append("../source")

import NNTSSD
import os

def test_create_training_datasets():
    """This function tests the method create_training_datasets().
    
    It fails if there is no file 'input.data' or 'nnp-select.log' in the created directories.
    """
    fails = []
    my_Class = NNTSSD.Tools()
    my_Class.create_training_datasets(np.array([0.8,0.9]),2,True)               
    
    if not os.path.isdir("Output"):
        fails.append("The file 'Output' has not been created.")
    if not np.in1d(np.array(['ratio0.80', 'ratio0.90']),np.sort(os.listdir("Output"))).all():
        fails.append("The file structure in 'Output' is not correct.")
    if not np.in1d(np.array(['ratio0.80_set1', 'ratio0.80_set2']),np.sort(os.listdir("Output/ratio0.80"))).all():
        fails.append("The file structure in 'Output/ratio0.80' is not correct.")
    if not np.in1d(np.array(['ratio0.90_set1', 'ratio0.90_set2']),np.sort(os.listdir("Output/ratio0.90"))).all():
        fails.append("The file structure in 'Output/ratio0.90' is not correct.")
    if not os.path.isfile("Output/ratio0.80/ratio0.80_set1/input.data"):
        fails.append("The file 'Output/ratio0.80/ratio0.80_set1/input.data' does not exist.")
    if not os.path.isfile("Output/ratio0.80/ratio0.80_set2/input.data"):
        fails.append("The file 'Output/ratio0.80/ratio0.80_set2/input.data' does not exist.")
    if not os.path.isfile("Output/ratio0.90/ratio0.90_set1/input.data"):
        fails.append("The file 'Output/ratio0.90/ratio0.90_set1/input.data' does not exist.")
    if not os.path.isfile("Output/ratio0.90/ratio0.90_set2/input.data"):
        fails.append("The file 'Output/ratio0.90/ratio0.90_set2/input.data' does not exist.")
    if not os.path.isfile("Output/ratio0.80/ratio0.80_set1/nnp-select.log"):
        fails.append("The file 'Output/ratio0.80/ratio0.80_set1/nnp-select.log' does not exist.")
    if not os.path.isfile("Output/ratio0.80/ratio0.80_set2/nnp-select.log"):
        fails.append("The file 'Output/ratio0.80/ratio0.80_set2/nnp-select.log' does not exist.")
    if not os.path.isfile("Output/ratio0.90/ratio0.90_set1/nnp-select.log"):
        fails.append("The file 'Output/ratio0.90/ratio0.90_set1/nnp-select.log' does not exist.")
    if not os.path.isfile("Output/ratio0.90/ratio0.90_set2/nnp-select.log"):
        fails.append("The file 'Output/ratio0.90/ratio0.90_set2/nnp-select.log' does not exist.")
    
    assert not fails, "Fails occured:\n{}".format("\n".join(fails))

def test_training_neural_network():
    """This function tests the method training_neural_network().
    
    It fails if there is no file 'learning-curve.out' created in the given directories.
    """
    fails = []
    my_Class = NNTSSD.Tools()
    my_Class.training_neural_network("mpirun -np 4 ../../../../../../../bin/nnp-train",False,False)
    
    if not os.path.isfile("Output/ratio0.80/ratio0.80_set1/learning-curve.out"):
        fails.append("The file 'Output/ratio0.80/ratio0.80_set1/learning-curve.out' does not exist.")
    if not os.path.isfile("Output/ratio0.80/ratio0.80_set2/learning-curve.out"):
        fails.append("The file 'Output/ratio0.80/ratio0.80_set2/learning-curve.out' does not exist.")
    if not os.path.isfile("Output/ratio0.90/ratio0.90_set1/learning-curve.out"):
        fails.append("The file 'Output/ratio0.90/ratio0.90_set1/learning-curve.out' does not exist.")
    if not os.path.isfile("Output/ratio0.90/ratio0.90_set2/learning-curve.out"):
        fails.append("The file 'Output/ratio0.90/ratio0.90_set2/learning-curve.out' does not exist.")
    
    assert not fails, "Fails occured:\n{}".format("\n".join(fails))

def test_analyse_learning_curves():
    """This function tests the method analyse_learning_curves().
    
    It fails if there is no file 'collect_data.out' or 'analyse_data.out' created in the given directories.
    """
    fails = []
    my_Class = NNTSSD.Tools()
    my_Class.analyse_learning_curves()
    
    if not os.path.isfile("Output/ratio0.80/collect_data.out"):
        fails.append("The file 'Output/ratio0.80/collect_data.out' does not exist.")
    if not os.path.isfile("Output/ratio0.90/collect_data.out"):
        fails.append("The file 'Output/ratio0.90/collect_data.out' does not exist.")
    if not os.path.isfile("Output/analyse_data.out"):
        fails.append("The file 'analyse_data.out' has not been created.")
        
    assert not fails, "Fails occured:\n{}".format("\n".join(fails))
    
    
def test_plot_size_dependence():
    """This function tests the method plot_size_dependence().
    
    It fails if there is no file 'Energy_RMSE.png' or 'Forces_RMSE.png' created in the given directories.
    """
    fails = []
    my_Class = NNTSSD.Tools()
    my_Class.plot_size_dependence()
    
    if not os.path.isfile("Output/Energy_RMSE.png"):
        fails.append("The file 'Output/Energy_RMSE.png' has not been created.")
    if not os.path.isfile("Output/Forces_RMSE.png"):
        fails.append("The file 'Output/Forces_RMSE.png' has not been created.")
        
    assert not fails, "Fails occured:\n{}".format("\n".join(fails))


def test_write_submission_script():
    """This function tests the function write_submission_script.
    
    It fails if the file 'submit.slrm' is not created with the correct input.
    """
    fails = []
    NNTSSD.write_submission_script(str("This is a test command"))
    
    if not os.path.isfile("submit.slrm"):
        fails.append("The file 'submit.slrm' has not been created.")
    submission_script = open("submit.slrm","r")
    found = False
    for line in submission_script:
        if str("This is a test command") in line:
            found = True
    if not found:
        fails.append("The command was written incorrectly to the submission script.")