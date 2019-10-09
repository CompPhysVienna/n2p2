# -*- coding: utf-8 -*-
"""
06.05.2019
@author: mr

PYTHON 3

Tests for NNTSSD: Neural Network Training Set Size Dependence
"""
import numpy as np
import sys
sys.path.append("../source")

import NNTSSD
import os
#%% Tests for NNTSSD.Tools class
def test_create_training_datasets():
    """This function tests the method NNTSSD.Tools.create_training_datasets().
    
    It fails if the expected directory structure is not created or there is no file 'input.data' or 'nnp-select.log' in the created directories.
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
    """This function tests the method NNTSSD.Tools.training_neural_network().
    
    It fails if there is no file 'learning-curve.out' created in the given directories.
    """
    fails = []
    my_Class = NNTSSD.Tools()
    my_Class.training_neural_network("mpirun -np 4 ../../../../../../../bin/nnp-train",2,0.1,False,True)
    
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
    """This function tests the method NNTSSD.Tools.analyse_learning_curves().
    
    It fails if the files 'collect_data_***.out' or 'analyse_data_***.out' are not created in the given directories.
    """
    fails = []
    my_Class = NNTSSD.Tools()
    my_Class.analyse_learning_curves()
    
    if not os.path.isfile("Output/ratio0.80/collect_data_min_energy.out"):
        fails.append("The file 'Output/ratio0.80/collect_data_min_energy.out' does not exist.")
    if not os.path.isfile("Output/ratio0.80/collect_data_min_force.out"):
        fails.append("The file 'Output/ratio0.80/collect_data_min_force.out' does not exist.")
    if not os.path.isfile("Output/ratio0.80/collect_data_min_comb.out"):
        fails.append("The file 'Output/ratio0.80/collect_data_min_comb.out' does not exist.")
    if not os.path.isfile("Output/ratio0.90/collect_data_min_energy.out"):
        fails.append("The file 'Output/ratio0.90/collect_data_min_energy.out' does not exist.")
    if not os.path.isfile("Output/ratio0.90/collect_data_min_force.out"):
        fails.append("The file 'Output/ratio0.90/collect_data_min_force.out' does not exist.")
    if not os.path.isfile("Output/ratio0.90/collect_data_min_comb.out"):
        fails.append("The file 'Output/ratio0.90/collect_data_min_comb.out' does not exist.")
    if not os.path.isfile("Output/analyse_data_min_energy.out"):
        fails.append("The file 'analyse_data_min_energy.out' has not been created.")
    if not os.path.isfile("Output/analyse_data_min_force.out"):
        fails.append("The file 'analyse_data_min_force.out' has not been created.")
    if not os.path.isfile("Output/analyse_data_min_comb.out"):
        fails.append("The file 'analyse_data_min_comb.out' has not been created.")
    if not os.path.isfile("Output/analyse_data_min_energy_all.out"):
        fails.append("The file 'analyse_data_min_energy_all.out' has not been created.")
    if not os.path.isfile("Output/analyse_data_min_force_all.out"):
        fails.append("The file 'analyse_data_min_force_all.out' has not been created.")
    if not os.path.isfile("Output/analyse_data_min_comb_all.out"):
        fails.append("The file 'analyse_data_min_comb_all.out' has not been created.")
    
    assert not fails, "Fails occured:\n{}".format("\n".join(fails))
    
    
def test_plot_size_dependence():
    """This function tests the method NNTSSD.Tools.plot_size_dependence().
    
    It fails if there is no file 'Energy_RMSE_epoch_comparison.png' or 'Forces_RMSE_epoch_comparison.png' created in the given directories.
    """
    fails = []
    my_Class = NNTSSD.Tools()
    my_Class.plot_size_dependence()
    
    if not os.path.isfile("Output/int_Energy_RMSE_epoch_comparison.png"):
        fails.append("The file 'Output/Energy_RMSE_epoch_comparison.png' has not been created.")
    if not os.path.isfile("Output/int_Forces_RMSE_epoch_comparison.png"):
        fails.append("The file 'Output/Forces_RMSE_epoch_comparison.png' has not been created.")
        
    assert not fails, "Fails occured:\n{}".format("\n".join(fails))


#%% Tests for NNTSSD.Validation class
def test_predict_validation_data():
    """This function tests the method NNTSSD.Validation.predict_validation_data().
    
    It fails if the directory 'validation' or the files 'validation_data_***.out' are not created in the given directories.
    """
    fails = []
    myClass = NNTSSD.Validation()
    myClass.predict_validation_data(4,True)
    
    if not os.path.isdir("Output/ratio0.80/ratio0.80_set1/validation"):
        fails.append("The folder 'validation' does not exist in 'Output/ratio0.80/ratio0.80_set1'")
    if not os.path.isdir("Output/ratio0.80/ratio0.80_set2/validation"):
        fails.append("The folder 'validation' does not exist in 'Output/ratio0.80/ratio0.80_set2'")
    if not os.path.isdir("Output/ratio0.90/ratio0.90_set1/validation"):
        fails.append("The folder 'validation' does not exist in 'Output/ratio0.90/ratio0.90_set1'")
    if not os.path.isdir("Output/ratio0.90/ratio0.90_set2/validation"):
        fails.append("The folder 'validation' does not exist in 'Output/ratio0.90/ratio0.90_set2'")
    if not os.path.isfile("Output/validation_data_min_comb.out"):
        fails.append("The file 'Output/validation_data_min_comb.out' has not been created.")
    if not os.path.isfile("Output/validation_data_min_energy.out"):
        fails.append("The file 'Output/validation_data_min_energy.out' has not been created.")
    if not os.path.isfile("Output/validation_data_min_force.out"):
        fails.append("The file 'Output/validation_data_min_force.out' has not been created.")
    if not os.path.isfile("Output/validation_data_min_comb_all.out"):
        fails.append("The file 'Output/validation_data_min_comb_all.out' has not been created.")
    if not os.path.isfile("Output/validation_data_min_energy_all.out"):
        fails.append("The file 'Output/validation_data_min_energy_all.out' has not been created.")
    if not os.path.isfile("Output/validation_data_min_force_all.out"):
        fails.append("The file 'Output/validation_data_min_force_all.out' has not been created.")
    
    assert not fails, "Fails occured:\n{}".format("\n".join(fails))


def test_plot_validation_data():
    """This function tests the method NNTSSD.Validation.plot_validation_data().
    
    It fails if the files 'val_Energy_***.png' or 'val_Forces_***.png' are not created in the 'Output'-folder.
    """
    fails = []
    myClass = NNTSSD.Validation()
    myClass.plot_validation_data()
    
    if not os.path.isfile("Output/val_Energy_RMSE.png"):
        fails.append("The file 'Output/val_Energy_RMSE.png' has not been created.")
    if not os.path.isfile("Output/val_Forces_RMSE.png"):
        fails.append("The file 'Output/val_Forces_RMSE.png' has not been created.")
    if not os.path.isfile("Output/val_Energy_RMSE_best_E.png"):
        fails.append("The file 'Output/val_Energy_RMSE_best_E.png' has not been created.")
    if not os.path.isfile("Output/val_Forces_RMSE_best_F.png"):
        fails.append("The file 'Output/val_Forces_RMSE_best_F.png' has not been created.")
    
    assert not fails, "Fails occured:\n{}".format("\n".join(fails))


#%% Tests for NNTSSD.External_Testdata class

def test_predict_test_data():
    """This function tests the method NNTSSD.External_Testdata.predict_test_data().
    
    It fails if there is no directory 'predict_testdata' created in the given directory or if no files 'learning-curve-testdata.out' are created inside this directory.
    """
    fails = []
    myClass = NNTSSD.External_Testdata()
    myClass.predict_test_data(4,True)
    
    if not os.path.isdir("Output/ratio0.80/ratio0.80_set1/predict_testdata"):
        fails.append("The folder 'predict_testdata' does not exist in 'Output/ratio0.80/ratio0.80_set1'")
    if not os.path.isdir("Output/ratio0.80/ratio0.80_set2/predict_testdata"):
        fails.append("The folder 'predict_testdata' does not exist in 'Output/ratio0.80/ratio0.80_set2'")
    if not os.path.isdir("Output/ratio0.90/ratio0.90_set1/predict_testdata"):
        fails.append("The folder 'predict_testdata' does not exist in 'Output/ratio0.90/ratio0.90_set1'")
    if not os.path.isdir("Output/ratio0.90/ratio0.90_set2/predict_testdata"):
        fails.append("The folder 'predict_testdata' does not exist in 'Output/ratio0.90/ratio0.90_set2'")
    if not os.path.isfile("Output/ratio0.80/ratio0.80_set1/predict_testdata/learning-curve-testdata.out"):
        fails.append("The file 'learning-curve-testdata.out' does not exist in 'Output/ratio0.80/ratio0.80_set1/predict_testdata'")
    if not os.path.isfile("Output/ratio0.80/ratio0.80_set2/predict_testdata/learning-curve-testdata.out"):
        fails.append("The file 'learning-curve-testdata.out' does not exist in 'Output/ratio0.80/ratio0.80_set2/predict_testdata'")
    if not os.path.isfile("Output/ratio0.90/ratio0.90_set1/predict_testdata/learning-curve-testdata.out"):
        fails.append("The file 'learning-curve-testdata.out' does not exist in 'Output/ratio0.90/ratio0.90_set1/predict_testdata'")
    if not os.path.isfile("Output/ratio0.90/ratio0.90_set2/predict_testdata/learning-curve-testdata.out"):
        fails.append("The file 'learning-curve-testdata.out' does not exist in 'Output/ratio0.90/ratio0.90_set2/predict_testdata'")
    
    assert not fails, "Fails occured:\n{}".format("\n".join(fails))


def test_analyse_lc_testdata():
    """This function tests the method NNTSSD.External_Testdata.analyse_learning_curves().
    
    It fails if the files 'ext_collect_data_***.out' or 'ext_analyse_data_***.out' are not created in the given directories.
    """
    fails = []
    myClass = NNTSSD.External_Testdata()
    myClass.analyse_learning_curves()
    
    if not os.path.isfile("Output/ratio0.80/ext_collect_data_min_energy.out"):
        fails.append("The file 'Output/ratio0.80/ext_collect_data_min_energy.out' does not exist.")
    if not os.path.isfile("Output/ratio0.80/ext_collect_data_min_force.out"):
        fails.append("The file 'Output/ratio0.80/ext_collect_data_min_force.out' does not exist.")
    if not os.path.isfile("Output/ratio0.80/ext_collect_data_min_comb.out"):
        fails.append("The file 'Output/ratio0.80/ext_collect_data_min_comb.out' does not exist.")
    if not os.path.isfile("Output/ratio0.90/ext_collect_data_min_energy.out"):
        fails.append("The file 'Output/ratio0.90/ext_collect_data_min_energy.out' does not exist.")
    if not os.path.isfile("Output/ratio0.90/ext_collect_data_min_force.out"):
        fails.append("The file 'Output/ratio0.90/ext_collect_data_min_force.out' does not exist.")
    if not os.path.isfile("Output/ratio0.90/ext_collect_data_min_comb.out"):
        fails.append("The file 'Output/ratio0.90/ext_collect_data_min_comb.out' does not exist.")
    if not os.path.isfile("Output/ext_analyse_data_min_energy.out"):
        fails.append("The file 'ext_analyse_data_min_energy.out' has not been created.")
    if not os.path.isfile("Output/ext_analyse_data_min_force.out"):
        fails.append("The file 'ext_analyse_data_min_force.out' has not been created.")
    if not os.path.isfile("Output/ext_analyse_data_min_comb.out"):
        fails.append("The file 'ext_analyse_data_min_comb.out' has not been created.")
    if not os.path.isfile("Output/ext_analyse_data_min_energy_all.out"):
        fails.append("The file 'ext_analyse_data_min_energy_all.out' has not been created.")
    if not os.path.isfile("Output/ext_analyse_data_min_force_all.out"):
        fails.append("The file 'ext_analyse_data_min_force_all.out' has not been created.")
    if not os.path.isfile("Output/ext_analyse_data_min_comb_all.out"):
        fails.append("The file 'ext_analyse_data_min_comb_all.out' has not been created.")
    
    assert not fails, "Fails occured:\n{}".format("\n".join(fails))


def test_plot_test_size_dependence():
    """This function tests the method NNTSSD.External_Testdata.analyse_learning_curves().
    
    It fails if the files 'ext_Energy_RMSE_***.png' or 'ext_Forces_RMSE_***.png' are not created in the 'Output' folder.
    """
    fails = []
    myClass = NNTSSD.External_Testdata()
    myClass.plot_test_size_dependence()
    
    if not os.path.isfile("Output/ext_Energy_RMSE.png"):
        fails.append("The file 'Output/ext_Energy_RMSE.png' has not been created.")
    if not os.path.isfile("Output/ext_Forces_RMSE.png"):
        fails.append("The file 'Output/ext_Forces_RMSE.png' has not been created.")
    if not os.path.isfile("Output/ext_Energy_RMSE_epoch_compare.png"):
        fails.append("The file 'Output/ext_Energy_RMSE_epoch_compare.png' has not been created.")
    if not os.path.isfile("Output/ext_Forces_RMSE_epoch_compare.png"):
        fails.append("The file 'Output/ext_Forces_RMSE_epoch_compare.png' has not been created.")
    
    assert not fails, "Fails occured:\n{}".format("\n".join(fails))

    
#%% Tests for other functions in NNTSSD.py
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