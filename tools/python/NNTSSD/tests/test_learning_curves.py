#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06.05.2019
@author: mr

PYTHON 3

Tests for analysing learning curves and plotting training performance.
"""

import numpy as np
import sys
sys.path.append("../source")

import learning_curves
import os

def test_analyse_learning_curves():
    """This function tests the method analyse_learning_curves().
    
    It fails if there is no file 'collect_data.out' or 'analyse_data.out' created in the given directories.
    """
    fails = []
    my_Class = learning_curves.Tools_LC()
    my_Class.analyse_learning_curves()
    
    if not os.path.isfile("Output/training_performance/learning_curve_E.out"):
        fails.append("The file 'learning_curve_E.out' has not been created.")
    if not os.path.isfile("Output/training_performance/learning_curve_F.out"):
        fails.append("The file 'learning_curve_F.out' has not been created.")
    if not os.path.isfile("Output/training_performance/learning_curve_Etrain_0.80.out"):
        fails.append("The file 'learning_curve_Etrain_0.80.out' has not been created.")
    if not os.path.isfile("Output/training_performance/learning_curve_Etest_0.80.out"):
        fails.append("The file 'learning_curve_Etest_0.80.out' has not been created.")
    if not os.path.isfile("Output/training_performance/learning_curve_Ftrain_0.80.out"):
        fails.append("The file 'learning_curve_Ftrain_0.80.out' has not been created.")
    if not os.path.isfile("Output/training_performance/learning_curve_Ftest_0.80.out"):
        fails.append("The file 'learning_curve_Ftest_0.80.out' has not been created.")
    if not os.path.isfile("Output/training_performance/learning_curve_Etrain_0.90.out"):
        fails.append("The file 'learning_curve_Etrain_0.90.out' has not been created.")
    if not os.path.isfile("Output/training_performance/learning_curve_Etest_0.90.out"):
        fails.append("The file 'learning_curve_Etest_0.90.out' has not been created.")
    if not os.path.isfile("Output/training_performance/learning_curve_Ftrain_0.90.out"):
        fails.append("The file 'learning_curve_Ftrain_0.90.out' has not been created.")
    if not os.path.isfile("Output/training_performance/learning_curve_Ftest_0.90.out"):
        fails.append("The file 'learning_curve_Ftest_0.90.out' has not been created.")
    
    assert not fails, "Fails occured:\n{}".format("\n".join(fails))
    
def test_plot_training_performance():
    """This function tests the method plot_training_performance().
    
    It fails if the expected *.png files are not created in the given directories.
    """
    fails = []
    my_Class = learning_curves.Tools_LC()
    my_Class.plot_training_performance(2,20,np.array([0.8,0.9]))
    
    if not os.path.isfile("Output/training_performance/Learning_curve_Energies_mean.png"):
        fails.append("The file 'Output/training_performance/Learning_curve_Energies_mean.png' has not been created.")
    if not os.path.isfile("Output/training_performance/Learning_curve_Forces_mean.png"):
        fails.append("The file 'Output/training_performance/Learning_curve_Forces_mean.png' has not been created.")
    if not os.path.isfile("Output/training_performance/Learning_curve_Energies_all.png"):
        fails.append("The file 'Output/training_performance/Learning_curve_Energies_all.png' has not been created.")
    if not os.path.isfile("Output/training_performance/Learning_curve_Forces_all.png"):
        fails.append("The file 'Output/training_performance/Learning_curve_Forces_all.png' has not been created.")
        
    assert not fails, "Fails occured:\n{}".format("\n".join(fails))