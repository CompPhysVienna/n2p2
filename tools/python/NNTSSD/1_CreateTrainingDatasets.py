# -*- coding: utf-8 -*-
"""
09.04.2019
@author: mr
"""

import os
import shutil
import numpy as np

def create_training_datasets(set_size_ratios,n_sets_per_size):
    """
    This function creates training datasets for
    a number of samples (user-given in n_sets per_size) for each element of
    a set of training sizes (user-given in set_size_ratios)
    from file input.data using the program nnp-select.
    It creates one folder for each training set size.
    Inside these folders, it creates one folder per sample, in which the output files
    (output.data and nnp-select.log, created by nnp-select) can be found.
    """
    n_set_size_ratios = np.size(set_size_ratios)
    print "---------------------------------------------"
    print "number of samples per training set size = ", n_sets_per_size
    print "number of different training set sizes = ", n_set_size_ratios
    print "---------------------------------------------"
    for ratios_counter in range(n_set_size_ratios):
        current_ratio = set_size_ratios[ratios_counter]
        print "We are working with ratio {:3.2f}".format(current_ratio)
        ratio_folder = "ratio"+str("{:3.2f}".format(current_ratio))
        os.system("mkdir "+ratio_folder)
        for sets_per_size_counter in range(n_sets_per_size):
            nnp_select = "../../../bin/nnp-select random "+str("{:3.2f}".format(current_ratio))+" "+str(int(np.random.randint(100,999,1)))
            print nnp_select
            os.system(nnp_select)
            os.chdir(ratio_folder)
            set_folder = "ratio"+str("{:3.2f}".format(current_ratio))+"_set"+str(sets_per_size_counter)
            os.system("mkdir "+set_folder)
            os.chdir("../")
            shutil.move("output.data",ratio_folder+"/"+set_folder+"/output.data")
            shutil.move("nnp-select.log",ratio_folder+"/"+set_folder+"/nnp-select.log")

create_training_datasets(np.arange(0.1,0.25,0.05),3)