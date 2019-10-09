#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 12:38:36 2019

@author: mr
"""

import split_data
import os
import sys
import shutil
import numpy as np

def prepare_original_data(pattern_foldername,sample_number):
    for sample_counter in range(1,sample_number+1):
        shutil.copytree(pattern_foldername,"Sample_"+str(sample_counter)+"_"+pattern_foldername)
        os.chdir("Sample_"+str(sample_counter)+"_"+pattern_foldername)
        try:
            shutil.rmtree("split_data")
            print("INFO: Removed old 'predict_testdata' folder.")
        except:
            pass
        os.system("mkdir split_data")
        split_data.split_data(0.1)
        shutil.copy("test.data","predict_test_data/input.data")
        shutil.copy("input.data","original.data")
        shutil.copy("train.data","input.data")
        os.chdir("../")

def perform():
    prepare_original_data("XXXX",5)
            
def main():
    pass
if __name__=="__main__":
    perform()