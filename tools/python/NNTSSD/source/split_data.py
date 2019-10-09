#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 13:10:59 2019

@author: mr
"""

import re
import numpy as np

def split_data(test_fraction):
    """Splits the file input.data in two files wrt given fraction.
    
    Idea taken from: `https://stackoverflow.com/questions/35916503/how-can-i-split-a-text-file-into-multiple-text-files-using-python`_
    
    Parameters
    ----------
    test_fraction : float
        Desired ratio of dataset that is kept for testing, eg. 0.1 means 10 percent.
    
    Notes
    -----
    Requirements
        ``input.data`` : file
            Contains original dataset.
    Outputs
        ``test.data`` : file
            Contains specified fraction of the original dataset, randomly choosen.
        ``train.data`` : file
            Contains remaining data.
    
    """
    with open('input.data', 'r') as f:
        data = f.read()
    
    found = re.findall(r'\n*( begin.*?\n end)\n*', data, re.M | re.S)
    
    for i in range(len(found)):
        if (np.random.rand() > test_fraction):
            open('train.data','a').write(found[i]+'\n')
        else:
            open('test.data','a').write(found[i]+'\n')

def perform():
    split_data(0.1)
            
def main():
    pass
if __name__=="__main__":
    perform()