#!/usr/bin/env python

import numpy as np
import sys
import itertools


class SymFuncParamGenerator:
    def __init__(self, elements):
        self.elements = elements

        self.sf_list_radial = []
        self.sf_list_ang_narrow = []
        self.sf_list_ang_wide = []

        self.generation_info_radial = ''
        self.generation_info_ang_narrow = ''
        self.generation_info_ang_wide = ''

    def get_elements(self):
        return self.elements

    def clear_radial(self):
        self.sf_list_radial = []
        self.generation_info_radial = ''

    def clear_ang_narrow(self):
        self.sf_list_ang_narrow = []
        self.generation_info_ang_narrow = ''

    def clear_ang_wide(self):
        self.sf_list_ang_wide = []
        self.generation_info_ang_wide = ''

    def generate_params_radial(self):




elems = ['Cu', 'S']

myGenerator = SymFuncParamGenerator(elems)

print(myGenerator.get_elements())






