#!/usr/bin/env python

import numpy as np
import sys
import itertools


class SymFuncDescriptorRadial:
    def __init__(self, central_element, neighbor_element, eta, r_cutoff, r_shift):
        # radial symmetry functions are called type 2
        self.type = 12

        self.central_element = central_element
        self.neighbor_element = neighbor_element
        self.eta = eta
        self.r_cutoff = r_cutoff
        self.r_shift = r_shift

    def get_descriptor_string(self):
        descriptor_string = "symfunction_short {0:2s} {1:2d} {2:2s} {3:9.3E} {4:9.3E} {5:9.3E}".format(
            self.central_element, self.type, self.neighbor_element, self.eta, self.r_shift, self.r_cutoff)
        return descriptor_string


class SymFuncDescriptorAng:
    def __init__(self, central_element, type_keyword, neighbor_element_1, neighbor_element_2, eta, lambd, zeta, r_cutoff, r_shift):
        if type_keyword == 'narrow':
            self.type = 3
        elif type_keyword == 'wide':
            self.type = 9
        else:
            raise(ValueError('type_keyword must be one of ["narrow", "wide"]'))

        self.central_element = central_element
        # store neighbor elements in a set to make explicit that the order is irrelevant
        self.neighbor_elements = {neighbor_element_1, neighbor_element_2}
        self.eta = eta
        self.r_cutoff = r_cutoff
        self.r_shift = r_shift
        self.lambd = lambd
        self.zeta = zeta

    def get_descriptor_string(self):
        if len(self.neighbor_elements) == 2:
            neighbor_element_1 = sorted(self.neighbor_elements)[0]
            neighbor_element_2 = sorted(self.neighbor_elements)[1]
        elif len(self.neighbor_elements) == 1:
            neighbor_element_1 = sorted(self.neighbor_elements)[0]
            neighbor_element_2 = neighbor_element_1
        else:
            raise(RuntimeError('Neighbor elements must be either the same or two different ones, but not more than two'))

        descriptor_string = "symfunction_short {0:2s} {1:2d} {2:2s} {3:2s} {4:9.3E} {5:2.0f} {6:9.3E} {7:9.3E} {8:9.3E}".format(
            self.central_element, self.type, neighbor_element_1, neighbor_element_2,
            self.eta, self.lambd, self.zeta, self.r_cutoff, self.r_shift)
        return descriptor_string


sf_radial_1 = SymFuncDescriptorRadial(central_element='Cu', neighbor_element='S', eta=4., r_shift=5, r_cutoff=6)
sf_radial_2 = SymFuncDescriptorRadial(central_element='S', neighbor_element='S', eta=1., r_shift=2, r_cutoff=3)
sf_ang_narrow = SymFuncDescriptorAng(central_element='Cu', type_keyword='narrow', neighbor_element_1='S', neighbor_element_2='S', eta=4., lambd=1, zeta=.5, r_shift=5, r_cutoff=6)
sf_ang_wide = SymFuncDescriptorAng(central_element='Cu', type_keyword='wide', neighbor_element_1='S', neighbor_element_2='Cu', eta=4., lambd=1, zeta=.5, r_shift=5, r_cutoff=6)


print(sf_radial_1.get_descriptor_string())
print(sf_radial_2.get_descriptor_string())

print(sf_ang_narrow.get_descriptor_string())
print(sf_ang_wide.get_descriptor_string())


print()