#!/usr/bin/env python

import numpy as np
import sys
import itertools
import traceback


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

    def generate_radial_grid(self, method, mode, r_cutoff, nb_gridpoints, r_lower=None):
        if method == 'gastegger2018':
            if r_lower is None:
                raise TypeError('Argument r_lower is required for method "gastegger2018"')

            r_upper = r_cutoff - 0.5  # TODO: decide if hardcoding this (and in Angstrom!) is a good idea
            # create auxiliary grid
            grid = np.linspace(r_lower, r_upper, nb_gridpoints)

            if mode == 'center':
                rs_grid = np.zeros(nb_gridpoints)
                eta_grid = 1.0 / (2.0 * grid ** 2)
            elif mode == 'shift':
                rs_grid = grid
                dr = (r_upper - r_lower) / (nb_gridpoints - 1)
                eta_grid = np.full(nb_gridpoints, 1.0 / (2.0 * dr * dr))
            else:
                raise ValueError('invalid argument for "mode"')

        elif method == 'imbalzano2018':
            if r_lower is not None:
                # TODO: decide if it makes sense to do the warning this way
                sys.stderr.write(
                    'Warning: argument r_lower is not used in method "imbalzano2018" and will be ignored.\n')
                traceback.print_stack()
                sys.stderr.flush()

            if mode == 'center':
                nb_intervals = nb_gridpoints - 1
                gridpoint_indices = np.array(range(0, nb_intervals + 1))
                eta_grid = (nb_intervals ** (gridpoint_indices / nb_intervals) / r_cutoff) ** 2
                rs_grid = np.zeros_like(eta_grid)
            elif mode == 'shift':
                # create extended auxiliary grid of r_shift values, that contains nb_gridpoints + 1 values
                nb_intervals_extended = nb_gridpoints
                gridpoint_indices_extended = np.array(range(0, nb_intervals_extended + 1))
                rs_grid_extended = r_cutoff / nb_intervals_extended ** (
                        gridpoint_indices_extended / nb_intervals_extended)
                # from pairs of neighboring r_shift values, compute eta values
                # doing this for the nb_gridpoints + 1 values in the auxiliary grid ultimately gives...
                # ...nb_gridpoints different values for eta
                eta_grid = np.zeros(nb_gridpoints)
                for idx in range(len(rs_grid_extended) - 1):
                    eta_current = 1 / (rs_grid_extended[idx] - rs_grid_extended[idx + 1]) ** 2
                    eta_grid[idx] = eta_current
                # create final grid of rs_shift values by excluding the first entry...
                # ...(for which r_shift coincides with the cutoff radius) from the extended grid
                rs_grid = rs_grid_extended[1:]
                # reverse the order of r_shift and eta values so they are sorted in ascending order of r_shift...
                # ...(not necessary, but makes the output consistent with the other methods)
                rs_grid = np.flip(rs_grid)
                eta_grid = np.flip(eta_grid)
            else:
                raise ValueError('invalid argument for "mode"')
        else:
            raise ValueError('invalid argument for "method"')

        return rs_grid, eta_grid


elems = ['Cu', 'S']
myGenerator = SymFuncParamGenerator(elems)

print(myGenerator.get_elements())

print('\ngastegger2018 center mode')
a, b = myGenerator.generate_radial_grid(method='gastegger2018', mode='center', r_cutoff=6., nb_gridpoints=9,
                                        r_lower=1.5)
print(a)
print(b)

print('\ngastegger2018 shift mode')
a, b = myGenerator.generate_radial_grid(method='gastegger2018', mode='shift', r_cutoff=6., nb_gridpoints=9, r_lower=1.5)
print(a)
print(b)

print('\nimbalzano2018 center mode')
a, b = myGenerator.generate_radial_grid(method='imbalzano2018', mode='center', r_cutoff=6., nb_gridpoints=5)
print(a)
print(b)

print('\nimbalzano2018 shift mode')
a, b = myGenerator.generate_radial_grid(method='imbalzano2018', mode='shift', r_cutoff=6., nb_gridpoints=5)
print(a)
print(b)
