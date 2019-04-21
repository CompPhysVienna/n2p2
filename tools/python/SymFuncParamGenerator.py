#!/usr/bin/env python

import numpy as np
import sys
import itertools
import traceback


class SymFuncParamGenerator:
    # TODO: might want to rename that variable
    symfunc_type_numbers = dict(radial=2,
                             angular_narrow=3,
                             angular_wide=9,
                             weighted_radial=12,
                             weighted_angular=13)

    def __init__(self, elements):
        # TODO: consider checking if valid input for elements
        self.elements = elements

        self.r_shift_grid = None
        self.eta_grid = None

        self.symfunc_type = None
        self.radial_grid_info = None

        self.lambdas = np.array([-1.0, 1.0])
        self.zetas = None

    def get_elements(self):
        return self.elements

    def clear_all(self):
        # TODO: either make this do something or remove...
        pass

    def check_symfunc_type(self):
        if not self.symfunc_type in self.symfunc_type_numbers.keys():
            raise ValueError('Invalid symmetry function type. Must be one of {}'.format(
                list(self.symfunc_type_numbers.keys())))

    def set_symfunc_type(self, symfunc_type):
        # TODO: set zetas to None if a radial type is given ??
        self.symfunc_type = symfunc_type
        self.check_symfunc_type()
    #
    def generate_radial_params(self, method, mode, r_cutoff, nb_gridpoints, r_lower=None):
        # store infos on radial parameter generation settings (that are independent of method)
        self.radial_grid_info = dict(method=method,
                                     mode=mode,
                                     r_cutoff=r_cutoff,
                                     nb_gridpoints=nb_gridpoints)

        if method == 'gastegger2018':
            if r_lower is None:
                raise TypeError('Argument r_lower is required for method "gastegger2018"')

            r_upper = r_cutoff - 0.5  # TODO: decide if hardcoding this (and in Angstrom!) is a good idea
            # store settings that are unique to this method
            self.radial_grid_info.update({'r_lower': r_lower, 'r_upper': r_upper})

            # create auxiliary grid
            grid = np.linspace(r_lower, r_upper, nb_gridpoints)

            if mode == 'center':
                r_shift_grid = np.zeros(nb_gridpoints)
                eta_grid = 1.0 / (2.0 * grid ** 2)
            elif mode == 'shift':
                r_shift_grid = grid
                dr = (r_upper - r_lower) / (nb_gridpoints - 1)
                eta_grid = np.full(nb_gridpoints, 1.0 / (2.0 * dr * dr))
            else:
                raise ValueError('invalid argument for "mode"')

        elif method == 'imbalzano2018':
            if r_lower is not None:
                # TODO: decide if necessary to do this warning and if doing it this way makes sense
                sys.stderr.write(
                    'Warning: argument r_lower is not used in method "imbalzano2018" and will be ignored.\n')
                traceback.print_stack()
                sys.stderr.flush()

            if mode == 'center':
                nb_intervals = nb_gridpoints - 1
                gridpoint_indices = np.array(range(0, nb_intervals + 1))
                eta_grid = (nb_intervals ** (gridpoint_indices / nb_intervals) / r_cutoff) ** 2
                r_shift_grid = np.zeros_like(eta_grid)
            elif mode == 'shift':
                # create extended auxiliary grid of r_shift values, that contains nb_gridpoints + 1 values
                nb_intervals_extended = nb_gridpoints
                gridpoint_indices_extended = np.array(range(0, nb_intervals_extended + 1))
                rs_grid_extended = r_cutoff / nb_intervals_extended ** (
                        gridpoint_indices_extended / nb_intervals_extended)
                # from pairs of neighboring r_shift values, compute eta values.
                # doing this for the nb_gridpoints + 1 values in the auxiliary grid ultimately gives...
                # ...nb_gridpoints different values for eta
                eta_grid = np.zeros(nb_gridpoints)
                for idx in range(len(rs_grid_extended) - 1):
                    eta_current = 1 / (rs_grid_extended[idx] - rs_grid_extended[idx + 1]) ** 2
                    eta_grid[idx] = eta_current
                # create final grid of r_shift values by excluding the first entry...
                # ...(for which r_shift coincides with the cutoff radius) from the extended grid
                r_shift_grid = rs_grid_extended[1:]
                # reverse the order of r_shift and eta values so they are sorted in order of ascending r_shift...
                # ...(not necessary, but makes the output consistent with the other methods)
                r_shift_grid = np.flip(r_shift_grid)
                eta_grid = np.flip(eta_grid)
            else:
                raise ValueError('invalid argument for "mode"')
        else:
            raise ValueError('invalid argument for "method"')

        self.r_shift_grid = r_shift_grid
        self.eta_grid = eta_grid

    def set_zetas(self, zeta_values):
        self.zetas = np.array(zeta_values)

    def check_all_settings(self):
        self.check_symfunc_type()
        if self.radial_grid_info is None:
            raise TypeError('No radial grid has been generated.')
        if self.symfunc_type is None:
            raise TypeError('Symmetry function type not set.')
        if self.symfunc_type in ['angular_narrow', 'angular_wide', 'weighted_angular']:
            if self.zetas is None:
                raise TypeError('zeta values not set.')

    def write_generation_info(self):
        self.check_all_settings()
        type_descriptions = dict(radial='Radial',
                                 angular_narrow='Narrow angular',
                                 angular_wide='Wide angular',
                                 weighted_radial='Weighted radial',
                                 weighted_angular='Weighted angular')

        sys.stdout.write(
            "# {} symmetry function set, generated with parameters:\n".format(type_descriptions[self.symfunc_type]))
        for key, value in self.radial_grid_info.items():
            sys.stdout.write('# {0:13s} = {1}\n'.format(key, value))
        # set numpy print precision to lower number of decimal places for the following outputs
        np.set_printoptions(precision=4)
        sys.stdout.write('# r_shift_grid  = {}\n'.format(self.r_shift_grid))
        sys.stdout.write('# eta_grid      = {}\n'.format(self.eta_grid))
        if self.symfunc_type in ['angular_narrow', 'angular_wide', 'weighted_angular']:
            sys.stdout.write('# lambdas       = {}\n'.format(self.lambdas))
            sys.stdout.write('# zetas         = {}\n'.format(self.zetas))
        # reset numpy print precision to default
        np.set_printoptions(precision=8)

    def create_element_combinations(self):
        self.check_symfunc_type()
        combinations = []

        if self.symfunc_type == 'radial':
            for elem_central in self.elements:
                for elem_neighbor in self.elements:
                    combinations.append((elem_central, elem_neighbor))
        elif self.symfunc_type in ['angular_narrow', 'angular_wide']:
            for elem_central in self.elements:
                for pair_of_neighbors in itertools.combinations_with_replacement(self.elements, 2):
                    comb = (elem_central,) + pair_of_neighbors
                    combinations.append(comb)
        elif self.symfunc_type in ['weighted_radial', 'weighted_angular']:
            for elem_central in self.elements:
                combinations.append((elem_central,))

        return combinations

    def write_parameter_strings(self):
        self.check_all_settings()

        element_combinations = self.create_element_combinations()
        r_cutoff = self.radial_grid_info['r_cutoff']
        sf_number = self.symfunc_type_numbers[self.symfunc_type]

        if self.symfunc_type == 'radial':
            for comb in element_combinations:
                for (eta, rs) in zip(self.eta_grid, self.r_shift_grid):
                    sys.stdout.write(
                        f'symfunction_short {comb[0]:2s} {sf_number} {comb[1]:2s} {eta:9.3E} {rs:9.3E} {r_cutoff:9.3E}\n')
                sys.stdout.write('\n')

        elif self.symfunc_type in ['angular_narrow', 'angular_wide']:
            for comb in element_combinations:
                for (eta, rs) in zip(self.eta_grid, self.r_shift_grid):
                    for zeta in self.zetas:
                        for lambd in self.lambdas:
                            sys.stdout.write(
                                f'symfunction_short {comb[0]:2s} {sf_number} {comb[1]:2s} {comb[2]:2s} {eta:9.3E} {lambd:2.0f} {zeta:9.3E} {r_cutoff:9.3E} {rs:9.3E}\n')
                sys.stdout.write('\n')

        elif self.symfunc_type == 'weighted_radial':
            for comb in element_combinations:
                for (eta, rs) in zip(self.eta_grid, self.r_shift_grid):
                    sys.stdout.write(
                        f'symfunction_short {comb[0]:2s} {sf_number} {eta:9.3E} {rs:9.3E} {r_cutoff:9.3E}\n')
                sys.stdout.write('\n')

        elif self.symfunc_type == 'weighted_angular':
            for comb in element_combinations:
                for (eta, rs) in zip(self.eta_grid, self.r_shift_grid):
                    for zeta in self.zetas:
                        for lambd in self.lambdas:
                            sys.stdout.write(
                                f'symfunction_short {comb[0]:2s} {sf_number} {eta:9.3E} {rs:9.3E} {lambd:2.0f} {zeta:9.3E} {r_cutoff:9.3E} \n')
                sys.stdout.write('\n')



if __name__ == '__main__':
    elems = ['S', 'Cu']
    # elems = ['H', 'C', 'O']
    myGen = SymFuncParamGenerator(elems)

    # print('gastegger2018 center mode')
    # myGen.generate_radial_params(method='gastegger2018', mode='center', r_cutoff=6., nb_gridpoints=3,
    #                              r_lower=1.5)
    # myGen.set_symfunc_type('angular_narrow')
    # myGen.set_zetas([1.0, 6.0])
    # myGen.write_generation_info()
    #
    # print('\ngastegger2018 shift mode')
    # myGen.generate_radial_params(method='gastegger2018', mode='shift', r_cutoff=6., nb_gridpoints=9, r_lower=1.5)
    # myGen.set_symfunc_type('angular_narrow')
    # myGen.set_zetas([1.0, 6.0])
    # myGen.write_generation_info()
    #
    # print('\nimbalzano2018 center mode')
    # myGen.generate_radial_params(method='imbalzano2018', mode='center', r_cutoff=5., nb_gridpoints=5)
    # myGen.set_symfunc_type('weighted_radial')
    # myGen.set_zetas([1.0, 6.0])
    # myGen.write_generation_info()

    print('\ngastegger2018 shift mode')
    myGen.generate_radial_params(method='gastegger2018', mode='shift', r_cutoff=6., nb_gridpoints=3, r_lower=1.5)
    myGen.set_symfunc_type('weighted_angular')
    myGen.set_zetas([1.0, 6.0])
    myGen.write_generation_info()

    # print('\nimbalzano2018 shift mode')
    # myGen.generate_radial_params(method='imbalzano2018', mode='shift', r_cutoff=6., nb_gridpoints=5)
    # myGen.set_symfunc_type('radial')
    # myGen.write_generation_info()


    print()
    myGen.write_parameter_strings()
