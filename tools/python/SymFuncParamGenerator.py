# -*- coding: utf-8 -*-

import numpy as np
import sys
import itertools
import traceback


class SymFuncParamGenerator:
    """Tools for generation, storage, and writing in the format required by n2p2, of symmetry function parameter sets.

    Parameters
    ----------
    elements : list of string
        The chemical elements present in the system.

    Attributes
    ----------
    symfunc_type_numbers : dict
        Dictionary mapping strings specifying the symmetry function type to the
        numbers used internally by n2p2 to distinguish symmetry function types.
    lambdas : numpy.ndarray
        Set of values for the parameter lambda of angular symmetry functions. Fixed to [-1, 1].
    radial_grid_info : dict or None
        Stores settings that were used for generating the symmetry function parameters r_shift and eta.
    r_shift_grid : numpy.ndarray or None
        The set of values for the symmetry function parameter r_shift that was generated.
    eta_grid : numpy.ndarray or None
        The set of values for the symmetry function parameter eta that was generated.
    elements
    element_combinations
    symfunc_type
    zetas
    """

    # TODO: might want to rename that variable
    symfunc_type_numbers = dict(radial=2,
                             angular_narrow=3,
                             angular_wide=9,
                             weighted_radial=12,
                             weighted_angular=13)

    def __init__(self, elements):
        # TODO: decide if there needs to be the option to change lambdas. If not, consider making it a class variable
        self.lambdas = np.array([-1.0, 1.0])

        # TODO: consider checking if valid input for elements
        self._elements = elements
        self._element_combinations = None
        self._symfunc_type = None
        self._zetas = None

        self.radial_grid_info = None
        self.r_shift_grid = None
        self.eta_grid = None

    @property
    def elements(self):
        """The chemical elements present in the system (list of string, read-only).
        """
        return self._elements

    @property
    def element_combinations(self):
        """Combinations of elements (list of tuple of string, read-only).

        This is computed and set automatically by the setter for symfunc_type.
        """
        return self._element_combinations

    @property
    def symfunc_type(self):
        """Type of symmetry function for which parameters are to be generated (`str`).

        When the setter for this is called it also checks the validity of
        the input, builds the necessary element combinations for the given
        symmetry function type, and stores it to member variable.
        """
        return self._symfunc_type

    @symfunc_type.setter
    def symfunc_type(self, value):
        # TODO: set zetas to None if a radial type is given ??
        self._symfunc_type = value
        self.check_symfunc_type()
        # once symmetry function type has been set and found to be valid,
        # build and store the element combinations
        self._element_combinations = self.construct_element_combinations()

    @property
    def zetas(self):
        """Set of values for the parameter zeta of angular symmetry functions (`numpy.ndarray`).
        """
        return self._zetas

    @zetas.setter
    def zetas(self, values):
        self._zetas = np.array(values)

    def check_symfunc_type(self):
        """Check if a symmetry function type has been set and if it is a valid one.

        Raises
        -------
        TypeError
            If the symmetry function type has not been set (i.e., it is None).
        ValueError
            If the symmetry function type is invalid.

        Returns
        -------
        None
        """
        if self._symfunc_type is None:
            raise TypeError('No symmetry function type has been set.')
        elif not self._symfunc_type in self.symfunc_type_numbers.keys():
            raise ValueError('Invalid symmetry function type. Must be one of {}'.format(
                list(self.symfunc_type_numbers.keys())))

    def generate_radial_params(self, method, mode, r_cutoff, nb_gridpoints, r_lower=None):
        """Generate a set of values for r_shift and eta.

        Such a set of values is required for any symmetry function type.
        The generated values are stored as arrays in the member variables r_shift_grid and eta_grid.
        The entries are to be understood pairwise, i.e., the i-th entry of r_shift_grid
        and the i-th entry of eta_grid belong to one symmetry function.
        Besides the set of values for r_shift and eta, the settings that
        were used for generating it are also stored, in the dictionary radial_grid_info.

        Parameters
        ----------
        method : {'gastegger2018', 'imbalzano2018'}
            if method=='gastegger2018' use the parameter generation rules presented in [1]_,
            if method=='imbalzano2018' use the parameter generation rules presented in [2]_
        mode : {'center', 'shift'}
            Selects which parameter generation rule to use, on top of the method argument,
            since there are again two different varieties presented in each of the two papers.
            'center' sets r_shift to zero for all symmetry functions, varying only eta.
            'shift' creates parameter sets where r_shift varies.
            The exact implementation details differ depending on
            the method parameter and are described in the papers.
        r_cutoff : float
            cutoff radius, at which the symmetry functions go to zero.
            must be greater than zero.
        nb_gridpoints : int
            number of (r_shift, eta)-pairs to be created.
        r_lower : float
            lowest value of radial grid points.
            required if method=='gastegger2018', ignored if method=='imbalzano2018'.

        Notes
        ----------
        [1] https://doi.org/10.1063/1.5019667
        [2] https://doi.org/10.1063/1.5024611

        The nomenclature from the papers was slightly adapted to make behavior
        consistent across all options for method and mode:
        nb_gridpoints universally specifies the number of (r_shift, eta)-pairs
        that the method ultimately generates, not the number of points in a
        preliminary auxiliary grid, or the number of intervals in the final grid.

        Returns
        -------
        None
        """
        # TODO: is this method the right place to pass the cutoff radius? Maybe even pass it to the constructor right away?
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
                # doing this for the nb_gridpoints + 1 values in the auxiliary
                # grid ultimately gives nb_gridpoints different values for eta.
                eta_grid = np.zeros(nb_gridpoints)
                for idx in range(len(rs_grid_extended) - 1):
                    eta_current = 1 / (rs_grid_extended[idx] - rs_grid_extended[idx + 1]) ** 2
                    eta_grid[idx] = eta_current
                # create final grid of r_shift values by excluding the first entry
                # (for which r_shift coincides with the cutoff radius) from the extended grid
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

    def check_all(self):
        """Check if a complete symmetry function set, with all required settings for writing, has been generated.

        Raises
        -------
        TypeError
            If no set of radial parameters (r_shift and eta) has been generated
        TypeError
            If an angular symmetry function type has been set, but no values for zeta were set.

        Returns
        -------
        None
        """
        self.check_symfunc_type()
        if self.radial_grid_info is None or self.r_shift_grid is None or self.eta_grid is None:
            raise TypeError('No radial grid has been generated.')
        if self._symfunc_type in ['angular_narrow', 'angular_wide', 'weighted_angular']:
            if self._zetas is None:
                raise TypeError('zeta values not set.')

    def write_generation_info(self):
        """Writes the settings used in generating the currently stored set of symmetry function parameters to stdout.

        Returns
        -------
        None
        """
        self.check_all()
        type_descriptions = dict(radial='Radial',
                                 angular_narrow='Narrow angular',
                                 angular_wide='Wide angular',
                                 weighted_radial='Weighted radial',
                                 weighted_angular='Weighted angular')

        sys.stdout.write(
            '# {} symmetry function set, generated with parameters:\n'.format(type_descriptions[self._symfunc_type]))
        sys.stdout.write(f'# elements      = {self.elements}\n')
        for key, value in self.radial_grid_info.items():
            sys.stdout.write(f'# {key:13s} = {value}\n')
        # set numpy print precision to lower number of decimal places for the following outputs
        np.set_printoptions(precision=4)
        sys.stdout.write(f'# r_shift_grid  = {self.r_shift_grid}\n')
        sys.stdout.write(f'# eta_grid      = {self.eta_grid}\n')
        if self._symfunc_type in ['angular_narrow', 'angular_wide', 'weighted_angular']:
            sys.stdout.write(f'# lambdas       = {self.lambdas}\n')
            sys.stdout.write(f'# zetas         = {self._zetas}\n')
        # reset numpy print precision to default
        np.set_printoptions(precision=8)

    def construct_element_combinations(self):
        """Create combinations of elements, depending on symmetry function type and the elements in the system.

        For radial symmetry functions, the combinations are all possible ordered pairs of elements in the system,
        including of an element with itself.
        For angular symmetry functions (narrow or wide), the combinations consist of all possible elements as the
        central atom, and then again for each central element all possible unordered pairs of neighbor elements.
        For weighted symmetry functions (radial or angular), the combinations run only over all possible
        central elements, with neighbors not taken into account at this stage.

        Returns
        -------
        combinations : list of tuple of string
            Each tuple in the list represents one element combination.
            Length of the individual tuples can be 1, 2 or 3, depending on symmetry function type.
            Zero-th entry of tuples is always the type of the central atom, 1st and 2nd entry are neighbor atom types
            (radial sf: one neighbor, angular sf: two neighbors, weighted sf: no neighbors)
        """
        self.check_symfunc_type()
        combinations = []

        if self._symfunc_type == 'radial':
            for elem_central in self.elements:
                for elem_neighbor in self.elements:
                    combinations.append((elem_central, elem_neighbor))
        elif self._symfunc_type in ['angular_narrow', 'angular_wide']:
            for elem_central in self.elements:
                for pair_of_neighbors in itertools.combinations_with_replacement(self.elements, 2):
                    comb = (elem_central,) + pair_of_neighbors
                    combinations.append(comb)
        elif self._symfunc_type in ['weighted_radial', 'weighted_angular']:
            for elem_central in self.elements:
                combinations.append((elem_central,))

        return combinations

    def write_parameter_strings(self):
        """Write symmetry function parameter sets to stdout, formatted as in the file 'input.nn' required by n2p2.

        Each line in the output corresponds to one symmetry function.
        Output is formatted in blocks separated by blank lines, each block corresponding to one element combination.
        Within each block, the other parameters are iterated over.

        Returns
        -------
        None
        """
        self.check_all()

        r_cutoff = self.radial_grid_info['r_cutoff']
        sf_number = self.symfunc_type_numbers[self.symfunc_type]

        # TODO: make some kind of linebreaks within the strings to keep code lines below hard limit of 120 chars
        if self._symfunc_type == 'radial':
            for comb in self.element_combinations:
                for (eta, rs) in zip(self.eta_grid, self.r_shift_grid):
                    sys.stdout.write(
                        f'symfunction_short {comb[0]:2s} {sf_number} {comb[1]:2s} {eta:9.3E} {rs:9.3E} {r_cutoff:9.3E}\n')
                sys.stdout.write('\n')

        elif self._symfunc_type in ['angular_narrow', 'angular_wide']:
            for comb in self.element_combinations:
                for (eta, rs) in zip(self.eta_grid, self.r_shift_grid):
                    for zeta in self._zetas:
                        for lambd in self.lambdas:
                            sys.stdout.write(
                                f'symfunction_short {comb[0]:2s} {sf_number} {comb[1]:2s} {comb[2]:2s} {eta:9.3E} {lambd:2.0f} {zeta:9.3E} {r_cutoff:9.3E} {rs:9.3E}\n')
                sys.stdout.write('\n')

        elif self._symfunc_type == 'weighted_radial':
            for comb in self.element_combinations:
                for (eta, rs) in zip(self.eta_grid, self.r_shift_grid):
                    sys.stdout.write(
                        f'symfunction_short {comb[0]:2s} {sf_number} {eta:9.3E} {rs:9.3E} {r_cutoff:9.3E}\n')
                sys.stdout.write('\n')

        elif self._symfunc_type == 'weighted_angular':
            for comb in self.element_combinations:
                for (eta, rs) in zip(self.eta_grid, self.r_shift_grid):
                    for zeta in self._zetas:
                        for lambd in self.lambdas:
                            sys.stdout.write(
                                f'symfunction_short {comb[0]:2s} {sf_number} {eta:9.3E} {rs:9.3E} {lambd:2.0f} {zeta:9.3E} {r_cutoff:9.3E} \n')
                sys.stdout.write('\n')



def main():
    elems = ['S', 'Cu']
    # elems = ['H', 'C', 'O']
    myGen = SymFuncParamGenerator(elems)

    print('\nimbalzano2018 shift mode')
    myGen.generate_radial_params(method='imbalzano2018', mode='shift', r_cutoff=6., nb_gridpoints=5)
    myGen.symfunc_type = 'radial'
    myGen.write_generation_info()

    # print('\ngastegger2018 shift mode')
    # myGen.generate_radial_params(method='gastegger2018', mode='shift', r_cutoff=6., nb_gridpoints=9, r_lower=1.5)
    # myGen.symfunc_type = 'angular_narrow'
    # myGen.zetas = [1.0, 6.0]
    # myGen.write_generation_info()

    # print('gastegger2018 center mode')
    # myGen.generate_radial_params(method='gastegger2018', mode='center', r_cutoff=6., nb_gridpoints=3, r_lower=1.5)
    # myGen.symfunc_type = 'angular_wide'
    # myGen.zetas = [1.0, 6.0]
    # myGen.write_generation_info()

    # print('\nimbalzano2018 center mode')
    # myGen.generate_radial_params(method='imbalzano2018', mode='center', r_cutoff=5., nb_gridpoints=5)
    # myGen.symfunc_type = 'weighted_radial'
    # myGen.write_generation_info()

    # print('\ngastegger2018 shift mode')
    # myGen.generate_radial_params(method='gastegger2018', mode='shift', r_cutoff=6., nb_gridpoints=3, r_lower=1.5)
    # myGen.symfunc_type = 'weighted_angular'
    # myGen.zetas = [1.0, 6.0]
    # myGen.write_generation_info()


    print()
    myGen.write_parameter_strings()

    print(myGen.element_combinations)


if __name__ == '__main__':
    main()
