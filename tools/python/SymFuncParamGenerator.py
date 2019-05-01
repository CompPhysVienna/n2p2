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

    symfunc_type_numbers = dict(radial=2,
                             angular_narrow=3,
                             angular_wide=9,
                             weighted_radial=12,
                             weighted_angular=13)
    lambdas = np.array([-1.0, 1.0])

    def __init__(self, elements):
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

        If the given symmetry function type is a radial one,
        the setter also clears any preexisting zetas
        (i.e., sets the member variable zetas to None).
        """
        return self._symfunc_type

    @symfunc_type.setter
    def symfunc_type(self, value):
        self._symfunc_type = value
        self.check_symfunc_type()
        # once symmetry function type has been set and found to be valid,
        # build and store the element combinations
        self._element_combinations = self.find_element_combinations()
        # Clear any previous zeta values, if the given symfunc type is a radial one
        if self.symfunc_type in ['radial', 'weighted_radial']:
            # set the member variable explicitly (with underscore) instead of
            # calling setter, because the setter would put None into an array
            self._zetas = None

    @property
    def zetas(self):
        """Set of values for the parameter zeta of angular symmetry functions (`numpy.ndarray`).
        """
        return self._zetas

    @zetas.setter
    def zetas(self, values):
        self._zetas = np.array(values)

    def check_symfunc_type(self):
        """Check if a symmetry function type has been set and if it is valid.

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
        if self.symfunc_type is None:
            raise TypeError('No symmetry function type has been set.')
        elif not self.symfunc_type in self.symfunc_type_numbers.keys():
            raise ValueError('Invalid symmetry function type. Must be one of {}'.format(
                list(self.symfunc_type_numbers.keys())))

    def generate_radial_params(self, rule, mode, r_cutoff, nb_param_pairs,
                               r_lower=None, r_upper=None):
        """Generate a set of values for r_shift and eta.

        Such a set of values is required for any symmetry function type.
        Its generation is independent of the symmetry function type
        and the angular symmetry function parameters zeta and lambda.

        Rules for parameter generation are implemented based on [1]_ and [2]_.

        The generated values are stored as arrays in the member variables r_shift_grid and eta_grid.
        The entries are to be understood pairwise, i.e., the i-th entry of r_shift_grid
        and the i-th entry of eta_grid belong to one symmetry function.
        Besides the set of values for r_shift and eta, the settings that
        were used for generating it are also stored, in the dictionary radial_grid_info.
        # TODO: rethink use of radial_grid_info

        Parameters
        ----------
        rule : {'gastegger2018', 'imbalzano2018'}
            If rule=='gastegger2018' use the parameter generation rules presented in [1]_.
            If rule=='imbalzano2018' use the parameter generation rules presented in [2]_.
        mode : {'center', 'shift'}
            Selects which parameter generation procedure to use, on top of the rule argument,
            since there are again two different varieties presented in each of the two papers.
            'center' sets r_shift to zero for all symmetry functions, varying only eta.
            'shift' creates parameter sets where r_shift varies.
            The exact implementation details differ depending on
            the rule parameter and are described in the papers.
        r_cutoff : float
            Cutoff radius, at which symmetry functions go to zero.
            Must be greater than zero.
        nb_param_pairs : int
            Number of (r_shift, eta)-pairs to be generated.
        r_lower : float
            lowest value in the radial grid from which r_shift and eta values
            are computed.
            required if rule=='gastegger2018'.
            ignored if rule=='imbalzano2018'.
        r_upper : float, optional
            largest value in the radial grid from which r_shift and eta
            values are computed.
            optional if rule=='gastegger2018, defaults to r_cutoff if not given.
            ignored if rule=='imbalzano2018'.

        Notes
        ----------
        [1] https://doi.org/10.1063/1.5019667
        [2] https://doi.org/10.1063/1.5024611

        The parameter nb_param_pairs invariably specifies the number of
        (r_shift, eta)-pairs ultimately generated, not the number of
        intervals between points in a grid of r_shift values, or the
        number of points in some auxiliary grid.
        This constitutes a slight modification of the nomenclature in [2],
        for the sake of consistent behavior across all options for rule and
        mode.

        While the parameter generation by this method does not itself
        depend on symmetry function type, be aware of the exact mathematical
        expressions for the different symmetry function types and how the
        parameters r_shift and eta appear slightly differently in each of them.

        All this method does is implement the procedures described in the two
        papers for generating values of the symmetry function parameters
        r_shift and eta. As far as this module is concerned, these can then
        be used (with the above caveat) in combination with any symmetry
        function type. However, this does not imply that their use with all
        the different types of symmetry functions is actually discussed in
        the papers or was necessarily intended by the authors.

        Returns
        -------
        None
        """
        # store those infos on radial parameter generation settings that are
        # independent of the rule argument
        self.radial_grid_info = dict(rule=rule,
                                     mode=mode,
                                     r_cutoff=r_cutoff,
                                     nb_param_pairs=nb_param_pairs)

        if rule == 'gastegger2018':
            if r_lower is None:
                raise TypeError('Argument r_lower is required for rule "gastegger2018"')
            if r_upper is None:
                # by default, set largest value of radial grid to cutoff radius
                r_upper = r_cutoff

            # store settings that are unique to this rule
            self.radial_grid_info.update({'r_lower': r_lower, 'r_upper': r_upper})

            # create auxiliary grid
            grid = np.linspace(r_lower, r_upper, nb_param_pairs)

            if mode == 'center':
                # r_lower = 0 is not allowed, because it leads to division by zero
                if not 0 < r_lower < r_upper <= r_cutoff:
                    raise ValueError(f'Invalid argument(s): rule = {rule:s}, mode = {mode:s} requires that 0 < r_lower < r_upper <= r_cutoff.')
                r_shift_grid = np.zeros(nb_param_pairs)
                eta_grid = 1.0 / (2.0 * grid ** 2)
            elif mode == 'shift':
                # conversely, in shift mode, r_lower = 0 is possible
                if not 0 <= r_lower < r_upper <= r_cutoff:
                    raise ValueError(f'Invalid argument(s): rule = {rule:s}, mode = {mode:s} requires that 0 <= r_lower < r_upper <= r_cutoff.')
                r_shift_grid = grid
                # compute the equidistant grid spacing
                dr = (r_upper - r_lower) / (nb_param_pairs - 1)
                eta_grid = np.full(nb_param_pairs, 1.0 / (2.0 * dr * dr))
            else:
                raise ValueError('invalid argument for "mode"')

        elif rule == 'imbalzano2018':
            if r_lower is not None:
                sys.stderr.write(
                    'Warning: argument r_lower is not used in rule "imbalzano2018" and will be ignored.\n')
                traceback.print_stack()
                sys.stderr.flush()
            if r_upper is not None:
                sys.stderr.write(
                    'Warning: argument r_upper is not used in rule "imbalzano2018" and will be ignored.\n')
                traceback.print_stack()
                sys.stderr.flush()

            if not r_cutoff > 0:
                raise ValueError('Invalid argument for r_cutoff. Must be greater than zero.')

            if mode == 'center':
                nb_intervals = nb_param_pairs - 1
                gridpoint_indices = np.array(range(0, nb_intervals + 1))
                eta_grid = (nb_intervals ** (gridpoint_indices / nb_intervals) / r_cutoff) ** 2
                r_shift_grid = np.zeros_like(eta_grid)
            elif mode == 'shift':
                # create extended auxiliary grid of r_shift values,
                # that contains nb_param_pairs + 1 values
                nb_intervals_extended = nb_param_pairs
                gridpoint_indices_extended = np.array(range(0, nb_intervals_extended + 1))
                rs_grid_extended = r_cutoff / nb_intervals_extended ** (
                        gridpoint_indices_extended / nb_intervals_extended)
                # from pairs of neighboring r_shift values, compute eta values.
                # doing this for the nb_param_pairs + 1 values in the auxiliary
                # grid ultimately gives nb_param_pairs different values for eta.
                eta_grid = np.zeros(nb_param_pairs)
                for idx in range(len(rs_grid_extended) - 1):
                    eta_current = 1 / (rs_grid_extended[idx] - rs_grid_extended[idx + 1]) ** 2
                    eta_grid[idx] = eta_current
                # create final grid of r_shift values by excluding the first entry
                # (for which r_shift coincides with the cutoff radius) from the extended grid
                r_shift_grid = rs_grid_extended[1:]
                # reverse the order of r_shift and eta values so they are sorted in order of ascending r_shift
                # (not necessary, but makes the output consistent with the other options)
                r_shift_grid = np.flip(r_shift_grid)
                eta_grid = np.flip(eta_grid)
            else:
                raise ValueError('invalid argument for "mode"')
        else:
            raise ValueError('invalid argument for "rule"')

        self.r_shift_grid = r_shift_grid
        self.eta_grid = eta_grid

    def check_all(self):
        """Check if a complete symmetry function set, with all required settings for writing, has been generated.

        Raises
        -------
        TypeError
            If no set of radial parameters (r_shift and eta) has been generated
        TypeError
            If an angular symmetry function type has been set, but no values
            for zeta were set.

        Returns
        -------
        None
        """
        self.check_symfunc_type()
        # TODO: consider removing the check for radial_grid_info (since a user might want to directly set r_shift and eta values on their own, in which case no info would have been generated)
        # -> but how to handle setting (and storing) of r_cutoff in that case?
        if self.radial_grid_info is None or self.r_shift_grid is None or self.eta_grid is None:
            raise TypeError('No radial grid has been generated.')
        if self.symfunc_type in ['angular_narrow', 'angular_wide', 'weighted_angular']:
            if self._zetas is None:
                raise TypeError('zeta values not set.')

    def write_parameter_info(self, file=None):
        """Write settings used in generating the currently stored set of symmetry function parameters.

        Parameters
        ----------
        file : path-like, optional
            The file to write the settings information to, using append mode.
            If not specified, write to sys.stdout instead.

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

        # depending on presence of file parameter, either direct output
        # to stdout, or open the specified file
        if file is None:
            handle = sys.stdout
        else:
            handle = open(file, 'a')

        handle.write('#########################################################################\n')
        handle.write(
            f'# {type_descriptions[self.symfunc_type]} symmetry function set, '
            f'for elements {self.elements}\n')
        handle.write('#########################################################################\n')

        handle.write('# Radial parameters were generated using the following settings:\n')
        for key, value in self.radial_grid_info.items():
            handle.write(f'# {key:14s} = {value}\n')

        handle.write('# Sets of values for parameters:\n')
        # set numpy print precision to lower number of decimal places for the following outputs
        np.set_printoptions(precision=4)
        handle.write(f'# r_shift_grid   = {self.r_shift_grid}\n')
        handle.write(f'# eta_grid       = {self.eta_grid}\n')
        if self.symfunc_type in ['angular_narrow', 'angular_wide', 'weighted_angular']:
            handle.write(f'# lambdas        = {self.lambdas}\n')
            handle.write(f'# zetas          = {self.zetas}\n')
        # reset numpy print precision to default
        np.set_printoptions(precision=8)
        handle.write('\n')

        # close the file again (unless writing to sys.stdout,
        # which does not need to be closed in the same sense as a file object)
        if handle is not sys.stdout:
            handle.close()

    def find_element_combinations(self):
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
        """Write symmetry function parameter sets to stdout, formatted as in the file 'input.nn' required by n2p2.

        Each line in the output corresponds to one symmetry function.
        Output is formatted in blocks separated by blank lines, each block
        corresponding to one element combination.
        Within each block, the other parameters are iterated over.

        Returns
        -------
        None
        """
        self.check_all()

        r_cutoff = self.radial_grid_info['r_cutoff']
        sf_number = self.symfunc_type_numbers[self.symfunc_type]

        if self.symfunc_type == 'radial':
            for comb in self.element_combinations:
                for (eta, rs) in zip(self.eta_grid, self.r_shift_grid):
                    sys.stdout.write(
                        f'symfunction_short {comb[0]:2s} {sf_number} {comb[1]:2s} {eta:9.3E} {rs:9.3E} {r_cutoff:9.3E}\n')
                sys.stdout.write('\n')

        elif self.symfunc_type in ['angular_narrow', 'angular_wide']:
            for comb in self.element_combinations:
                for (eta, rs) in zip(self.eta_grid, self.r_shift_grid):
                    for zeta in self.zetas:
                        for lambd in self.lambdas:
                            sys.stdout.write(
                                f'symfunction_short {comb[0]:2s} {sf_number} {comb[1]:2s} {comb[2]:2s} {eta:9.3E} {lambd:2.0f} {zeta:9.3E} {r_cutoff:9.3E} {rs:9.3E}\n')
                sys.stdout.write('\n')

        elif self.symfunc_type == 'weighted_radial':
            for comb in self.element_combinations:
                for (eta, rs) in zip(self.eta_grid, self.r_shift_grid):
                    sys.stdout.write(
                        f'symfunction_short {comb[0]:2s} {sf_number} {eta:9.3E} {rs:9.3E} {r_cutoff:9.3E}\n')
                sys.stdout.write('\n')

        elif self.symfunc_type == 'weighted_angular':
            for comb in self.element_combinations:
                for (eta, rs) in zip(self.eta_grid, self.r_shift_grid):
                    for zeta in self.zetas:
                        for lambd in self.lambdas:
                            sys.stdout.write(
                                f'symfunction_short {comb[0]:2s} {sf_number} {eta:9.3E} {rs:9.3E} {lambd:2.0f} {zeta:9.3E} {r_cutoff:9.3E} \n')
                sys.stdout.write('\n')



def main():
    elems = ['S', 'Cu']
    # elems = ['H', 'C', 'O']
    myGen = SymFuncParamGenerator(elems)

    # # imbalzano2018 shift mode
    # myGen.symfunc_type = 'radial'
    # myGen.generate_radial_params(rule='imbalzano2018', mode='shift',
    #                              r_cutoff=6., nb_param_pairs=5)

    # # imbalzano2018 shift mode
    # myGen.symfunc_type = 'radial'
    # myGen.generate_radial_params(rule='gastegger2018', mode='shift',
    #                              r_cutoff=6., nb_param_pairs=5, r_lower=1.5,
    #                              r_upper=5.5)

    # # gastegger2018 shift mode
    # myGen.symfunc_type = 'angular_narrow'
    # myGen.generate_radial_params(rule='gastegger2018', mode='shift',
    #                              r_cutoff=6., nb_param_pairs=9, r_lower=1.5)
    # myGen.zetas = [1.0, 6.0]

    # gastegger2018 center mode
    myGen.symfunc_type = 'angular_wide'
    myGen.generate_radial_params(rule='gastegger2018', mode='center',
                                 r_cutoff=6., nb_param_pairs=3, r_lower=1.5)
    myGen.zetas = [1.0, 6.0]

    # # imbalzano2018 center mode
    # myGen.symfunc_type = 'weighted_radial'
    # myGen.generate_radial_params(rule='imbalzano2018', mode='center',
    #                              r_cutoff=5., nb_param_pairs=5)

    # # gastegger2018 shift mode
    # myGen.generate_radial_params(rule='gastegger2018', mode='shift',
    #                              r_cutoff=6., nb_param_pairs=3, r_lower=1.5)
    # myGen.symfunc_type = 'weighted_angular'
    # myGen.zetas = [1.0, 6.0]


    myGen.write_parameter_info()
    myGen.write_parameter_strings()


if __name__ == '__main__':
    main()
