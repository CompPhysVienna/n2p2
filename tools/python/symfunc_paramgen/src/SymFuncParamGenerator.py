# -*- coding: utf-8 -*-

import numpy as np
import sys
import inspect
import itertools
import warnings


class SymFuncParamGenerator:
    """Tools for generation, storage, and writing in the format required by n2p2, of symmetry function parameter sets.

    Parameters
    ----------
    elements : list of string
        The chemical elements present in the system.
    r_cutoff : float
        Cutoff radius, at which symmetry functions go to zero.
        Must be greater than zero.

    Attributes
    ----------
    symfunc_type_numbers : dict
        Dictionary mapping strings specifying the symmetry function type to the
        numbers used internally by n2p2 to distinguish symmetry function types.
    lambdas : numpy.ndarray
        Set of values for the parameter lambda of angular symmetry functions. Fixed to [-1, 1].
    radial_paramgen_settings : dict or None
        Stores settings that were used in generating the symmetry function parameters r_shift and eta
        using the class method provided for that purpose.
        None, if no radial parameters have been generated yet, or if custom ones
        (without using the method for generating radial parameters) were set.
    _r_shift_grid : numpy.ndarray or None
        The set of values for the symmetry function parameter r_shift that was generated.
    _eta_grid : numpy.ndarray or None
        The set of values for the symmetry function parameter eta that was generated.
    elements
    element_combinations
    r_cutoff
    symfunc_type
    zetas
    """

    symfunc_type_numbers = dict(radial=2,
                             angular_narrow=3,
                             angular_wide=9,
                             weighted_radial=12,
                             weighted_angular=13)
    lambdas = np.array([-1.0, 1.0])

    def __init__(self, elements, r_cutoff: float):
        self._elements = elements
        if not r_cutoff > 0:
            raise ValueError('Invalid cutoff radius given. '
                             'Must be greater than zero.')
        else:
            self._r_cutoff = r_cutoff

        self._element_combinations = None
        self._symfunc_type = None
        self._zetas = None

        self._r_shift_grid = None
        self._eta_grid = None
        self.radial_paramgen_settings = None

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

        Raises
        ------
        ValueError
            If invalid symmetry function type is given.
        """
        return self._symfunc_type

    @symfunc_type.setter
    def symfunc_type(self, value):
        if value not in self.symfunc_type_numbers.keys():
            raise ValueError('Invalid symmetry function type. Must be one of {}'.format(
                list(self.symfunc_type_numbers.keys())))
        else:
            self._symfunc_type = value
        # once symmetry function type has been set and found to be valid,
        # build and store the element combinations
        self._element_combinations = self.find_element_combinations()
        # Clear any previous zeta values, if the given symfunc type is a radial one
        if value in ['radial', 'weighted_radial']:
            # set the member variable explicitly (with underscore) instead of
            # calling setter, because the setter would put None into an array
            self._zetas = None

    @property
    def r_cutoff(self):
        '''Cutoff radius where symmetry functions go to zero (float, read-only).
        '''
        return self._r_cutoff

    @property
    def zetas(self):
        """Set of values for the parameter zeta of angular symmetry functions (`numpy.ndarray`).
        """
        return self._zetas

    @zetas.setter
    def zetas(self, values):
        # TODO: possibly add checks on values for zeta
        self._zetas = np.array(values)

    @property
    def r_shift_grid(self):
        return self._r_shift_grid

    @property
    def eta_grid(self):
        return self._eta_grid

    def check_symfunc_type(self, calling_method_name=None):
        """Check if a symmetry function type has been set.

        Parameters
        ----------
        calling_method_name : string, optional
            The name of another method that calls this method.
            If this parameter is given, a modified error message is printed
            by this method, mentioning the method from which it was called.
            This should make it clearer to the user in which part
            of their own code to look for an error.

        Raises
        ------
        TypeError
            If the symmetry function type has not been set (i.e., it is None).

        Returns
        -------
        None
        """
        if self.symfunc_type is None:
            if calling_method_name is None:
                raise ValueError('Symmetry function type not set.')
            else:
                raise ValueError(f'Symmetry function type not set. '
                                f'Calling method {calling_method_name} requires'
                                f' that symmetry function type have been set before.')

    def generate_radial_params(self, rule, mode, nb_param_pairs: int,
                               r_lower=None, r_upper=None):
        """Generate a set of values for r_shift and eta.

        Such a set of (r_shift, eta)-values is required for any
        symmetry function type.
        Its generation is independent of the symmetry function type
        and the angular symmetry function parameters zeta and lambda.

        Rules for parameter generation are implemented based on [1]_ and [2]_.

        The generated values are stored as arrays in the member variables _r_shift_grid and _eta_grid.
        The entries are to be understood pairwise, i.e., the i-th entry of _r_shift_grid
        and the i-th entry of _eta_grid belong to one symmetry function.
        Besides the set of values for r_shift and eta, the settings that
        were used for generating it are also stored, in the dictionary radial_paramgen_settings.

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
        nb_param_pairs : int
            Number of (r_shift, eta)-pairs to be generated.
        r_lower : float
            lowest value in the radial grid from which r_shift and eta values
            are computed.
            required if rule=='gastegger2018'.
            ignored if rule=='imbalzano2018'.
        r_upper : float, optional
            Largest value in the radial grid from which r_shift and eta
            values are computed.
            Optional if rule=='gastegger2018, defaults to cutoff radius if not
            given.
            Ignored if rule=='imbalzano2018'.

        Raises
        ------
        ValueError
            If nb_param_pairs is not two or greater.
        TypeError
            If parameter r_lower is not given, when using rule 'gastegger2018'.
        ValueError
            If illegal relation between r_lower, r_upper, r_cutoff.
        ValueError
            If invalid argument for parameters rule or mode.

        Notes
        -----
        [1] https://doi.org/10.1063/1.5019667
        [2] https://doi.org/10.1063/1.5024611

        The parameter nb_param_pairs invariably specifies the number of
        (r_shift, eta)-pairs ultimately generated, not the number of
        intervals between points in a grid of r_shift values, or the
        number of points in some auxiliary grid.
        This constitutes a slight modification of the nomenclature in [2]_,
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
        if not nb_param_pairs >= 2:
            raise ValueError('nb_param_pairs must be two or greater.')

        # store those infos on radial parameter generation settings that are
        # independent of the rule argument
        self.radial_paramgen_settings = dict(rule=rule,
                                             mode=mode,
                                             nb_param_pairs=nb_param_pairs)

        r_cutoff = self.r_cutoff

        if rule == 'gastegger2018':
            if r_lower is None:
                raise TypeError('Argument r_lower is required for rule "gastegger2018"')
            if r_upper is None:
                # by default, set largest value of radial grid to cutoff radius
                r_upper = r_cutoff

            # store those settings that are unique to this rule
            self.radial_paramgen_settings.update({'r_lower': r_lower, 'r_upper': r_upper})

            # create auxiliary grid
            grid = np.linspace(r_lower, r_upper, nb_param_pairs)

            if mode == 'center':
                # r_lower = 0 is not allowed in center mode,
                # because it causes division by zero
                if not 0 < r_lower < r_upper <= r_cutoff:
                    raise ValueError(f'Invalid argument(s): rule = {rule:s}, mode = {mode:s} requires that 0 < r_lower < r_upper <= r_cutoff.')
                r_shift_grid = np.zeros(nb_param_pairs)
                eta_grid = 1.0 / (2.0 * grid ** 2)
            elif mode == 'shift':
                # on the other hand, in shift mode, r_lower = 0 is possible
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
                this_method_name = inspect.currentframe().f_code.co_name
                warnings.warn(f'The argument r_lower to method'
                              f' {this_method_name} will be ignored,'
                              f' since it is unused when calling the method'
                              f' with rule="imbalzano2018".')
            if r_upper is not None:
                this_method_name = inspect.currentframe().f_code.co_name
                warnings.warn(f'The argument r_upper to method'
                              f' {this_method_name} will be ignored,'
                              f' since it is unused when calling the method'
                              f' with rule="imbalzano2018".')

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

        # store the generated parameter sets
        self._r_shift_grid = r_shift_grid
        self._eta_grid = eta_grid

    def set_custom_radial_params(self, r_shift_values, eta_values):
        """Set custom r_shift and eta, bypassing the class's generation method.

        The parameters r_shift_values and eta_values must have the same
        length.
        Ttheir entries are to be understood pairwise, i.e.,
        the i-th entry of r_shift_values and the i-th entry of eta_values
        belong together, describing one symmetry function.

        Parameters
        ----------
        r_shift_values : sequence of float or 1D-array
            Set of values for the symmetry function parameter r_shift.
        eta_values : sequence of float or 1D-array
            Set of values for the symmetry function parameter eta.

        Raises
        ------
        TypeError
            If r_shift_values and eta_values do not have equal length.
        ValueError
            If r_shift_values or eta_values contain non-positive entries.

        Notes
        -----
        Setting r_shift and eta manually via this method instead of using the
        method generate_radial_params somewhat defeats the purpose of the
        class as a generator of symmetry function parameter values. However,
        it might still be useful, in case one wants to use custom values for
        r_shift and eta, for which the generation is not implemented as a
        class method, while still benefiting from the  parameter writing
        functionality of the class.

        Returns
        -------
        None
        """
        if len(r_shift_values) != len(eta_values):
            raise TypeError('r_shift_values and eta_values must have same length.')
        if min(r_shift_values) <= 0:
            raise ValueError('r_shift_values must all be greater than zero.')
        if min(eta_values) <= 0:
            raise ValueError('eta_values must all be greater than zero.')
        # (re)set radial_paramgen_settings to None, indicating that custom
        # values for (r_shift, eta) are used, rather than ones generated by
        # class method.
        self.radial_paramgen_settings = None
        # set the values
        self._r_shift_grid = np.array(r_shift_values)
        self._eta_grid = np.array(eta_values)

    def check_writing_prerequisites(self, calling_method_name=None):
        """Check if all data required for writing symmetry function sets are present.

        This comprises checking if the following have been set:
        -) symmetry function type
        -) values for r_shift and eta
        -) values for zeta, if the symmetry function type is an angular one

        Parameters
        ----------
        calling_method_name : string, optional
            The name of another method that calls this method.
            If this parameter is given, a modified error message is printed
            by this method, mentioning the method from which it was called.
            This should make it clearer to the user in which part
            of their own code to look for an error.

        Raises
        ------
        TypeError
            If values for r_shift or eta are not set.
        TypeError
            If an angular symmetry function type has been set, but no values
            for zeta were set.

        Returns
        -------
        None
        """
        self.check_symfunc_type(calling_method_name=calling_method_name)

        if calling_method_name is None:
            if self._r_shift_grid is None or self._eta_grid is None:
                raise ValueError('Values for r_shift and/or eta not set.')
            if self.symfunc_type in ['angular_narrow', 'angular_wide', 'weighted_angular']:
                if self.zetas is None:
                    raise ValueError(f'Values for zeta not set (required for symmetry'
                                    f' function type {self.symfunc_type}).')
        else:
            if self._r_shift_grid is None or self._eta_grid is None:
                raise ValueError(f'Values for r_shift and/or eta not set. '
                                f'Calling method {calling_method_name} requires'
                                f' that values for r_shift and eta have been set before.')
            if self.symfunc_type in ['angular_narrow', 'angular_wide', 'weighted_angular']:
                if self.zetas is None:
                    raise ValueError(f'Values for zeta not set. '
                                    f'Calling method {calling_method_name} requires'
                                    f' zetas to have been set before, when using'
                                    f' symmetry function type {self.symfunc_type}.')

    def write_settings_overview(self, file=None):
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
        this_method_name = inspect.currentframe().f_code.co_name
        self.check_writing_prerequisites(calling_method_name=this_method_name)

        type_descriptions = dict(radial='Radial',
                                 angular_narrow='Narrow angular',
                                 angular_wide='Wide angular',
                                 weighted_radial='Weighted radial',
                                 weighted_angular='Weighted angular')

        # depending on presence of file parameter, either direct output
        # to stdout, or open the specified file, in append mode
        handle = open(file, 'a') if file is not None else sys.stdout

        handle.write('#########################################################################\n')
        handle.write(
            f'# {type_descriptions[self.symfunc_type]} symmetry function set, '
            f'for elements {self.elements}\n')
        handle.write('#########################################################################\n')

        handle.write(f'# r_cutoff       = {self.r_cutoff}\n')

        # depending on whether radial parameters were generated using the method
        # or custom-set (indicated by presence or absence of radial parameter
        # generation settings), write the settings used or not
        if self.radial_paramgen_settings is not None:
            handle.write('# The following settings were used for generating sets\n')
            handle.write('# of values for the radial parameters r_shift and eta:\n')
            for key, value in self.radial_paramgen_settings.items():
                handle.write(f'# {key:14s} = {value}\n')
        else:
            handle.write('# A custom set of values was used for the radial parameters r_shift and eta.\n')
            handle.write('# Thus, there are no settings on radial parameter generation available for display.\n')

        handle.write('# Sets of values for parameters:\n')
        # set numpy print precision to lower number of decimal places for the following outputs
        np.set_printoptions(precision=4)
        handle.write(f'# r_shift_grid   = {self._r_shift_grid}\n')
        handle.write(f'# eta_grid       = {self._eta_grid}\n')
        if self.symfunc_type in ['angular_narrow', 'angular_wide', 'weighted_angular']:
            handle.write(f'# lambdas        = {self.lambdas}\n')
            handle.write(f'# zetas          = {self.zetas}\n')
        # reset numpy print precision to default
        np.set_printoptions(precision=8)
        handle.write('\n')

        # close the file again (unless writing to sys.stdout,
        # which should not be closed)
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
        this_method_name = inspect.currentframe().f_code.co_name
        self.check_symfunc_type(calling_method_name=this_method_name)

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

    def write_parameter_strings(self, file=None):
        """Write symmetry function parameter sets, formatted as n2p2 requires.

        Each line in the output corresponds to one symmetry function.
        Output is formatted in blocks separated by blank lines, each block
        corresponding to one element combination.
        Within each block, the other parameters are iterated over.

        Parameters
        ----------
        file : path-like, optional
            The file to write the parameter strings to, using append mode.
            If not specified, write to sys.stdout instead.

        Returns
        -------
        None
        """
        this_method_name = inspect.currentframe().f_code.co_name
        self.check_writing_prerequisites(calling_method_name=this_method_name)

        # depending on presence of file parameter, either direct output
        # to stdout, or open the specified file, in append mode
        handle = open(file, 'a') if file is not None else sys.stdout

        r_cutoff = self.r_cutoff
        sf_number = self.symfunc_type_numbers[self.symfunc_type]

        if self.symfunc_type == 'radial':
            for comb in self.element_combinations:
                for (eta, rs) in zip(self._eta_grid, self._r_shift_grid):
                    handle.write(
                        f'symfunction_short {comb[0]:2s} {sf_number} {comb[1]:2s} {eta:9.3E} {rs:9.3E} {r_cutoff:9.3E}\n')
                handle.write('\n')

        elif self.symfunc_type in ['angular_narrow', 'angular_wide']:
            for comb in self.element_combinations:
                for (eta, rs) in zip(self._eta_grid, self._r_shift_grid):
                    for zeta in self.zetas:
                        for lambd in self.lambdas:
                            handle.write(
                                f'symfunction_short {comb[0]:2s} {sf_number} {comb[1]:2s} {comb[2]:2s} {eta:9.3E} {lambd:2.0f} {zeta:9.3E} {r_cutoff:9.3E} {rs:9.3E}\n')
                handle.write('\n')

        elif self.symfunc_type == 'weighted_radial':
            for comb in self.element_combinations:
                for (eta, rs) in zip(self._eta_grid, self._r_shift_grid):
                    handle.write(
                        f'symfunction_short {comb[0]:2s} {sf_number} {eta:9.3E} {rs:9.3E} {r_cutoff:9.3E}\n')
                handle.write('\n')

        elif self.symfunc_type == 'weighted_angular':
            for comb in self.element_combinations:
                for (eta, rs) in zip(self._eta_grid, self._r_shift_grid):
                    for zeta in self.zetas:
                        for lambd in self.lambdas:
                            handle.write(
                                f'symfunction_short {comb[0]:2s} {sf_number} {eta:9.3E} {rs:9.3E} {lambd:2.0f} {zeta:9.3E} {r_cutoff:9.3E} \n')
                handle.write('\n')

        # close the file again (unless writing to sys.stdout,
        # which should not be closed)
        if handle is not sys.stdout:
            handle.close()
