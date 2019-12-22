import sys
sys.path.append('../src')

import numpy as np
import pytest
import os
import sys
from sfparamgen import SymFuncParamGenerator


REFERENCE_PATH = 'reference-output_write_parameter_strings.txt'


@pytest.fixture
def basic_generator():
    elems = ['S', 'Cu']
    return SymFuncParamGenerator(elements=elems, r_cutoff=11.22)


def isnotcomment(line):
    """Filter lines according to whether they start in '#'.
    """
    if line[0] == '#':
        return False
    return True


def test_write_parameter_strings_file(basic_generator, tmpdir):
    '''Test parameter strings written to file match reference, for toy inputs.
    '''
    outfile_path = os.path.join(tmpdir, 'outfile.txt')

    basic_generator.set_custom_radial_params([1,2], [4,5])

    with open(outfile_path, 'w') as outfile_obj:
        ## radial
        basic_generator.symfunc_type = 'radial'
        basic_generator.write_parameter_strings(outfile_obj)

        ## weighted_radial
        basic_generator.symfunc_type = 'weighted_radial'
        basic_generator.write_parameter_strings(outfile_obj)

        ## angular narrow
        basic_generator.symfunc_type = 'angular_narrow'
        basic_generator.zetas = [5.5, 7.5]
        basic_generator.write_parameter_strings(outfile_obj)

        ## angular_wide
        basic_generator.symfunc_type = 'angular_wide'
        basic_generator.zetas = [5.5, 7.5]
        basic_generator.write_parameter_strings(outfile_obj)

        ## weighted_angular
        basic_generator.symfunc_type = 'weighted_angular'
        basic_generator.zetas = [5.5, 7.5]
        basic_generator.write_parameter_strings(outfile_obj)

    # Ignore comment lines in the reference output file (lines starting
    # with '#'). These lines are included in the reference output file as a
    # reminder of what settings to use in the tests to recreate the reference
    # output, but they are otherwise not essential for the test.
    with open(outfile_path) as f_out, open(REFERENCE_PATH) as f_reference:
        f_reference = filter(isnotcomment, f_reference)
        assert all(x == y for x, y in zip(f_out, f_reference))


def test_write_parameter_strings_stdout(basic_generator, capsys):
    '''Test parameter strings written to stdout match reference, for toy inputs.
    '''
    basic_generator.set_custom_radial_params([1,2], [4,5])

    ## radial
    basic_generator.symfunc_type = 'radial'
    basic_generator.write_parameter_strings()

    ## weighted_radial
    basic_generator.symfunc_type = 'weighted_radial'
    basic_generator.write_parameter_strings()

    ## angular narrow
    basic_generator.symfunc_type = 'angular_narrow'
    basic_generator.zetas = [5.5, 7.5]
    basic_generator.write_parameter_strings()

    ## angular_wide
    basic_generator.symfunc_type = 'angular_wide'
    basic_generator.zetas = [5.5, 7.5]
    basic_generator.write_parameter_strings()

    ## weighted_angular
    basic_generator.symfunc_type = 'weighted_angular'
    basic_generator.zetas = [5.5, 7.5]
    basic_generator.write_parameter_strings()

    # capture the output written to stdout by parameter string writing method
    captured = capsys.readouterr()

    # compare what was written to stdout with the reference output file
    reference_data = []
    with open(REFERENCE_PATH, 'r') as f_reference:
        for line in f_reference:
            # ignore comment lines in reference output file
            if not line[0] == '#':
                reference_data.append(line)
        assert ''.join(reference_data) == captured.out


