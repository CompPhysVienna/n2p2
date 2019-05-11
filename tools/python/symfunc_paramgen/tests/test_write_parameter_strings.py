import sys
sys.path.append('../src')

import numpy as np
import pytest
import os
import sys
import SymFuncParamGenerator as sfpg


def isnotcomment(line):
    """Filter lines according to whether they start in '#'.
    """
    if line[0] == '#':
        return False
    return True


def test_write_parameter_strings_file(tmpdir):
    '''Test parameter strings written to file match reference file, for toy inputs.
    '''
    outfile_path = os.path.join(tmpdir, 'outfile.txt')
    reference_path = 'reference-output_write_parameter_strings.txt'

    elems = ['S', 'Cu']
    myGen = sfpg.SymFuncParamGenerator(elements=elems, r_cutoff=11.22)

    myGen.set_custom_radial_params([1,2], [4,5])

    ## radial
    myGen.symfunc_type = 'radial'
    myGen.write_parameter_strings(outfile_path)

    ## weighted_radial
    myGen.symfunc_type = 'weighted_radial'
    myGen.write_parameter_strings(outfile_path)

    ## angular narrow
    myGen.symfunc_type = 'angular_narrow'
    myGen.zetas = [5.5, 7.5]
    myGen.write_parameter_strings(outfile_path)

    ## angular_wide
    myGen.symfunc_type = 'angular_wide'
    myGen.zetas = [5.5, 7.5]
    myGen.write_parameter_strings(outfile_path)

    ## weighted_angular
    myGen.symfunc_type = 'weighted_angular'
    myGen.zetas = [5.5, 7.5]
    myGen.write_parameter_strings(outfile_path)

    # Ignore comment lines in the reference output file (lines starting
    # with '#'). These lines are included in the reference output file as a
    # reminder of what settings to use in the tests to recreate the reference
    # output, but they are otherwise not essential for the test.
    with open(outfile_path) as f_out, open(reference_path) as f_reference:
        f_reference = filter(isnotcomment, f_reference)
        assert all(x == y for x, y in zip(f_out, f_reference))


def test_write_parameter_strings_stdout(capsys):
    '''Test parameter strings written to stdout match reference file, for toy inputs.
    '''
    reference_path = 'reference-output_write_parameter_strings.txt'

    elems = ['S', 'Cu']
    myGen = sfpg.SymFuncParamGenerator(elements=elems, r_cutoff=11.22)

    myGen.set_custom_radial_params([1,2], [4,5])

    ## radial
    myGen.symfunc_type = 'radial'
    myGen.write_parameter_strings()

    ## weighted_radial
    myGen.symfunc_type = 'weighted_radial'
    myGen.write_parameter_strings()

    ## angular narrow
    myGen.symfunc_type = 'angular_narrow'
    myGen.zetas = [5.5, 7.5]
    myGen.write_parameter_strings()

    ## angular_wide
    myGen.symfunc_type = 'angular_wide'
    myGen.zetas = [5.5, 7.5]
    myGen.write_parameter_strings()

    ## weighted_angular
    myGen.symfunc_type = 'weighted_angular'
    myGen.zetas = [5.5, 7.5]
    myGen.write_parameter_strings()

    # capture the output written to stdout by parameter string writing method
    captured = capsys.readouterr()

    # compare what was written to stdout with the reference output file
    reference_data = []
    with open(reference_path, 'r') as f_reference:
        for line in f_reference:
            # ignore comment lines in reference output file
            if not line[0] == '#':
                reference_data.append(line)
        assert ''.join(reference_data) == captured.out


