import sys
sys.path.append('../src')

import numpy as np
import pytest
import os
import SymFuncParamGenerator as sfpg


def isnotcomment(line):
    """Filter lines according to whether they start in '#'.
    """
    if line[0] == '#':
        return False
    return True


def test_write_parameter_strings(tmpdir):
    '''Test parameter strings are the same as in reference file, for toy inputs.
    '''
    elems = ['S', 'Cu']

    outfile_path = os.path.join(tmpdir, 'outfile.txt')
    reference_path = 'reference-output_write_parameter_strings.txt'

    elems = ['S', 'Cu']
    myGen = sfpg.SymFuncParamGenerator(elements=elems, r_cutoff=11.22)

    myGen.set_custom_radial_params([1,2], [4,5])

    ## radial
    myGen.symfunc_type = 'radial'
    myGen.write_settings_overview(outfile_path)
    myGen.write_parameter_strings(outfile_path)

    ## weighted_radial
    myGen.symfunc_type = 'weighted_radial'
    myGen.write_settings_overview(outfile_path)
    myGen.write_parameter_strings(outfile_path)

    ## angular narrow
    myGen.symfunc_type = 'angular_narrow'
    myGen.zetas = [5.5, 7.5]
    myGen.write_settings_overview(outfile_path)
    myGen.write_parameter_strings(outfile_path)

    ## angular_wide
    myGen.symfunc_type = 'angular_wide'
    myGen.zetas = [5.5, 7.5]
    myGen.write_settings_overview(outfile_path)
    myGen.write_parameter_strings(outfile_path)

    ## weighted_angular
    myGen.symfunc_type = 'weighted_angular'
    myGen.zetas = [5.5, 7.5]
    myGen.write_settings_overview(outfile_path)
    myGen.write_parameter_strings(outfile_path)

    # ignore comment lines, so the test does not immediately fail when only
    # the parameter information is changed (which I would like to keep just
    # for keeping track of the settings used in the reference output file, but
    # is otherwise not essential)
    with open(outfile_path) as f_out, open(reference_path) as f_reference:
        f_reference = filter(isnotcomment, f_reference)
        f_out = filter(isnotcomment, f_out)
        assert all(x == y for x, y in zip(f_out, f_reference))