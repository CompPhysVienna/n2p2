import sys
sys.path.append('../src')

import pytest
import os
import SymFuncParamGenerator as sfpg
import filecmp


def test_element_combinations():
    elems = ['S', 'Cu']
    myGen = sfpg.SymFuncParamGenerator(elems)
    myGen.symfunc_type = 'radial'
    assert myGen.element_combinations == [('S', 'S'), ('S', 'Cu'), ('Cu', 'S'), ('Cu', 'Cu')]


def isnotcomment(line):
    if line[0] == '#':
        return False
    return True


def test_output(tmpdir):
    '''To detect changes in parameter generation and writing, compared to older versions.

    '''
    elems = ['S', 'Cu']
    myGen = sfpg.SymFuncParamGenerator(elems)

    outfile_path = os.path.join(tmpdir, 'outfile.txt')

    myGen.symfunc_type = 'radial'
    myGen.generate_radial_params(rule='imbalzano2018', mode='shift',
                                 r_cutoff=6., nb_param_pairs=5)
    myGen.write_parameter_info(outfile_path)
    myGen.write_parameter_strings(outfile_path)

    myGen.symfunc_type = 'radial'
    myGen.generate_radial_params(rule='gastegger2018', mode='shift',
                                 r_cutoff=6., nb_param_pairs=5, r_lower=1.5,
                                 r_upper=5.5)
    myGen.write_parameter_info(outfile_path)
    myGen.write_parameter_strings(outfile_path)

    myGen.symfunc_type = 'angular_narrow'
    myGen.generate_radial_params(rule='gastegger2018', mode='shift',
                                 r_cutoff=6., nb_param_pairs=9, r_lower=1.5)
    myGen.zetas = [1.0, 6.0]
    myGen.write_parameter_info(outfile_path)
    myGen.write_parameter_strings(outfile_path)

    myGen.symfunc_type = 'angular_wide'
    myGen.generate_radial_params(rule='gastegger2018', mode='center',
                                 r_cutoff=6., nb_param_pairs=3, r_lower=1.5)
    myGen.zetas = [1.0, 6.0]
    myGen.write_parameter_info(outfile_path)
    myGen.write_parameter_strings(outfile_path)

    myGen.symfunc_type = 'weighted_radial'
    myGen.generate_radial_params(rule='imbalzano2018', mode='center',
                                 r_cutoff=5., nb_param_pairs=5)
    myGen.write_parameter_info(outfile_path)
    myGen.write_parameter_strings(outfile_path)

    myGen.generate_radial_params(rule='gastegger2018', mode='shift',
                                 r_cutoff=6., nb_param_pairs=3, r_lower=1.5)
    myGen.symfunc_type = 'weighted_angular'
    myGen.zetas = [1.0, 6.0]
    myGen.write_parameter_info(outfile_path)
    myGen.write_parameter_strings(outfile_path)

    # ignore comment lines, so the test does not immediately fail when only
    # the parameter informations are changed (which I would like to keep just
    # for keeping track of the settings used in the reference output file, but
    # are otherwise not essential)
    with open(outfile_path) as f_out, open('reference_outputs.txt') as f_reference:
        f_reference = filter(isnotcomment, f_reference)
        f_out = filter(isnotcomment, f_out)
        assert all(x == y for x, y in zip(f_out, f_reference))

