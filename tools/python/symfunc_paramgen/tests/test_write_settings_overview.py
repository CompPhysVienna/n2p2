import sys
sys.path.append('../src')

import numpy as np
import pytest
import os
import SymFuncParamGenerator as sfpg
import filecmp


def test_write_settings_overview_file(tmpdir):
    ########### general settings ###########
    outfile_path = os.path.join(tmpdir, 'outfile.txt')
    reference_path = 'reference-output_write_settings_overview.txt'

    elems = ['S', 'Cu']
    myGen = sfpg.SymFuncParamGenerator(elements=elems, r_cutoff=11.22)

    ########### using custom radial parameters ###########
    myGen.set_custom_radial_params(r_shift_values=[1,2], eta_values=[4,5])

    ## radial
    myGen.symfunc_type = 'radial'
    myGen.write_settings_overview(outfile_path)

    ## weighted_radial
    myGen.symfunc_type = 'weighted_radial'
    myGen.write_settings_overview(outfile_path)

    myGen.zetas = [5.5, 7.5]

    ## angular narrow
    myGen.symfunc_type = 'angular_narrow'
    myGen.write_settings_overview(outfile_path)

    ## angular_wide
    myGen.symfunc_type = 'angular_wide'
    myGen.write_settings_overview(outfile_path)

    ## weighted_angular
    myGen.symfunc_type = 'weighted_angular'
    myGen.write_settings_overview(outfile_path)

    ########### using the method for radial parameter generation ###########
    myGen.generate_radial_params(rule='gastegger2018', mode='center',
                                 nb_param_pairs=2, r_lower=1.5, r_upper=9.)

    ## radial
    myGen.symfunc_type = 'radial'
    myGen.write_settings_overview(outfile_path)

    ## weighted_radial
    myGen.symfunc_type = 'weighted_radial'
    myGen.write_settings_overview(outfile_path)

    myGen.zetas = [5.5, 7.5]

    ## angular narrow
    myGen.symfunc_type = 'angular_narrow'
    myGen.write_settings_overview(outfile_path)

    ## angular_wide
    myGen.symfunc_type = 'angular_wide'
    myGen.write_settings_overview(outfile_path)

    ## weighted_angular
    myGen.symfunc_type = 'weighted_angular'
    myGen.write_settings_overview(outfile_path)

    ########### test equality with target output ###########
    assert filecmp.cmp(outfile_path, reference_path)


def test_write_settings_overview_stdout(capsys):
    ########### general settings ###########
    reference_path = 'reference-output_write_settings_overview.txt'

    elems = ['S', 'Cu']
    myGen = sfpg.SymFuncParamGenerator(elements=elems, r_cutoff=11.22)

    ########### using custom radial parameters ###########
    myGen.set_custom_radial_params(r_shift_values=[1,2], eta_values=[4,5])

    ## radial
    myGen.symfunc_type = 'radial'
    myGen.write_settings_overview()

    ## weighted_radial
    myGen.symfunc_type = 'weighted_radial'
    myGen.write_settings_overview()

    myGen.zetas = [5.5, 7.5]

    ## angular narrow
    myGen.symfunc_type = 'angular_narrow'
    myGen.write_settings_overview()

    ## angular_wide
    myGen.symfunc_type = 'angular_wide'
    myGen.write_settings_overview()

    ## weighted_angular
    myGen.symfunc_type = 'weighted_angular'
    myGen.write_settings_overview()

    ########### using the method for radial parameter generation ###########
    myGen.generate_radial_params(rule='gastegger2018', mode='center',
                                 nb_param_pairs=2, r_lower=1.5, r_upper=9.)

    ## radial
    myGen.symfunc_type = 'radial'
    myGen.write_settings_overview()

    ## weighted_radial
    myGen.symfunc_type = 'weighted_radial'
    myGen.write_settings_overview()

    myGen.zetas = [5.5, 7.5]

    ## angular narrow
    myGen.symfunc_type = 'angular_narrow'
    myGen.write_settings_overview()

    ## angular_wide
    myGen.symfunc_type = 'angular_wide'
    myGen.write_settings_overview()

    ## weighted_angular
    myGen.symfunc_type = 'weighted_angular'
    myGen.write_settings_overview()

    # capture the output written to stdout by settings writing method
    captured = capsys.readouterr()

    ########### test equality with target output ###########
    with open(reference_path, 'r') as f_reference:
        assert captured.out == f_reference.read()


