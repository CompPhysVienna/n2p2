import sys
sys.path.append('../src')

import numpy as np
import pytest
import os
from sfparamgen import SymFuncParamGenerator
import filecmp


@pytest.fixture
def basic_generator():
    elems = ['S', 'Cu']
    return SymFuncParamGenerator(elements=elems, r_cutoff=11.22)


def test_write_settings_overview_file(basic_generator, tmpdir):
    ########### general settings ###########
    outfile_path = os.path.join(tmpdir, 'outfile.txt')
    reference_path = 'reference-output_write_settings_overview.txt'

    elems = ['S', 'Cu']
    basic_generator = SymFuncParamGenerator(elements=elems, r_cutoff=11.22)

    ########### using custom radial parameters ###########
    basic_generator.set_custom_radial_params(r_shift_values=[1,2], eta_values=[4,5])

    ## radial
    basic_generator.symfunc_type = 'radial'
    basic_generator.write_settings_overview(outfile_path)

    ## weighted_radial
    basic_generator.symfunc_type = 'weighted_radial'
    basic_generator.write_settings_overview(outfile_path)

    basic_generator.zetas = [5.5, 7.5]

    ## angular narrow
    basic_generator.symfunc_type = 'angular_narrow'
    basic_generator.write_settings_overview(outfile_path)

    ## angular_wide
    basic_generator.symfunc_type = 'angular_wide'
    basic_generator.write_settings_overview(outfile_path)

    ## weighted_angular
    basic_generator.symfunc_type = 'weighted_angular'
    basic_generator.write_settings_overview(outfile_path)

    ########### using the method for radial parameter generation ###########
    basic_generator.generate_radial_params(rule='gastegger2018', mode='center',
                                 nb_param_pairs=2, r_lower=1.5, r_upper=9.)

    ## radial
    basic_generator.symfunc_type = 'radial'
    basic_generator.write_settings_overview(outfile_path)

    ## weighted_radial
    basic_generator.symfunc_type = 'weighted_radial'
    basic_generator.write_settings_overview(outfile_path)

    basic_generator.zetas = [5.5, 7.5]

    ## angular narrow
    basic_generator.symfunc_type = 'angular_narrow'
    basic_generator.write_settings_overview(outfile_path)

    ## angular_wide
    basic_generator.symfunc_type = 'angular_wide'
    basic_generator.write_settings_overview(outfile_path)

    ## weighted_angular
    basic_generator.symfunc_type = 'weighted_angular'
    basic_generator.write_settings_overview(outfile_path)

    ########### test equality with target output ###########
    assert filecmp.cmp(outfile_path, reference_path)


def test_write_settings_overview_stdout(basic_generator, capsys):
    ########### general settings ###########
    reference_path = 'reference-output_write_settings_overview.txt'

    ########### using custom radial parameters ###########
    basic_generator.set_custom_radial_params(r_shift_values=[1,2], eta_values=[4,5])

    ## radial
    basic_generator.symfunc_type = 'radial'
    basic_generator.write_settings_overview()

    ## weighted_radial
    basic_generator.symfunc_type = 'weighted_radial'
    basic_generator.write_settings_overview()

    basic_generator.zetas = [5.5, 7.5]

    ## angular narrow
    basic_generator.symfunc_type = 'angular_narrow'
    basic_generator.write_settings_overview()

    ## angular_wide
    basic_generator.symfunc_type = 'angular_wide'
    basic_generator.write_settings_overview()

    ## weighted_angular
    basic_generator.symfunc_type = 'weighted_angular'
    basic_generator.write_settings_overview()

    ########### using the method for radial parameter generation ###########
    basic_generator.generate_radial_params(rule='gastegger2018', mode='center',
                                 nb_param_pairs=2, r_lower=1.5, r_upper=9.)

    ## radial
    basic_generator.symfunc_type = 'radial'
    basic_generator.write_settings_overview()

    ## weighted_radial
    basic_generator.symfunc_type = 'weighted_radial'
    basic_generator.write_settings_overview()

    basic_generator.zetas = [5.5, 7.5]

    ## angular narrow
    basic_generator.symfunc_type = 'angular_narrow'
    basic_generator.write_settings_overview()

    ## angular_wide
    basic_generator.symfunc_type = 'angular_wide'
    basic_generator.write_settings_overview()

    ## weighted_angular
    basic_generator.symfunc_type = 'weighted_angular'
    basic_generator.write_settings_overview()

    # capture the output written to stdout by settings writing method
    captured = capsys.readouterr()

    ########### test equality with target output ###########
    with open(reference_path, 'r') as f_reference:
        assert captured.out == f_reference.read()


