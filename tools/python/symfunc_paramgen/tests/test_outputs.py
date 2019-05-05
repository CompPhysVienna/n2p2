import sys
sys.path.append('../src')

import pytest
import os
import SymFuncParamGenerator as sfpg


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
    myGen.r_cutoff = 6.
    myGen.generate_radial_params(rule='imbalzano2018', mode='shift',
                                 nb_param_pairs=5)
    myGen.write_settings_overview(outfile_path)
    myGen.write_parameter_strings(outfile_path)

    myGen.symfunc_type = 'radial'
    myGen.r_cutoff = 6.
    myGen.generate_radial_params(rule='gastegger2018', mode='shift',
                                 nb_param_pairs=5, r_lower=1.5, r_upper=5.5)
    myGen.write_settings_overview(outfile_path)
    myGen.write_parameter_strings(outfile_path)

    myGen.symfunc_type = 'angular_narrow'
    myGen.r_cutoff = 6.
    myGen.generate_radial_params(rule='gastegger2018', mode='shift',
                                 nb_param_pairs=9, r_lower=1.5)
    myGen.zetas = [1.0, 6.0]
    myGen.write_settings_overview(outfile_path)
    myGen.write_parameter_strings(outfile_path)

    myGen.symfunc_type = 'angular_wide'
    myGen.r_cutoff = 6.
    myGen.generate_radial_params(rule='gastegger2018', mode='center',
                                 nb_param_pairs=3, r_lower=1.5)
    myGen.zetas = [1.0, 6.0]
    myGen.write_settings_overview(outfile_path)
    myGen.write_parameter_strings(outfile_path)

    myGen.symfunc_type = 'weighted_radial'
    myGen.r_cutoff = 5.
    myGen.generate_radial_params(rule='imbalzano2018', mode='center',
                                 nb_param_pairs=5)
    myGen.write_settings_overview(outfile_path)
    myGen.write_parameter_strings(outfile_path)

    myGen.symfunc_type = 'weighted_angular'
    myGen.r_cutoff = 6.
    myGen.generate_radial_params(rule='gastegger2018', mode='shift',
                                 nb_param_pairs=3, r_lower=1.5)
    myGen.zetas = [1.0, 6.0]
    myGen.write_settings_overview(outfile_path)
    myGen.write_parameter_strings(outfile_path)

    # ignore comment lines, so the test does not immediately fail when only
    # the parameter informations are changed (which I would like to keep just
    # for keeping track of the settings used in the reference output file, but
    # are otherwise not essential)
    with open(outfile_path) as f_out, open('reference_outputs.txt') as f_reference:
        f_reference = filter(isnotcomment, f_reference)
        f_out = filter(isnotcomment, f_out)
        assert all(x == y for x, y in zip(f_out, f_reference))


def test_setter_sf_type():
    elems = ['S', 'Cu']
    myGen = sfpg.SymFuncParamGenerator(elems)
    with pytest.raises(ValueError):
        myGen.symfunc_type = 'illegal_type'


def test_setter_rcutoff():
    elems = ['S', 'Cu']
    myGen = sfpg.SymFuncParamGenerator(elems)
    with pytest.raises(ValueError):
        myGen.r_cutoff = 0
    with pytest.raises(ValueError):
        myGen.r_cutoff = -5


@pytest.mark.parametrize("rule,mode,nb_param_pairs,r_lower", [
    ('gastegger2018', 'center', 5, 1.5),
    ('gastegger2018', 'shift', 5, 1.5),
    ('imbalzano2018', 'center', 5, None),
    ('imbalzano2018', 'shift', 5, None)
])
def test_radial_paramgen_cutoff(rule, mode, nb_param_pairs, r_lower):
    """Test for TypeError when generating radial params without having set r_cutoff.
    """
    elems = ['S', 'Cu']
    myGen = sfpg.SymFuncParamGenerator(elems)
    with pytest.raises(TypeError):
        myGen.generate_radial_params(rule=rule, mode=mode,
                                     nb_param_pairs=nb_param_pairs, r_lower=r_lower)


@pytest.mark.parametrize("rule,mode,nb_param_pairs,r_lower", [
    ('bad_argument', 'center', 5, 1.5),
    ('bad_argument', 'shift', 5, 1.5),
    ('gastegger2018', 'bad_argument', 5, 1.5),
    ('imbalzano2018', 'bad_argument', 5, None)
])
def test_radial_paramgen_2(rule, mode, nb_param_pairs, r_lower):
    """Test for ValueError when invalid arguments for rule or method.
    """
    elems = ['S', 'Cu']
    myGen = sfpg.SymFuncParamGenerator(elems)
    myGen.r_cutoff = 6.
    with pytest.raises(ValueError):
        myGen.generate_radial_params(rule=rule, mode=mode,
                                     nb_param_pairs=nb_param_pairs, r_lower=r_lower)


@pytest.mark.parametrize("rule,mode,nb_param_pairs", [
    ('gastegger2018', 'center', 5),
    ('gastegger2018', 'shift', 5)
])
def test_radial_paramgen_3(rule, mode, nb_param_pairs):
    """Test for TypeError when omitting r_lower argument in rule 'gastegger2018'
    """
    elems = ['S', 'Cu']
    myGen = sfpg.SymFuncParamGenerator(elems)
    myGen.r_cutoff = 6.
    with pytest.raises(TypeError):
        myGen.generate_radial_params(rule=rule, mode=mode,
                                     nb_param_pairs=nb_param_pairs)


@pytest.mark.parametrize("rule,mode,nb_param_pairs,r_lower", [
    ('gastegger2018', 'center', 1, 1.5),
    ('gastegger2018', 'shift', 1, 1.5),
    ('imbalzano2018', 'center', 1, None),
    ('imbalzano2018', 'shift', 1, None)
])
def test_radial_paramgen_4(rule, mode, nb_param_pairs, r_lower):
    """Test for ValueError when nb_param_pairs less than two.
    """
    elems = ['S', 'Cu']
    myGen = sfpg.SymFuncParamGenerator(elems)
    myGen.r_cutoff = 6.
    with pytest.raises(ValueError):
        myGen.generate_radial_params(rule=rule, mode=mode,
                                     nb_param_pairs=nb_param_pairs, r_lower=r_lower)


@pytest.mark.parametrize("rule,mode,nb_param_pairs,r_lower", [
    ('gastegger2018', 'center', 1, 1.5),
    ('gastegger2018', 'center', 1, 1.5),
    ('gastegger2018', 'center', 1, None),
    ('gastegger2018', 'shift', 1, None)
])
def test_radial_paramgen_5(rule, mode, nb_param_pairs, r_lower):
    """Test for ValueError when illegal relation between r_lower, r_upper, r_cutoff and zero.
    """
    elems = ['S', 'Cu']
    myGen = sfpg.SymFuncParamGenerator(elems)
    myGen.r_cutoff = 6.
    with pytest.raises(ValueError):
        myGen.generate_radial_params(rule=rule, mode=mode,
                                     nb_param_pairs=nb_param_pairs, r_lower=r_lower)