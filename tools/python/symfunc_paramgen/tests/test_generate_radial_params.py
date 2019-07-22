import sys
sys.path.append('../src')

import numpy as np
import pytest
import os
from sfparamgen import SymFuncParamGenerator


@pytest.fixture
def basic_generator():
    elems = ['S', 'Cu']
    return SymFuncParamGenerator(elements=elems, r_cutoff=6.)


@pytest.mark.parametrize("rule,mode,nb_param_pairs,r_lower", [
    ('bad_argument', 'center', 5, 1.5),
    ('bad_argument', 'shift', 5, 1.5),
    ('gastegger2018', 'bad_argument', 5, 1.5),
    ('imbalzano2018', 'bad_argument', 5, None)
])
def test_errors_rule_mode(basic_generator,
                          rule, mode, nb_param_pairs, r_lower):
    """Test for ValueError when invalid arguments for rule or method.
    """
    with pytest.raises(ValueError):
        basic_generator.generate_radial_params(rule=rule, mode=mode,
                                               nb_param_pairs=nb_param_pairs, r_lower=r_lower)


@pytest.mark.parametrize("rule,mode,nb_param_pairs", [
    ('gastegger2018', 'center', 5),
    ('gastegger2018', 'shift', 5)
])
def test_errors_rlower(basic_generator,
                       rule, mode, nb_param_pairs):
    """Test for TypeError when omitting r_lower argument in rule 'gastegger2018'
    """
    with pytest.raises(TypeError):
        basic_generator.generate_radial_params(rule=rule, mode=mode,
                                     nb_param_pairs=nb_param_pairs)


@pytest.mark.parametrize("rule,mode,nb_param_pairs,r_lower", [
    ('gastegger2018', 'center', 1, 1.5),
    ('gastegger2018', 'shift', 1, 1.5),
    ('imbalzano2018', 'center', 1, None),
    ('imbalzano2018', 'shift', 1, None)
])
def test_errors_nb_param_pairs(basic_generator,
                               rule, mode, nb_param_pairs, r_lower):
    """Test for ValueError when nb_param_pairs less than two.
    """
    with pytest.raises(ValueError):
        basic_generator.generate_radial_params(rule=rule, mode=mode,
                                               nb_param_pairs=nb_param_pairs, r_lower=r_lower)


@pytest.mark.parametrize("rule,mode,nb_param_pairs,r_lower,r_upper", [
    ('gastegger2018', 'center', 2, 0., 5.),
    ('gastegger2018', 'center', 2, 0., None),
    ('gastegger2018', 'center', 2, 5., 1.),
    ('gastegger2018', 'center', 2, 7., None),
    ('gastegger2018', 'center', 2, 1., 7.),
    ('gastegger2018', 'shift', 2, -1., 5.),
    ('gastegger2018', 'shift', 2, -1., None),
    ('gastegger2018', 'shift', 2, 5., 1.),
    ('gastegger2018', 'shift', 2, 7, None),
    ('gastegger2018', 'shift', 2, 1., 7.)
])
def test_errors_numerical_order(basic_generator,
                                rule, mode, nb_param_pairs, r_lower, r_upper):
    """Test for ValueError when illegal relation between r_lower, r_upper, r_cutoff in rule 'gastegger2018'
    """
    with pytest.raises(ValueError):
        basic_generator.generate_radial_params(rule=rule, mode=mode,
                                     nb_param_pairs=nb_param_pairs,
                                     r_lower=r_lower, r_upper=r_upper)

@pytest.mark.parametrize("rule,mode,nb_param_pairs,r_lower,r_upper,target_r_shift,target_eta", [
    ('gastegger2018', 'center', 3, 1., 5.,
     [0, 0, 0],
     [1/2, 1/18, 1/50]),
    ('gastegger2018', 'shift', 3, 1., 5.,
     [1, 3, 5],
     [1/8, 1/8, 1/8]),
    ('gastegger2018', 'center', 3, 1., None,
     [0, 0, 0],
     [1/2, 1/(2*3.5**2), 1/(2*6**2)]),
    ('gastegger2018', 'shift', 3, 1., None,
     [1, 3.5, 6],
     [1/(2*2.5**2)]*3)
])
def test_parameter_generation_gastegger2018(basic_generator,
                                            rule, mode, nb_param_pairs, r_lower, r_upper,
                                            target_r_shift, target_eta):
    """Test if generated r_shift and eta values match target values.
    """
    # when passing r_upper
    if r_upper is not None:
        basic_generator.generate_radial_params(rule=rule, mode=mode,
                                     nb_param_pairs=nb_param_pairs,
                                     r_lower=r_lower, r_upper=r_upper)
    # when not passing r_upper, using default value of r_cutoff
    else:
        basic_generator.generate_radial_params(rule=rule, mode=mode,
                                     nb_param_pairs=nb_param_pairs,
                                     r_lower=r_lower)

    # test if generated parameter sets match target values
    assert np.allclose(basic_generator.r_shift_grid, np.array(target_r_shift))
    assert np.allclose(basic_generator.eta_grid, np.array(target_eta))

    # test if settings have been correctly stored
    assert basic_generator.radial_paramgen_settings['rule'] == rule
    assert basic_generator.radial_paramgen_settings['mode'] == mode
    assert basic_generator.radial_paramgen_settings['nb_param_pairs'] == nb_param_pairs
    assert basic_generator.radial_paramgen_settings['r_lower'] == r_lower
    if r_upper is not None:
        assert basic_generator.radial_paramgen_settings['r_upper'] == r_upper
    else:
        assert basic_generator.radial_paramgen_settings['r_upper'] == basic_generator.r_cutoff


@pytest.mark.parametrize("rule,mode,nb_param_pairs,target_r_shift,target_eta", [
    ('imbalzano2018', 'center', 4,
     [0, 0, 0, 0],
     [1/36, 0.0577801, 0.1201874, 1/4]),
    ('imbalzano2018', 'shift', 4,
     [1.5, 2.1213203, 3.0, 4.2426407],
     [2.5904121, 1.2952060, 0.6476030, 0.3238015])
])
def test_parameter_generation_imbalzano2018(basic_generator,
                                            rule, mode, nb_param_pairs,
                                            target_r_shift, target_eta):
    """Test if generated r_shift and eta values match target values.
    """
    basic_generator.generate_radial_params(rule=rule, mode=mode,
                                 nb_param_pairs=nb_param_pairs)

    # test if generated parameter sets match target values
    assert np.allclose(basic_generator.r_shift_grid, np.array(target_r_shift))
    assert np.allclose(basic_generator.eta_grid, np.array(target_eta))

    # test if settings have been correctly stored
    assert basic_generator.radial_paramgen_settings['rule'] == rule
    assert basic_generator.radial_paramgen_settings['mode'] == mode
    assert basic_generator.radial_paramgen_settings['nb_param_pairs'] == nb_param_pairs


@pytest.mark.parametrize("mode", ['center', 'shift'])
def test_warning_unused_args(basic_generator, mode):
    """Test if warning when passing r_lower or r_upper while using rule 'imbalzano2018'
    """
    with pytest.warns(UserWarning):
        basic_generator.generate_radial_params(rule='imbalzano2018', mode=mode,
                                     nb_param_pairs=3,
                                     r_lower=1)
    with pytest.warns(UserWarning):
        basic_generator.generate_radial_params(rule='imbalzano2018', mode=mode,
                                     nb_param_pairs=3,
                                     r_upper=5)
    with pytest.warns(UserWarning):
        basic_generator.generate_radial_params(rule='imbalzano2018', mode=mode,
                                     nb_param_pairs=3,
                                     r_lower=1, r_upper=5)










