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


@pytest.mark.parametrize("symfunc_type,target_combinations", [
    ('radial',
     [('H', 'H'), ('H', 'C'), ('H', 'O'),
      ('C', 'H'), ('C', 'C'), ('C', 'O'),
      ('O', 'H'), ('O', 'C'), ('O', 'O')]),
    ('angular_narrow',
     [('H', 'H', 'H'), ('H', 'H', 'C'), ('H', 'H', 'O'), ('H', 'C', 'C'),
      ('H', 'C', 'O'), ('H', 'O', 'O'),
      ('C', 'H', 'H'), ('C', 'H', 'C'), ('C', 'H', 'O'), ('C', 'C', 'C'),
      ('C', 'C', 'O'), ('C', 'O', 'O'),
      ('O', 'H', 'H'), ('O', 'H', 'C'), ('O', 'H', 'O'), ('O', 'C', 'C'),
      ('O', 'C', 'O'), ('O', 'O', 'O')]),
    ('angular_wide',
     [('H', 'H', 'H'), ('H', 'H', 'C'), ('H', 'H', 'O'), ('H', 'C', 'C'),
      ('H', 'C', 'O'), ('H', 'O', 'O'),
      ('C', 'H', 'H'), ('C', 'H', 'C'), ('C', 'H', 'O'), ('C', 'C', 'C'),
      ('C', 'C', 'O'), ('C', 'O', 'O'),
      ('O', 'H', 'H'), ('O', 'H', 'C'), ('O', 'H', 'O'), ('O', 'C', 'C'),
      ('O', 'C', 'O'), ('O', 'O', 'O')]),
    ('weighted_radial',
     [('H',), ('C',), ('O',)]),
    ('weighted_angular',
     [('H',), ('C',), ('O',)])
])
def test_element_combinations(symfunc_type, target_combinations):
    """Test if element combinations are correctly constructed.
    """
    elems = ['H', 'C', 'O']
    myGen = SymFuncParamGenerator(elements=elems, r_cutoff=6.)
    myGen.symfunc_type = symfunc_type
    assert myGen.element_combinations == target_combinations


def test_rcutoff():
    elems = ['S', 'Cu']
    # test for errors when cutoff radius not greater than zero
    with pytest.raises(ValueError):
        myGen = SymFuncParamGenerator(elements=elems, r_cutoff=0)
    with pytest.raises(ValueError):
        myGen = SymFuncParamGenerator(elements=elems, r_cutoff=-5)
    # test if initializing and retrieving r_cutoff works as expected
    myGen = SymFuncParamGenerator(elements=elems, r_cutoff=6)
    assert myGen.r_cutoff == 6
    # test for AttributeError when trying to change r_cutoff afterwards
    with pytest.raises(AttributeError):
        myGen.r_cutoff = 10


def test_setter_symfunc_type(basic_generator):
    """Test if error when trying to set invalid symmetry function type.
    """
    with pytest.raises(ValueError):
        basic_generator.symfunc_type = 'illegal_type'


@pytest.mark.parametrize("symfunc_type,r_shift_grid,eta_grid,zetas", [
    (None, np.array([1,2,3]), np.array([4,5,6]), None),
    ('radial', None, np.array([4,5,6]), None),
    ('radial', np.array([1,2,3]), None, None),
    ('weighted_radial', None, np.array([4, 5, 6]), None),
    ('weighted_radial', np.array([1, 2, 3]), None, None),
    ('angular_narrow', None, np.array([4,5,6]), np.array([1,4,9])),
    ('angular_narrow', np.array([1,2,3]), None, np.array([1,4,9])),
    ('angular_narrow', np.array([1,2,3]), np.array([4,5,6]), None),
    ('angular_wide', None, np.array([4,5,6]), np.array([1,4,9])),
    ('angular_wide', np.array([1,2,3]), None, np.array([1,4,9])),
    ('angular_wide', np.array([1,2,3]), np.array([4,5,6]), None),
    ('weighted_angular', None, np.array([4,5,6]), np.array([1,4,9])),
    ('weighted_angular', np.array([1,2,3]), None, np.array([1,4,9])),
    ('weighted_angular', np.array([1,2,3]), np.array([4,5,6]), None)
])
def test_check_writing_prerequisites(basic_generator,
                                     symfunc_type, r_shift_grid, eta_grid, zetas):
    """Test if error when writing parameters or settings while data are missing.
    """
    if symfunc_type is not None:
        basic_generator.symfunc_type = symfunc_type
    if r_shift_grid is not None:
        basic_generator._r_shift_grid = r_shift_grid
    if eta_grid is not None:
        basic_generator._eta_grid = eta_grid
    if zetas is not None:
        basic_generator.zetas = zetas

    # test the completeness checker on its own (without argument)
    with pytest.raises(ValueError):
        basic_generator.check_writing_prerequisites()
    # test the methods that call the completeness checker (which call it with argument)
    with pytest.raises(ValueError):
        basic_generator.write_settings_overview()
    with pytest.raises(ValueError):
        basic_generator.write_parameter_strings()


def test_set_custom_radial_params(basic_generator):
    """Test if set_custom_radial_params correctly raises exceptions and sets values.
    """
    # test for exception when unequal length
    with pytest.raises(TypeError):
        basic_generator.set_custom_radial_params([1.1, 2.2, 3.3], [4.4, 5.5])
    # test for exception when negative value in values for r_shift
    with pytest.raises(ValueError):
        basic_generator.set_custom_radial_params([-0.1, 2.2, 3.3], [1.1, 2.2, 3.3])
    # test for exception when non-positive value in values for eta
    with pytest.raises(ValueError):
        basic_generator.set_custom_radial_params([1.1, 2.2, 3.3], [0, 2.2, 3.3])

    # test if setting custom r_shift and eta values works correctly
    basic_generator.set_custom_radial_params([1.1, 2.2, 3.3], [3.3, 2.2, 1.1])
    assert np.array_equal(basic_generator.r_shift_grid, np.array([1.1, 2.2, 3.3]))
    assert np.array_equal(basic_generator.eta_grid, np.array([3.3, 2.2, 1.1]))

    # test if the dict containing radial parameter generation settings is
    # correctly reset to None, when setting custom radial parameters
    basic_generator.generate_radial_params(rule='imbalzano2018', mode='center', nb_param_pairs=3)
    basic_generator.set_custom_radial_params([1.1, 2.2, 3.3], [3.3, 2.2, 1.1])
    assert basic_generator.radial_paramgen_settings is None















