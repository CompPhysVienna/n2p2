import pytest
from pynnp import CutoffFunction

cutoff_radius = 3.0
cutoff_parameter = 0.2
cutoff_function_types = [CutoffFunction.CT_HARD,
                         CutoffFunction.CT_COS,
                         CutoffFunction.CT_TANHU,
                         CutoffFunction.CT_TANH,
                         CutoffFunction.CT_EXP,
                         CutoffFunction.CT_POLY1,
                         CutoffFunction.CT_POLY2,
                         CutoffFunction.CT_POLY3,
                         CutoffFunction.CT_POLY4]

test_radius = 0.8 * cutoff_radius

cutoff_function_values = [1.0,
                          0.1464466094067261,
                          0.007689153712069772,
                          0.01740635089776904,
                          0.276453046629564,
                          0.1562499999999999,
                          0.103515625,
                          0.07055664062499922,
                          0.04892730712889859]

cutoff_function_dvalues = [0.0,
                           -0.46280030605816297,
                           -0.0374393678577051,
                           -0.08475351107871792,
                           -0.9027038257291897,
                           -0.4687499999999998,
                           -0.43945312499999933,
                           -0.38452148437500017,
                           -0.3244400024414215]

@pytest.fixture
def c1():
    c = CutoffFunction()
    c.setCutoffRadius(cutoff_radius)
    c.setCutoffParameter(cutoff_parameter)
    return c

class Test___cinit__:
    def test_skeleton_initialization(self):
        l = CutoffFunction(True)
        assert isinstance(l, CutoffFunction), (
               "Can not create skeleton CutoffFunction instance.")
    def test_empty_initialization(self):
        l = CutoffFunction()
        assert isinstance(l, CutoffFunction), (
               "Can not create empty CutoffFunction instance.")

class Test_setCutoffType:
    @pytest.mark.parametrize("cf_type", cutoff_function_types)
    def test_set_correct_type(self, c1, cf_type):
        c1.setCutoffType(cf_type)
    def test_set_unknown_type(self, c1):
        with pytest.raises(ValueError):
            c1.setCutoffType(len(cutoff_function_types))

class Test_setCutoffRadius:
    def test_set_float_value(self, c1):
        c1.setCutoffRadius(12.0)
    def test_set_wrong_argument(self, c1):
        with pytest.raises(TypeError):
            c1.setCutoffRadius("a")

class Test_setCutoffParameter:
    def test_set_float_value(self, c1):
        c1.setCutoffParameter(0.3)
    def test_set_wrong_float_value(self, c1):
        with pytest.raises(ValueError):
            c1.setCutoffParameter(-0.000001)
        with pytest.raises(ValueError):
            c1.setCutoffParameter(1.0)
        with pytest.raises(ValueError):
            c1.setCutoffParameter(1.000001)
    def test_set_wrong_argument(self, c1):
        with pytest.raises(TypeError):
            c1.setCutoffParameter("a")

class Test_f:
    @pytest.mark.parametrize("cf_type", cutoff_function_types)
    def test_boundaries(self, c1, cf_type):
        c1.setCutoffType(cf_type)
        if cf_type not in [CutoffFunction.CT_TANHU]:
            assert c1.f(0.0) == pytest.approx(1.0), (
                   "Wrong cutoff function value.")
        if cf_type not in [CutoffFunction.CT_TANHU, CutoffFunction.CT_TANH]:
            assert c1.f(cutoff_radius * cutoff_parameter) == pytest.approx(1.0), (
                   "Wrong cutoff function value.")
        assert c1.f(cutoff_radius) == pytest.approx(0.0), (
               "Wrong cutoff function value.")
    @pytest.mark.parametrize("cf_type, cf_value",
                             zip(cutoff_function_types, cutoff_function_values))
    def test_nonzero_radius(self, c1, cf_type, cf_value):
        c1.setCutoffType(cf_type)
        assert c1.f(test_radius) == pytest.approx(cf_value), (
               "Wrong cutoff function value.")
    def test_nonfloat_argument(self, c1):
        with pytest.raises(TypeError):
            c1.f("a")

class Test_df:
    @pytest.mark.parametrize("cf_type", cutoff_function_types)
    def test_boundaries(self, c1, cf_type):
        c1.setCutoffType(cf_type)
        if cf_type not in [CutoffFunction.CT_TANHU, CutoffFunction.CT_TANH]:
            assert c1.df(0.0) == pytest.approx(0.0), (
                   "Wrong cutoff derivative value.")
        assert c1.df(cutoff_radius) == pytest.approx(0.0), (
               "Wrong cutoff derivative value.")
    @pytest.mark.parametrize("cf_type, cf_dvalue",
                             zip(cutoff_function_types,
                                 cutoff_function_dvalues))
    def test_nonzero_radius(self, c1, cf_type, cf_dvalue):
        c1.setCutoffType(cf_type)
        assert c1.df(test_radius) == pytest.approx(cf_dvalue), (
               "Wrong cutoff derivative value.")
    def test_nonfloat_argument(self, c1):
        with pytest.raises(TypeError):
            c1.df("a")

class Test_fdf:
    @pytest.mark.parametrize("cf_type", cutoff_function_types)
    def test_boundaries(self, c1, cf_type):
        c1.setCutoffType(cf_type)

        f, df = c1.fdf(0.0)
        if cf_type not in [CutoffFunction.CT_TANHU]:
            assert f == pytest.approx(1.0), (
                   "Wrong cutoff function value.")
        if cf_type not in [CutoffFunction.CT_TANHU, CutoffFunction.CT_TANH]:
            assert df == pytest.approx(0.0), (
                   "Wrong cutoff derivative value.")

        f, df = c1.fdf(cutoff_radius * cutoff_parameter)
        if cf_type not in [CutoffFunction.CT_TANHU, CutoffFunction.CT_TANH]:
            assert f == pytest.approx(1.0), (
                   "Wrong cutoff function value.")

        f, df = c1.fdf(cutoff_radius)
        assert f == pytest.approx(0.0), (
               "Wrong cutoff function value.")
        assert df == pytest.approx(0.0), (
               "Wrong cutoff derivative value.")
    @pytest.mark.parametrize("cf_type, cf_value, cf_dvalue",
                             zip(cutoff_function_types,
                                 cutoff_function_values,
                                 cutoff_function_dvalues))
    def test_nonzero_radius(self, c1, cf_type, cf_value, cf_dvalue):
        c1.setCutoffType(cf_type)
        f, df = c1.fdf(test_radius)
        assert f == pytest.approx(cf_value), (
               "Wrong cutoff function value.")
        assert df == pytest.approx(cf_dvalue), (
               "Wrong cutoff derivative value.")
    def test_nonfloat_argument(self, c1):
        with pytest.raises(TypeError):
            c1.fdf("a")
