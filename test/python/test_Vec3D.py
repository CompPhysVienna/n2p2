import pytest
from pynnp import Vec3D

a = 2.5

@pytest.fixture
def v1():
    global v1x, v1y, v1z
    v1x, v1y, v1z = 1.0, 2.0, 3.0
    v = Vec3D(v1x, v1y, v1z)
    return v

@pytest.fixture
def v2():
    global v2x, v2y, v2z
    v2x, v2y, v2z = -5.0, -2.0, 3.0
    v = Vec3D(v2x, v2y, v2z)
    return v

class Test___cinit__:
    def test_skeleton_initialization(self):
        v = Vec3D(True)
        assert isinstance(v, Vec3D), "Can not create skeleton Vec3D instance."
    def test_empty_initialization(self):
        v = Vec3D()
        assert isinstance(v, Vec3D), "Can not create empty Vec3D instance."
        assert v.r == [0.0, 0.0, 0.0], ("Empty Vec3D is not initialized "
               "correctly.")
    def test_float_initialization(self):
        v = Vec3D(1.0, 2.0, 3.0)
        assert isinstance(v, Vec3D), "Can not create Vec3D instance."
        assert v.r == [1.0, 2.0, 3.0], "Vec3D is not initialized correctly."
        with pytest.raises(ValueError):
            Vec3D(1.0)
        with pytest.raises(ValueError):
            Vec3D(1.0, 2.0)
        with pytest.raises(TypeError):
            Vec3D("a", "b", "c")
    def test_Vec3D_initialization(self):
        v = Vec3D(1.0, 2.0, 3.0)
        w = Vec3D(v)
        assert isinstance(w, Vec3D), "Can not create Vec3D instance."
        assert w.r == [1.0, 2.0, 3.0], "Vec3D is not initialized correctly."
        assert id(v) != id(w), "Copy constructor did not return new object."

class Test___getitem__():
    def test_no_integer_index(self, v1):
        with pytest.raises(TypeError):
            v1["a"]
    def test_negative_index(self, v1):
        with pytest.raises(IndexError):
            v1[-10]
    def test_large_index(self, v1):
        with pytest.raises(IndexError):
            v1[3]
    def test_components(self, v1):
        assert v1[0] == v1x, "Wrong x component."
        assert v1[1] == v1y, "Wrong y component."
        assert v1[2] == v1z, "Wrong z component."

class Test___setitem__():
    def test_no_integer_index(self, v1):
        with pytest.raises(TypeError):
            v1["a"] = 1.0
    def test_negative_index(self, v1):
        with pytest.raises(IndexError):
            v1[-10] = 1.0
    def test_large_index(self, v1):
        with pytest.raises(IndexError):
            v1[3] = 1.0
    def test_wrong_type(self, v1):
        with pytest.raises(TypeError):
            v1[0] = "a"
    def test_components(self, v1):
        v1[0] = -v1x
        assert v1[0] == -v1x, "Wrong x component."
        v1[1] = -v1y
        assert v1[1] == -v1y, "Wrong x component."
        v1[2] = -v1z
        assert v1[2] == -v1z, "Wrong x component."

class Test___str__():
    def test_return_string(self, v1):
        assert isinstance(v1.__str__(), str), "No string returned."
    def test_string_content(self, v1):
        assert v1.__str__() == "x: {0:f} y: {1:f} z: {2:f}".format(v1x,
                                                                   v1y,
                                                                   v1z), (
               "Wrong string format.")

class Test___iadd__():
    def test_wrong_type(self, v1):
        with pytest.raises(TypeError):
            v1 += "a"
    def test_correct_addition(self, v1, v2):
        v1 += v2
        assert v1[0] == pytest.approx(v1x + v2x), "Wrong x component."
        assert v1[1] == pytest.approx(v1y + v2y), "Wrong y component."
        assert v1[2] == pytest.approx(v1z + v2z), "Wrong z component."
    def test_added_Vec3D_unchanged(self, v1, v2):
        v1 += v2
        assert v2[0] == v2x, "Wrong x component."
        assert v2[1] == v2y, "Wrong y component."
        assert v2[2] == v2z, "Wrong z component."

class Test___isub__():
    def test_wrong_type(self, v1):
        with pytest.raises(TypeError):
            v1 -= "a"
    def test_correct_subtraction(self, v1, v2):
        v1 -= v2
        assert v1[0] == pytest.approx(v1x - v2x), "Wrong x component."
        assert v1[1] == pytest.approx(v1y - v2y), "Wrong y component."
        assert v1[2] == pytest.approx(v1z - v2z), "Wrong z component."
    def test_subtracted_Vec3D_unchanged(self, v1, v2):
        v1 -= v2
        assert v2[0] == v2x, "Wrong x component."
        assert v2[1] == v2y, "Wrong y component."
        assert v2[2] == v2z, "Wrong z component."

class Test___imul__():
    def test_wrong_type(self, v1):
        with pytest.raises(TypeError):
            v1 *= "a"
    def test_correct_float_multiplication(self, v1):
        v1 *= a
        assert v1[0] == pytest.approx(a * v1x), "Wrong x component."
        assert v1[1] == pytest.approx(a * v1y), "Wrong y component."
        assert v1[2] == pytest.approx(a * v1z), "Wrong z component."

class Test___itruediv__():
    def test_wrong_type(self, v1):
        with pytest.raises(TypeError):
            v1 /= "a"
    def test_correct_float_division(self, v1):
        v1 /= a
        assert v1[0] == pytest.approx(v1x / a), "Wrong x component."
        assert v1[1] == pytest.approx(v1y / a), "Wrong y component."
        assert v1[2] == pytest.approx(v1z / a), "Wrong z component."

class Test___mul__():
    def test_wrong_type(self, v1):
        with pytest.raises(TypeError):
            res = v1 * "a"
        with pytest.raises(TypeError):
            res = "a" * v1
    def test_scalar_product(self, v1, v2):
        sp = v1 * v2
        assert isinstance(sp, float), "Wrong type returned."
        assert sp == pytest.approx(v1x * v2x + v1y * v2y + v1z * v2z), (
               "Wrong scalar product.")
    def test_right_float_multiplication(self, v1):
        res = v1 * a
        assert isinstance(res, Vec3D), "Wrong type returned."
        assert res[0] == pytest.approx(v1x * a), "Wrong x component."
        assert res[1] == pytest.approx(v1y * a), "Wrong y component."
        assert res[2] == pytest.approx(v1z * a), "Wrong z component."
    def test_left_float_multiplication(self, v1):
        res = a * v1
        assert isinstance(res, Vec3D), "Wrong type returned."
        assert res[0] == pytest.approx(v1x * a), "Wrong x component."
        assert res[1] == pytest.approx(v1y * a), "Wrong y component."
        assert res[2] == pytest.approx(v1z * a), "Wrong z component."

class Test___eq__():
    def test_wrong_type(self, v1):
        with pytest.raises(TypeError):
            v1 == "a"
    def test_equal_components(self, v1):
        ve = Vec3D(v1x, v1y, v1z)
        assert id(v1) != id(ve), "Test Vec3Ds are identical."
        assert (v1 == ve) == (
               v1[0] == ve[0] and v1[1] == ve[1] and v1[2] == ve[2]), (
               "Wrong comparison.")

class Test___ne__():
    def test_wrong_type(self, v1):
        with pytest.raises(TypeError):
            v1 != "a"
    def test_one_different_component(self, v1):
        ve = Vec3D(v1x + 1.0, v1y, v1z)
        assert (v1 != ve) == (
               v1[0] != ve[0] or v1[1] != ve[1] or v1[2] != ve[2]), (
               "Wrong comparison (x).")
        ve = Vec3D(v1x, v1y + 1.0, v1z)
        assert (v1 != ve) == (
               v1[0] != ve[0] or v1[1] != ve[1] or v1[2] != ve[2]), (
               "Wrong comparison (y).")
        ve = Vec3D(v1x, v1y, v1z + 1.0)
        assert (v1 != ve) == (
               v1[0] != ve[0] or v1[1] != ve[1] or v1[2] != ve[2]), (
               "Wrong comparison (z).")

class Test_norm():
    def test_correct_result(self, v1):
        from math import sqrt
        assert v1.norm() == pytest.approx(sqrt(v1x * v1x +
                                               v1y * v1y +
                                               v1z * v1z)), "Norm incorrect."

class Test_norm2():
    def test_correct_result(self, v1):
        assert v1.norm2() == pytest.approx(v1x * v1x +
                                           v1y * v1y +
                                           v1z * v1z), "Norm incorrect."

class Test_normalize():
    def test_correct_result(self, v1):
        v1.normalize()
        assert v1.norm() == pytest.approx(1.0), "Norm not equal to 1.0."

class Test_cross():
    def test_wrong_type(self, v1):
        with pytest.raises(TypeError):
            v1.cross("a")
    def test_correct_return_type(self, v1, v2):
        v3 = v1.cross(v2)
        assert isinstance(v3, Vec3D), "Wrong type returned."
    def test_correct_result(self, v1, v2):
        v3 = v1.cross(v2)
        assert v3[0] == pytest.approx(v1y * v2z - v2y * v1z), "Wrong x result."
        assert v3[1] == pytest.approx(v1z * v2x - v2z * v1x), "Wrong y result."
        assert v3[2] == pytest.approx(v1x * v2y - v2x * v1y), "Wrong z result."

class Test___add__():
    def test_wrong_type(self, v1):
        with pytest.raises(TypeError):
            v1 + "a"
        with pytest.raises(TypeError):
            "a" + v1
    def test_correct_return_type(self, v1, v2):
        v3 = v1 + v2
        assert isinstance(v3, Vec3D), "Wrong type returned."
    def test_new_object_returned(self, v1, v2):
        v3 = v1 + v2
        assert id(v1) != id(v3), "No new Vec3D generated."
        assert id(v2) != id(v3), "No new Vec3D generated."
    def test_correct_result(self, v1, v2):
        v3 = v1 + v2
        assert v3[0] == pytest.approx(v1x + v2x), "Wrong x result."
        assert v3[1] == pytest.approx(v1y + v2y), "Wrong y result."
        assert v3[2] == pytest.approx(v1z + v2z), "Wrong z result."

class Test___sub__():
    def test_wrong_type(self, v1):
        with pytest.raises(TypeError):
            v1 - "a"
        with pytest.raises(TypeError):
            "a" - v1
    def test_correct_return_type(self, v1, v2):
        v3 = v1 - v2
        assert isinstance(v3, Vec3D), "Wrong type returned."
    def test_new_object_returned(self, v1, v2):
        v3 = v1 - v2
        assert id(v1) != id(v3), "No new Vec3D generated."
        assert id(v2) != id(v3), "No new Vec3D generated."
    def test_correct_result(self, v1, v2):
        v3 = v1 - v2
        assert v3[0] == pytest.approx(v1x - v2x), "Wrong x result."
        assert v3[1] == pytest.approx(v1y - v2y), "Wrong y result."
        assert v3[2] == pytest.approx(v1z - v2z), "Wrong z result."

class Test___neg__():
    def test_correct_return_type(self, v1):
        assert isinstance(-v1, Vec3D), "Wrong type returned."
    def test_new_object_returned(self, v1):
        assert id(v1) != id(-v1), "No new Vec3D generated."
    def test_correct_result(self, v1):
        v2 = -v1
        assert v2[0] == pytest.approx(-v1x), "Wrong x result."
        assert v2[1] == pytest.approx(-v1y), "Wrong y result."
        assert v2[2] == pytest.approx(-v1z), "Wrong z result."

class Test___truediv__():
    def test_wrong_type(self, v1, v2):
        with pytest.raises(TypeError):
            v1 / "a"
        with pytest.raises(TypeError):
            "a" / v1
        with pytest.raises(TypeError):
            v1 / v2
    def test_correct_return_type(self, v1):
        v2 = v1 / a
        assert isinstance(v2, Vec3D), "Wrong type returned."
    def test_new_object_returned(self, v1):
        v2 = v1 / a
        assert id(v1) != id(v2), "No new Vec3D generated."
    def test_correct_result(self, v1):
        v2 = v1 / a
        assert v2[0] == pytest.approx(v1x / a), "Wrong x result."
        assert v2[1] == pytest.approx(v1y / a), "Wrong y result."
        assert v2[2] == pytest.approx(v1z / a), "Wrong z result."

class Test_r():
    def test_getter_correct_type(self, v1):
        assert isinstance(v1.r, list), "r is not a list."
    def test_getter_correct_size(self, v1):
        assert len(v1.r) == 3, "Wrong size of r."
    def test_getter_correct_content(self, v1):
        assert v1.r == [v1x, v1y, v1z], "Wrong content of r."
    def test_setter_wrong_size(self, v1):
        with pytest.raises(IndexError):
            v1.r = "a"
    def test_setter_wrong_type(self, v1):
        with pytest.raises(TypeError):
            v1.r = ["a", "b", "c"]
    def test_setter_wrong_list_size(self, v1):
        with pytest.raises(IndexError):
            v1.r = [1.0, 2.0]
    def test_setter_correct_assignment(self, v1, v2):
        v1.r = [v2x, v2y, v2z]
        assert v1.r == v2.r, "Content not correctly assigned."
    def test_setter_correct_assignment_to_Vec3D(self, v1, v2):
        v1.r = [v2x, v2y, v2z]
        assert v1 == v2, "Content not correctly assigned."
    # Assignment of individual components does NOT work!
    def test_setter_index_assignment_not_possible(self, v1, v2):
        v1.r[0] = v2x
        v1.r[1] = v2y
        v1.r[2] = v2z
        assert v1.r[0] == v1x, "Assignment did something unexpected."
        assert v1.r[1] == v1y, "Assignment did something unexpected."
        assert v1.r[2] == v1z, "Assignment did something unexpected."
