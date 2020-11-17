import pytest
from pynnp import Neighbor, Vec3D

a = 13
b = "b"
c = 13.25323

@pytest.fixture
def n1():
    global d1x, d1y, d1z
    d1x, d1y, d1z = 1.0, 2.0, 3.0
    n = Neighbor()
    n.dr = Vec3D(d1x, d1y, d1z)
    n.d = n.dr.norm()
    n.element = 0
    return n

@pytest.fixture
def n2():
    global d2x, d2y, d2z
    d2x, d2y, d2z = 4.0, 5.0, 6.0
    n = Neighbor()
    n.dr = Vec3D(d2x, d2y, d2z)
    n.d = n.dr.norm()
    n.element = 1
    return n

class Test___cinit__:
    def test_skeleton_initialization(self):
        n = Neighbor(True)
        assert isinstance(n, Neighbor), (
               "Can not create skeleton Neighbor instance.")
    def test_empty_initialization(self):
        n = Neighbor()
        assert isinstance(n, Neighbor), (
               "Can not create empty Neighbor instance.")
        assert isinstance(n.dr, Vec3D), "Member Vec3D not initialized."

class Test_index:
    def test_correct_type(self, n1):
        assert isinstance(n1.index, int), "Wrong attribute type."
    def test_set_and_get(self, n1):
        with pytest.raises(TypeError):
            n1.index = b
        n1.index = a
        assert n1.index == a, "Wrong attribute setter or getter."

class Test_tag:
    def test_correct_type(self, n1):
        assert isinstance(n1.tag, int), "Wrong attribute type."
    def test_set_and_get(self, n1):
        with pytest.raises(TypeError):
            n1.tag = b
        n1.tag = a
        assert n1.tag == a, "Wrong attribute setter or getter."

class Test_element:
    def test_correct_type(self, n1):
        assert isinstance(n1.element, int), "Wrong attribute type."
    def test_set_and_get(self, n1):
        with pytest.raises(TypeError):
            n1.element = b
        n1.element = a
        assert n1.element == a, "Wrong attribute setter or getter."

class Test_d:
    def test_correct_type(self, n1):
        assert isinstance(n1.d, float), "Wrong attribute type."
    def test_set_and_get(self, n1):
        with pytest.raises(TypeError):
            n1.d = b
        n1.d = c
        assert n1.d == c, "Wrong attribute setter or getter."

class Test_dr:
    def test_correct_type(self, n1):
        assert isinstance(n1.dr, Vec3D), "Wrong attribute type."
    def test_set_and_get(self, n1):
        with pytest.raises(TypeError):
            n1.dr = b
        n1.dr = Vec3D(1.0, 2.0, 3.0)
        assert n1.dr[0] == 1.0, "Wrong attribute setter or getter."
        assert n1.dr[1] == 2.0, "Wrong attribute setter or getter."
        assert n1.dr[2] == 3.0, "Wrong attribute setter or getter."

class Test_dGdr:
    def test_correct_type(self, n1):
        assert isinstance(n1.dGdr, list), "Wrong attribute type."
    # TODO: Test content.

class Test___eq__:
    def test_not_equal(self, n1, n2):
        assert (n1 == n2) is False, "Different neighbors are equal."
    def test_element_different(self, n1, n2):
        n1.d = n2.d
        assert (n1 == n2) is False, "Different neighbors are equal."
    def test_distance_different(self, n1, n2):
        n1.element = n2.element
        assert (n1 == n2) is False, "Different neighbors are equal."
    def test_equal(self, n1, n2):
        n1.d = n2.d
        n1.element = n2.element
        assert n1 == n2, "Equal neighbors are not equal."

class Test___ne__:
    def test_equal(self, n1, n2):
        assert (n1 != n2) is False, "Different neighbors are not different."
    def test_element_different(self, n1, n2):
        n1.d = n2.d
        assert n1 != n2, "Different neighbors are not different."
    def test_distance_different(self, n1, n2):
        n1.element = n2.element
        assert n1 != n2, "Different neighbors are not different."
    def test_equal(self, n1, n2):
        n1.d = n2.d
        n1.element = n2.element
        assert (n1 != n2) is False, "Equal neighbors are not not different."

class Test___lt__:
    def test_less_than(self, n1, n2):
        assert n1 < n2, "Wrong ordering of neighbors."
    def test_greater_than(self, n1, n2):
        assert (n2 < n1) is False, "Wrong ordering of neighbors."
    def test_equal_distance(self, n1, n2):
        n1.d = n2.d
        assert n1 < n2, "Wrong ordering of neighbors."
    def test_equal_element(self, n1, n2):
        n1.element = n2.element
        assert n1 < n2, "Wrong ordering of neighbors."
    def test_equal(self, n1, n2):
        n1.d = n2.d
        n1.element = n2.element
        assert (n1 < n2) is False, "Wrong ordering of neighbors."

class Test___gt__:
    def test_less_than(self, n1, n2):
        assert (n1 > n2) is False, "Wrong ordering of neighbors."
    def test_greater_than(self, n1, n2):
        assert n2 > n1, "Wrong ordering of neighbors."
    def test_equal_distance(self, n1, n2):
        n1.d = n2.d
        assert n2 > n1, "Wrong ordering of neighbors."
    def test_equal_element(self, n1, n2):
        n1.element = n2.element
        assert n2 > n1, "Wrong ordering of neighbors."
    def test_equal(self, n1, n2):
        n1.d = n2.d
        n1.element = n2.element
        assert (n2 > n1) is False, "Wrong ordering of neighbors."

class Test___le__:
    def test_less_than(self, n1, n2):
        assert n1 <= n2, "Wrong ordering of neighbors."
    def test_greater_than(self, n1, n2):
        assert (n2 <= n1) is False, "Wrong ordering of neighbors."
    def test_equal_distance(self, n1, n2):
        n1.d = n2.d
        assert n1 <= n2, "Wrong ordering of neighbors."
    def test_equal_element(self, n1, n2):
        n1.element = n2.element
        assert n1 <= n2, "Wrong ordering of neighbors."
    def test_equal(self, n1, n2):
        n1.d = n2.d
        n1.element = n2.element
        assert n1 <= n2, "Wrong ordering of neighbors."

class Test___ge__:
    def test_less_than(self, n1, n2):
        assert n2 >= n1, "Wrong ordering of neighbors."
    def test_greater_than(self, n1, n2):
        assert (n1 >= n2) is False, "Wrong ordering of neighbors."
    def test_equal_distance(self, n1, n2):
        n1.d = n2.d
        assert n2 >= n1, "Wrong ordering of neighbors."
    def test_equal_element(self, n1, n2):
        n1.element = n2.element
        assert n2 >= n1, "Wrong ordering of neighbors."
    def test_equal(self, n1, n2):
        n1.d = n2.d
        n1.element = n2.element
        assert n2 >= n1, "Wrong ordering of neighbors."
