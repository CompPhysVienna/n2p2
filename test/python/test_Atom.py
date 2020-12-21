import pytest
from pynnp import Atom, Vec3D

a = 13
b = "b"
c = 13.25323
d = True

@pytest.fixture
def a1():
    global a1x, a1y, a1z
    a1x, a1y, a1z = 1.0, 2.0, 3.0
    a = Atom()
    a.r = Vec3D(a1x, a1y, a1z)
    return a

@pytest.fixture
def v1():
    global v1x, v1y, v1z
    v1x, v1y, v1z = 1.0, 2.0, 3.0
    v = Vec3D(v1x, v1y, v1z)
    return v

class Test___cinit__:
    def test_skeleton_initialization(self):
        a = Atom(True)
        assert isinstance(a, Atom), "Can not create skeleton Atom instance."
    def test_empty_initialization(self):
        a = Atom()
        assert isinstance(a, Atom), "Can not create empty Atom instance."
        assert isinstance(a.r, Vec3D), "Member Vec3D not initialized."
        assert isinstance(a.f, Vec3D), "Member Vec3D not initialized."
        assert isinstance(a.fRef, Vec3D), "Member Vec3D not initialized."

class Test_hasNeighborList:
    def test_correct_type(self, a1):
        assert isinstance(a1.hasNeighborList, bool), "Wrong attribute type."
    def test_set_and_get(self, a1):
        a1.hasNeighborList = not d
        a1.hasNeighborList = d
        assert a1.hasNeighborList == d, "Wrong attribute setter or getter."

class Test_hasSymmetryFunctions:
    def test_correct_type(self, a1):
        assert isinstance(a1.hasSymmetryFunctions, bool), (
               "Wrong attribute type.")
    def test_set_and_get(self, a1):
        a1.hasSymmetryFunctions = not d
        a1.hasSymmetryFunctions = d
        assert a1.hasSymmetryFunctions == d, (
               "Wrong attribute setter or getter.")

class Test_hasSymmetryFunctionDerivatives:
    def test_correct_type(self, a1):
        assert isinstance(a1.hasSymmetryFunctionDerivatives, bool), (
               "Wrong attribute type.")
    def test_set_and_get(self, a1):
        a1.hasSymmetryFunctionDerivatives = not d
        a1.hasSymmetryFunctionDerivatives = d
        assert a1.hasSymmetryFunctionDerivatives == d, (
               "Wrong attribute setter or getter.")

class Test_index:
    def test_correct_type(self, a1):
        assert isinstance(a1.index, int), "Wrong attribute type."
    def test_set_and_get(self, a1):
        with pytest.raises(TypeError):
            a1.index = b
        a1.index = a
        assert a1.index == a, "Wrong attribute setter or getter."

class Test_indexStructure:
    def test_correct_type(self, a1):
        assert isinstance(a1.indexStructure, int), "Wrong attribute type."
    def test_set_and_get(self, a1):
        with pytest.raises(TypeError):
            a1.indexStructure = b
        a1.indexStructure = a
        assert a1.indexStructure == a, "Wrong attribute setter or getter."

class Test_tag:
    def test_correct_type(self, a1):
        assert isinstance(a1.tag, int), "Wrong attribute type."
    def test_set_and_get(self, a1):
        with pytest.raises(TypeError):
            a1.tag = b
        a1.tag = a
        assert a1.tag == a, "Wrong attribute setter or getter."

class Test_element:
    def test_correct_type(self, a1):
        assert isinstance(a1.element, int), "Wrong attribute type."
    def test_set_and_get(self, a1):
        with pytest.raises(TypeError):
            a1.element = b
        a1.element = a
        assert a1.element == a, "Wrong attribute setter or getter."

class Test_numNeighbors:
    def test_correct_type(self, a1):
        assert isinstance(a1.numNeighbors, int), "Wrong attribute type."
    def test_set_and_get(self, a1):
        with pytest.raises(TypeError):
            a1.numNeighbors = b
        a1.numNeighbors = a
        assert a1.numNeighbors == a, "Wrong attribute setter or getter."

class Test_numNeighborsUnique:
    def test_correct_type(self, a1):
        assert isinstance(a1.numNeighborsUnique, int), "Wrong attribute type."
    def test_set_and_get(self, a1):
        with pytest.raises(TypeError):
            a1.numNeighborsUnique = b
        a1.numNeighborsUnique = a
        assert a1.numNeighborsUnique == a, "Wrong attribute setter or getter."

class Test_numSymmetryFunctions:
    def test_correct_type(self, a1):
        assert isinstance(a1.numSymmetryFunctions, int), (
               "Wrong attribute type.")
    def test_set_and_get(self, a1):
        with pytest.raises(TypeError):
            a1.numSymmetryFunctions = b
        a1.numSymmetryFunctions = a
        assert a1.numSymmetryFunctions == a, (
               "Wrong attribute setter or getter.")

class Test_energy:
    def test_correct_type(self, a1):
        assert isinstance(a1.energy, float), "Wrong attribute type."
    def test_set_and_get(self, a1):
        with pytest.raises(TypeError):
            a1.energy = b
        a1.energy = c
        assert a1.energy == c, "Wrong attribute setter or getter."

class Test_charge:
    def test_correct_type(self, a1):
        assert isinstance(a1.charge, float), "Wrong attribute type."
    def test_set_and_get(self, a1):
        with pytest.raises(TypeError):
            a1.charge = b
        a1.charge = c
        assert a1.charge == c, "Wrong attribute setter or getter."

class Test_r:
    def test_correct_type(self, a1):
        assert isinstance(a1.r, Vec3D), "Wrong attribute type."
    def test_set_and_get(self, a1, v1):
        with pytest.raises(TypeError):
            a1.r = b
        a1.r = v1
        assert a1.r.r == [v1x, v1y, v1z], (
               "Wrong attribute setter or getter.")

class Test_f:
    def test_correct_type(self, a1):
        assert isinstance(a1.f, Vec3D), "Wrong attribute type."
    def test_set_and_get(self, a1, v1):
        with pytest.raises(TypeError):
            a1.f = b
        a1.f = v1
        assert a1.f.r == [v1x, v1y, v1z], (
               "Wrong attribute setter or getter.")

class Test_fRef:
    def test_correct_type(self, a1):
        assert isinstance(a1.fRef, Vec3D), "Wrong attribute type."
    def test_set_and_get(self, a1, v1):
        with pytest.raises(TypeError):
            a1.fRef = b
        a1.fRef = v1
        assert a1.fRef.r == [v1x, v1y, v1z], (
               "Wrong attribute setter or getter.")

class Test_neighborsUnique:
    def test_correct_type(self, a1):
        assert isinstance(a1.neighborsUnique, list), "Wrong attribute type."
    # TODO: Test content.

class Test_numNeighborsPerElement:
    def test_correct_type(self, a1):
        assert isinstance(a1.numNeighborsPerElement, list), (
               "Wrong attribute type.")
    # TODO: Test content.

class Test_G:
    def test_correct_type(self, a1):
        assert isinstance(a1.G, list), (
               "Wrong attribute type.")
    # TODO: Test content.

class Test_dEdG:
    def test_correct_type(self, a1):
        assert isinstance(a1.dEdG, list), (
               "Wrong attribute type.")
    # TODO: Test content.

#class Test_dGdxia:
#    def test_correct_type(self, a1):
#        assert isinstance(a1.dGdxia, list), (
#               "Wrong attribute type.")
#    # TODO: Test content.

class Test_dGdr:
    def test_correct_type(self, a1):
        assert isinstance(a1.dGdr, list), (
               "Wrong attribute type.")
    # TODO: Test content.

class Test_neighbors:
    def test_correct_type(self, a1):
        assert isinstance(a1.neighbors, list), (
               "Wrong attribute type.")
    # TODO: Test content.

class Test_info():
    def test_return_list(self, a1):
        info = a1.info()
        assert isinstance(info, list), "Wrong return type."
        assert all(isinstance(i, str) for i in info), (
               "Wrong type of list elements.")
