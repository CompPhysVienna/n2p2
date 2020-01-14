import pytest
from pynnp import ElementMap

@pytest.fixture
def em1():
    global elements
    elements = ["H", "O", "S", "Cu"]
    em = ElementMap()
    em.registerElements(elements[3] + " " + elements[1] + " " +
                        elements[2] + " " + elements[0])
    return em

class Test___cinit__:
    def test_skeleton_initialization(self):
        em = ElementMap(True)
        assert isinstance(em, ElementMap), (
               "Can not create skeleton ElementMap instance.")
    def test_empty_initialization(self):
        em = ElementMap()
        assert isinstance(em, ElementMap), (
               "Can not create empty ElementMap instance.")

class Test_registerElements:
    def test_nonstring_argument(self):
        em = ElementMap()
        with pytest.raises(TypeError):
            em.registerElements(123)
    def test_correct_elements_registered(self):
        em = ElementMap()
        em.registerElements("Cu O S H")
        assert em[0] == "H", "Wrong element registration."
        assert em[1] == "O", "Wrong element registration."
        assert em[2] == "S", "Wrong element registration."
        assert em[3] == "Cu", "Wrong element registration."

class Test___getitem__:
    def test_correct_element_index(self, em1):
        for i, e in enumerate(elements):
            assert em1[e] == i, "Wrong element index returned."
    def test_correct_element_symbol(self, em1):
        for i, e in enumerate(elements):
            assert em1[i] == e, "Wrong element symbol returned."
    def test_out_of_bounds_index(self, em1):
        with pytest.raises(IndexError):
            em1[4]
    def test_unknown_symbol(self, em1):
        with pytest.raises(ArithmeticError):
            em1["XY"]
    def test_wrong_argument_type(self, em1):
        with pytest.raises(NotImplementedError):
            em1[1.23]

class Test_size:
    def test_correct_empty_size_returned(self):
        em = ElementMap()
        assert em.size() == 0, "Wrong size returned."
    def test_correct_size_returned(self, em1):
        assert em1.size() == 4, "Wrong size returned."

class Test_index:
    def test_correct_element_index(self, em1):
        for i, e in enumerate(elements):
            assert em1.index(e) == i, "Wrong element order."
    def test_wrong_argument_type(self, em1):
        with pytest.raises(TypeError):
            em1.index(1)

class Test_symbol:
    def test_correct_element_index(self, em1):
        for i, e in enumerate(elements):
            assert em1.symbol(i) == e, "Wrong element order."
    def test_wrong_argument_type(self, em1):
        with pytest.raises(TypeError):
            em1.symbol("H")

class Test_deregisterElements:
    def test_empty_map(self, em1):
        em1.deregisterElements()
        assert em1.size() == 0, "ElementMap not empty."

class Test_symbolFromAtomicNumber:
    def test_some_elements(self, em1):
        assert em1.symbolFromAtomicNumber(1) == "H", "Wrong element symbol."
        assert em1.symbolFromAtomicNumber(2) == "He", "Wrong element symbol."
        assert em1.symbolFromAtomicNumber(8) == "O", "Wrong element symbol."
        assert em1.symbolFromAtomicNumber(16) == "S", "Wrong element symbol."
        assert em1.symbolFromAtomicNumber(29) == "Cu", "Wrong element symbol."
        with pytest.raises(RuntimeError):
            em1.symbolFromAtomicNumber(0)
            em1.symbolFromAtomicNumber(300)

class Test_info():
    def test_return_list(self, em1):
        info = em1.info()
        assert isinstance(info, list), "Wrong return type."
        assert all(isinstance(i, str) for i in info), (
               "Wrong type of list elements.")
