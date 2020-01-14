import pytest
from pynnp import Log

d = True

@pytest.fixture
def l1():
    l = Log()
    return l

class Test___cinit__:
    def test_skeleton_initialization(self):
        l = Log(True)
        assert isinstance(l, Log), (
               "Can not create skeleton Log instance.")
    def test_empty_initialization(self):
        l = Log()
        assert isinstance(l, Log), (
               "Can not create empty Log instance.")

class Test_addLogEntry:
    def test_add_string_entry(self):
        l = Log()
        l.addLogEntry("Testing...")
    def test_add_nonstring_entry(self):
        l = Log()
        with pytest.raises(TypeError):
            l.addLogEntry(123)

class Test_getLog:
    def test_read_two_entries(self):
        l = Log()
        l.addLogEntry("First entry.")
        l.addLogEntry("Second entry.")
        log = l.getLog()
        assert log[0] == "First entry.", "Incorrect log entry returned."
        assert log[1] == "Second entry.", "Incorrect log entry returned."

class Test_writeToStdout:
    def test_correct_type(self, l1):
        assert isinstance(l1.writeToStdout, bool), "Wrong attribute type."
    def test_set_and_get(self, l1):
        l1.writeToStdout = not d
        l1.writeToStdout = d
        assert l1.writeToStdout == d, "Wrong attribute setter or getter."
