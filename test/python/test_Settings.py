import pytest
import os, shutil
from pynnp import Settings

@pytest.fixture
def prepare_dir(request):
    dirname = "workdir"
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    shutil.copytree(request.param, dirname)
    olddir = os.getcwd()
    os.chdir(dirname)
    yield
    os.chdir(olddir)
    shutil.rmtree(dirname)

prepare_default_dir = pytest.mark.parametrize('prepare_dir',
                                              ["setups/Settings"],
                                              indirect=True)

@pytest.fixture
def s1():
    s = Settings()
    s.loadFile("input.nn")
    return s

class Test___cinit__:
    def test_skeleton_initialization(self):
        s = Settings(True)
        assert isinstance(s, Settings), (
               "Can not create skeleton Settings instance.")
    def test_empty_initialization(self):
        s = Settings()
        assert isinstance(s, Settings), (
               "Can not create empty Settings instance.")

@prepare_default_dir
class Test_loadFile:
    def test_file_read(self, prepare_dir):
        s = Settings()
        assert len(s.getSettingsLines()) == 0, "Settings not empty."
        s.loadFile("input.nn")
        assert len(s.getSettingsLines()) > 0, "No lines were read."

@prepare_default_dir
class Test___getitem__:
    def test_existing_key(self, prepare_dir, s1):
        assert isinstance(s1["number_of_elements"], str), (
               "Did not get a value.")
        assert int(s1["number_of_elements"]) == 2, (
               "Incorrect result returned.")
    def test_nonexisting_key(self, prepare_dir, s1):
        with pytest.raises(RuntimeError):
            s1["asdf"]
    def test_nonstring_key(self, prepare_dir, s1):
        with pytest.raises(TypeError):
            s1[3]

@prepare_default_dir
class Test_keywordExists:
    def test_existing_key(self, prepare_dir, s1):
        assert s1.keywordExists("number_of_elements") is True, (
               "Keyword exists but return value is False.")
    def test_nonexisting_key(self, prepare_dir, s1):
        with pytest.raises(RuntimeError):
            s1.keywordExists("asdf")
    def test_nonstring_key(self, prepare_dir, s1):
        with pytest.raises(TypeError):
            s1.keywordExists(3)

@prepare_default_dir
class Test_getValue:
    def test_existing_key(self, prepare_dir, s1):
        assert isinstance(s1.getValue("number_of_elements"), str), (
               "Did not get a value.")
        assert int(s1.getValue("number_of_elements")) == 2, (
               "Incorrect result returned.")
    def test_nonexisting_key(self, prepare_dir, s1):
        with pytest.raises(RuntimeError):
            s1.getValue("asdf")
    def test_nonstring_key(self, prepare_dir, s1):
        with pytest.raises(TypeError):
            s1.getValue(3)

@prepare_default_dir
class Test_info():
    def test_return_list(self, prepare_dir, s1):
        info = s1.info()
        assert isinstance(info, list), "Wrong return type."
        assert all(isinstance(i, str) for i in info), (
               "Wrong type of list elements.")

@prepare_default_dir
class Test_getSettingsLines():
    def test_return_list(self, prepare_dir, s1):
        lines = s1.getSettingsLines()
        assert isinstance(lines, list), "Wrong return type."
        assert all(isinstance(i, str) for i in lines), (
               "Wrong type of list elements.")
        f = open("input.nn")
        for i, fline in enumerate(f):
            assert fline.rstrip("\r\n") == lines[i], "Different line content."
