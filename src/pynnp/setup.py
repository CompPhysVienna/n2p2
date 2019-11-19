from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import glob

include_dir = "../libnnp/"
sources = ["pynnp.pyx"]
sources += glob.glob(include_dir + "*.cpp")
compile_options = ["-std=c++11"] #, "-fopenmp"]
link_options = [] #"-fopenmp"]

extension = [Extension("*",
                       sources,
                       extra_compile_args=compile_options,
                       extra_link_args=link_options,
                       include_dirs=[include_dir],
                       language="c++")]

setup(
    ext_modules=cythonize(extension,
                          compiler_directives={'language_level' : "3",
                                               'c_string_type' : 'unicode',
                                               'c_string_encoding' : 'utf8'})
)
