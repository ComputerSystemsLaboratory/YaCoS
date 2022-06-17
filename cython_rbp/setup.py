from setuptools import setup
from Cython.Build import cythonize
import numpy
setup(
    name="cythonrbp",
    version="1.0.0",
    description="RBP/LBPEQ",
    ext_modules = cythonize("src/cython_rbp.pyx"),
    include_dirs=[numpy.get_include()]
)
