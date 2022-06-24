from setuptools import setup
from Cython.Build import cythonize
import numpy
setup(
    name="cythonlbpeq",
    version="1.0.0",
    description="RBP/LBPEQ",
    ext_modules = cythonize("src/cython_lbpeq.pyx"),
    include_dirs=[numpy.get_include()]
)
