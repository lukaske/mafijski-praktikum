from setuptools import setup
from Cython.Build import cythonize

setup(
    name='Matrix Diagonalization Jacoby',
    ext_modules=cythonize("jacobi_lib.pyx"),
)

#python setup.py build_ext --inplace