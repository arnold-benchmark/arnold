from distutils.core import setup
from Cython.Build import cythonize

setup(name="compute_points", ext_modules=cythonize('compute_points.pyx'),)