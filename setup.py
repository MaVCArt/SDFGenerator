from setuptools import setup, Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "sdf_generator",
        ["sdf_generator.pyx"],
        # -- this argument is necessary to support open multi threading
        extra_compile_args=['-openmp'],
    )
]

setup(
    ext_modules=cythonize(ext_modules, annotate=True),
)
