"""Build the isolated guard with: python setup.py build_ext --inplace.

This is an experiment, not part of the TA wheel or production launcher.
"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(name="experimental-column-pack-guard", version="0.0.1",
      ext_modules=[CppExtension("native_guard", ["native_guard.cpp"], extra_compile_args=["-O2"])],
      cmdclass={"build_ext": BuildExtension})
