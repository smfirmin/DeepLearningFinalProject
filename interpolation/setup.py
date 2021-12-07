from __future__ import division, absolute_import, with_statement, print_function
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob
import shutil

import pointnet2

requirements = ["hydra-core==0.11.3", "pytorch-lightning==0.7.1", "h5py", "enum34", "future"]

setup(
    name="pointnet2",
    version=1.0,
    author="Erik Wijmans",
    packages=find_packages(),
    install_requires=requirements,
)