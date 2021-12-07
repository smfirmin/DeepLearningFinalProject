from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob
import shutil

_ext_src_root = "pointnet2_ops_lib/_ext-src/"
_ext_sources = glob.glob("{}/src/*.cpp".format(_ext_src_root)) + glob.glob(
    "{}/src/*.cu".format(_ext_src_root)
)
_ext_headers = glob.glob("{}/include/*".format(_ext_src_root))

requirements = ["etw_pytorch_utils==1.1.1", "h5py", "enum34", "future"]

setup(name='pointnet',
      packages=['pointnet'],
      package_dir={'pointnet': 'pointnet'},
      install_requires=['torch',
                        'tqdm',
                        'plyfile'],
    version='0.0.1')
