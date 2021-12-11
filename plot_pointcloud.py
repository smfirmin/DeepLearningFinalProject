# from __future__ import print_function
import os
import os.path
import numpy as np
from plyfile import PlyData
import argparse

import matplotlib.pyplot as plt

import numpy as np
import open3d as o3d


def main():
    cloud = o3d.io.read_point_cloud("./ModelNet40/airplane/test/airplane_0627.ply") # Read the point cloud
    o3d.visualization.draw_geometries([cloud]) # Visualize the point cloud     

if __name__ == "__main__":
    main()
