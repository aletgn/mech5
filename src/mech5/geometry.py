import os
import sys
sys.path.append('../../src/')
from pathlib import Path

from typing import Union, List

import numpy as np
import pandas as pd

import yaml
from tqdm import tqdm

# import pyvista as pv

from mech5.manager import H5File, SegmentedDatasetH5File

from scipy.spatial import cKDTree
import matplotlib.pyplot as plt


class GeometryPostProcessor:

    def __init__(self, cell_size: float = 1.):
        self.cell_size = 1.


    def decorate(self):
        ...
    

    def make_tree(self, point_cloud: np.ndarray):
        self.tree = cKDTree(point_cloud)
        print("tree done")


    def distance(self, point_cloud: np.ndarray, array: np.ndarray):
        distances, indices = self.tree.query(array)
        distances = distances * self.cell_size  
        nearest = point_cloud[indices]
        return distances, indices, nearest
    

    def fit_ellipse(self ):
        ...


    def project(self):
        ...


class VoxelGeometryPostProcessor(GeometryPostProcessor):

    def __init__(self, h5: SegmentedDatasetH5File, cell_size = 1) -> None:
        super().__init__(cell_size)
        self.h5 = h5
        self.surface = None
    

    def make_tree(self, samples: int = None):
        surface = self.h5.read(f"{self.h5._surface}/voxels")

        if samples is not None and samples < len(surface):
            idx = np.random.choice(len(surface), size=samples, replace=False)
            self.surface = surface[idx]
        
        else:
            self.surface = surface
        
        return super().make_tree(self.surface)
    

    def distance(self, pore_id):
        pore = self.h5.query_pore(pore_id)
        array = np.asarray([pore["cx_pix"], pore["cy_pix"], pore["cz_pix"]])
        return super().distance(self.surface, array)


    def all_distances(self):
        """wraps distance to loop over all pores"""
        pore_id = self.h5.read(f"{self.h5._root}/pores/ID")
        distances = []
        indices = []
        nearest = []

        for p in tqdm(pore_id, desc="Computing distances"):
            d, idx, n = self.distance(p)
            distances.append(d)
            indices.append(idx)
            nearest.append(n)

        
        self.h5.write(f"{self.h5._pores}/distance", np.asarray(distances))
        self.h5.write(f"{self.h5._pores}/nearest", np.vstack(nearest))
        
    

def test_tree():
    h5 = SegmentedDatasetH5File(filename="/home/ale/Desktop/example/cyl_3.7_27_v3.h5", mode="r")
    v = GeometryPostProcessor(cell_size=3.7)
    p_test = np.array([[10, 10, 10], [60, 60, 60], [9, 9, 9]])*50
    with h5 as h:
        surface = h.read("/ct/surface/voxels")[:100000]
        print(surface.shape)
        v.make_tree(surface)
        print(v.distance(surface, p_test))


def test_distance():
    
    surface = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    queries = np.array([
        [0.1, 0.0, 0.0],   # nearest: [0,0,0], dist = 0.1
        [0.9, 0.0, 0.0],   # nearest: [1,0,0], dist = 0.1
        [0.0, 0.4, 0.0],   # nearest: [0,0,0], dist = 0.4
        [0.0, 0.0, 0.6],   # nearest: [0,0,1], dist = 0.4
        [0.1, 1.0, 0.0],
    ])

    processor = GeometryPostProcessor(cell_size=1.0)
    processor.make_tree(surface)
    distances, indices, nearest = processor.distance(surface, queries)

    # --- numeric output
    print("Distances:", distances)
    print("Nearest indices:", indices)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(surface[:, 0], surface[:, 1], surface[:, 2], marker="o", s=80, label="Surface")
    ax.scatter(queries[:, 0], queries[:, 1], queries[:, 2], marker="^", s=80, label="Query points")

    for q, n in zip(queries, nearest):
        ax.plot([q[0], n[0]], [q[1], n[1]], [q[2], n[2]])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    plt.show()


def test_voxel_distance():
    h5 = SegmentedDatasetH5File(filename="/home/ale/Desktop/example/cyl_3.7_27_v3.h5", mode="r")
    processor = VoxelGeometryPostProcessor(h5, cell_size=1.0)
    samples = 5000

    with h5 as h:
        surface = h5.read("/ct/surface/voxels/")[: 1000]
        
        processor.make_tree(1000)
        distances, indices, nearest = processor.distance(2, 1000)
        
        pore = h.query_pore(2)
        centroid = np.asarray([pore["cx_pix"], pore["cy_pix"], pore["cz_pix"]])
        print(nearest)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(surface[:, 0], surface[:, 1], surface[:, 2], marker="o", s=5, label="Surface")
        ax.scatter(nearest[0], nearest[1], nearest[2], marker="^", s=10, label="Query points")
        ax.scatter(centroid[0], centroid[1], centroid[2], marker="^", s=10, label="Query points")

        ax.plot([centroid[0], nearest[0]], [centroid[1], nearest[1]], [centroid[2], nearest[2]])

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend()
        plt.show()


def test_all_distances():
    h5 = SegmentedDatasetH5File(filename="/home/ale/Desktop/example/cyl_3.7_27_v3.h5", mode="a")
    processor = VoxelGeometryPostProcessor(h5, cell_size=3.7)
    with h5 as h:
        processor.make_tree()
        processor.all_distances()

        h.inspect()
        print(h.read("/ct/pores/nearest").shape)


def test_display_distances():
    h5 = SegmentedDatasetH5File(filename="/home/ale/Desktop/example/cyl_3.7_27_v3.h5", mode="r")
    with h5 as h:
        cx = h.read("/ct/pores/cx_pix")       # x-coordinates
        cy = h.read("/ct/pores/cy_pix")       # y-coordinates
        cz = h.read("/ct/pores/cz_pix")       # z-coordinates
        near = h.read("/ct/pores/nearest")    # optional, can be used for coloring
        dist = h.read("/ct/pores/distance")
        points = np.stack([cx, cy, cz], axis=-1)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=20, alpha=0.3)
        ax.scatter(near[:, 0], near[:, 1], near[:, 2], s=20, alpha=0.8)
        for p, n, d in zip(points, near, dist):
            print(np.inner(p-n, p-n)**0.5, d)
            # ax.plot([p[0], n[0]], [p[1], n[1]], [p[2], n[2]], color="gray", alpha=0.5)


        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        ax.set_zlabel("Z (pixels)")
        ax.set_title("3D Pore Positions")

        # plt.show()
        


if __name__ == "__main__":
    # test_tree()
    # test_distance()
    # test_voxel_distance()
    test_all_distances()
    # test_display_distances()
    ...