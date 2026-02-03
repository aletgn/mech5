import os
import sys
sys.path.append('../../src/')

from typing import Tuple
from itertools import product

import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

from tqdm import tqdm

from mech5.manager import H5File, SegmentedDatasetH5File


class GeometryPostProcessor:

    def __init__(self, cell_size: float = 1.) -> None:
        self.cell_size = 1.
        self.C = np.zeros(shape=(3,))
        self.R = np.ones(shape=(3,3))


    def decorate(self, point):
        # offsets = np.array(list(product([-self.cell_size/2, self.cell_size/2],
        #                                 repeat=3)))
        # return point + offsets
        vertices = np.array([
                            [-1, -1, -1],
                            [ 1, -1, -1],
                            [ 1,  1, -1],
                            [-1,  1, -1],
                            [-1, -1,  1],
                            [ 1, -1,  1],
                            [ 1,  1,  1],
                            [-1,  1,  1],
                        ])
        
        return point + (self.cell_size / 2) * vertices


    def make_tree(self, point_cloud: np.ndarray) -> None:
        self.tree = cKDTree(point_cloud)
        print("tree done")


    def distance(self, point_cloud: np.ndarray, array: np.ndarray) -> Tuple[np.ndarray]:
        distances, indices = self.tree.query(array)
        distances = distances * self.cell_size
        nearest = point_cloud[indices]
        return distances, indices, nearest


    def project(self, points: np.ndarray, n: np.ndarray, ):
        ...


    def fit_ellipse(self, points):
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
        distances, indices, nearest = processor.distance(2)

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
    h5 = SegmentedDatasetH5File(filename="/home/ale/Desktop/example/cyl_3.7_27_v3.h5",
                                mode="r")
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

        plt.show()


def to_vtu(name, source):
    from mech5.interface import ArrayToVTK, SegmentedH5FileToVTK
    h5name = source + name
    h5 = SegmentedDatasetH5File(h5name, "r", overwrite=False)
    v = SegmentedH5FileToVTK(h5)
    with h5 as h:
        surface_voxels = v.surface_voxels_to_vtu(samples=100000)
        surface_voxels.save("test_surface.vtu")

        pore_voxels = v.pore_voxels_to_vtu()
        pore_voxels.save("./test_pores.vtu")


def test_decorate():
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    proc = GeometryPostProcessor()

    centre = np.array([0.0, 0.0, 0.0])
    points = proc.decorate(centre)

    points = np.asarray(points)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.scatter(centre[0], centre[1], centre[2])

    faces = [
        [points[0], points[1], points[5], points[4]],
        [points[2], points[3], points[7], points[6]],
        [points[0], points[3], points[7], points[4]],
        [points[1], points[2], points[6], points[5]],
        [points[0], points[1], points[2], points[3]],
        [points[4], points[5], points[6], points[7]] 
    ]

    poly = Poly3DCollection(faces, alpha=0.3, edgecolor="k")
    ax.add_collection3d(poly)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()


def test_project():
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    proc = GeometryPostProcessor()
    o = np.array([0,0,0])
    centre = np.array([10.0, 10.0, 10.0])
    points = proc.decorate(centre)

    n = np.array([0., 1., 0.])
    n /= np.linalg.norm(n)

    m = np.array([0., 0., 1.])
    m /= np.linalg.norm(m)

    l = np.cross(n, m)
    distances = np.dot(points - o, n)
    projected_3d = points - np.outer(distances, n)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.scatter(centre[0], centre[1], centre[2])
    ax.scatter(projected_3d[:, 0], projected_3d[:, 1], projected_3d[:, 2], marker="X")

    faces = [
        [points[0], points[1], points[5], points[4]],
        [points[2], points[3], points[7], points[6]],
        [points[0], points[3], points[7], points[4]],
        [points[1], points[2], points[6], points[5]],
        [points[0], points[1], points[2], points[3]],
        [points[4], points[5], points[6], points[7]] 
    ]

    poly = Poly3DCollection(faces, alpha=0.3, edgecolor="k")
    ax.add_collection3d(poly)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.axis("equal")

    plt.show()


def test_polycube():
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    proc = GeometryPostProcessor()
    o = np.array([0,0,0])
    
    centre1 = np.array([1.0, 1.0, 1.0])
    centre2 = centre1 + np.array([proc.cell_size, 0, 0])
    centres = np.array([centre1, centre2])
    print(centres)
    points = proc.decorate(centres)

    # n = np.array([0., 1., 0.])
    # n /= np.linalg.norm(n)

    # m = np.array([0., 0., 1.])
    # m /= np.linalg.norm(m)

    # l = np.cross(n, m)
    # distances = np.dot(points - o, n)
    # projected_3d = points - np.outer(distances, n)

    # fig = plt.figure()
    # ax = fig.add_subplot(projection="3d")

    # ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    # ax.scatter(centre[0], centre[1], centre[2])
    # ax.scatter(projected_3d[:, 0], projected_3d[:, 1], projected_3d[:, 2], marker="X")

    # faces = [
    #     [points[0], points[1], points[5], points[4]],
    #     [points[2], points[3], points[7], points[6]],
    #     [points[0], points[3], points[7], points[4]],
    #     [points[1], points[2], points[6], points[5]],
    #     [points[0], points[1], points[2], points[3]],
    #     [points[4], points[5], points[6], points[7]] 
    # ]

    # poly = Poly3DCollection(faces, alpha=0.3, edgecolor="k")
    # ax.add_collection3d(poly)

    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    # ax.axis("equal")

    # plt.show()


if __name__ == "__main__":
    # test_tree()
    # test_distance()
    # test_voxel_distance()
    # test_all_distances()
    # test_display_distances()
    # to_vtu("cyl_3.7_27_v3.h5", "/home/ale/Desktop/example/")
    # test_decorate()
    # test_project()
    test_polycube()