import os
import sys
sys.path.append('../../src/')

from typing import Tuple
from itertools import product

import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.ops import unary_union

from tqdm import tqdm

from mech5.manager import H5File, SegmentedDatasetH5File


class GeometryPostProcessor:

    def __init__(self, cell_size: float = 1.) -> None:
        self.cell_size = 1.
        self.C = np.zeros(shape=(3, ))
        self.R = np.ones(shape=(3, 3))
        self.shape = None


    def decorate(self, points):
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

        # return point + (self.cell_size / 2) * vertices
        offsets = (self.cell_size / 2) * vertices
        return points[:, np.newaxis, :] + offsets[np.newaxis, :, :]


    def make_tree(self, point_cloud: np.ndarray) -> None:
        self.tree = cKDTree(point_cloud)
        print("tree done")


    def distance(self, point_cloud: np.ndarray, array: np.ndarray) -> Tuple[np.ndarray]:
        distances, indices = self.tree.query(array)
        distances = distances * self.cell_size
        nearest = point_cloud[indices]
        return distances, indices, nearest


    def project(self, points: np.ndarray,
                n: np.ndarray, m: np.ndarray,
                o: np.ndarray = np.zeros(shape=(3, ))) -> Tuple[np.ndarray]:

        cols = points.shape[1]
        assert cols == 3
        self.decorated_shape = points.shape

        # Plane normal
        n = np.array([1., 1., 1.])
        n /= np.linalg.norm(n)

        # m to form a plane basis with n
        m = np.array([-1., 1., 0.])
        m /= np.linalg.norm(m)

        # l to complete the basis
        l = np.cross(n, m)

        # Rotation matrix
        R = np.vstack([l, m, n])

        decorated = self.decorate(points)
        decorated_flat = decorated.reshape(-1, cols)
        distances = np.dot(decorated_flat - o, n)
        projected = decorated_flat - np.outer(distances[:, None], n)
        plane_coords = (projected - o) @ R.T

        return decorated, decorated_flat, distances, projected, plane_coords


    def polygon(self, points: np.ndarray):
        cols = points.shape[1]
        assert cols == 2
        return Polygon(points).convex_hull


    def polygon_area(self, points: np.ndarray):
        return self.polygon(points).area


    def polygon_vertices(self, points: np.ndarray):
        return np.array(self.polygon(points).exterior.coords)[0: -1]


    def projected_2_polygons(self, points: np.ndarray):
        return points.reshape(self.shape)


    def union_polygon(self, points: np.ndarray):
        """wrap polygon for an array N x X x 2 decorated points where N is the numbers of polygons"""
        self.polygons = [self.polygon(p[:, :2]) for p in points]
        self.union = unary_union(self.polygons)


    def union_area(self):
        return self.union.area


    def union_vertices(self):
        """First and last vertex is repeated to close the polygon. Keep only once."""
        if self.union.geom_type == 'Polygon':
            print("Joint projection.")
            return np.array(self.union.exterior.coords)[0: -1]

        elif self.union.geom_type == 'MultiPolygon':
            print("Disjoint projection")
            return [np.array(poly.exterior.coords) for poly in self.union.geoms]
        else:
            raise Exception("Fatal error.")


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
    o = np.array([0., 0., 0.])

    cell = proc.cell_size  # size of each cube
    centre1 = np.array([1.0, 1.0, 1.0])
    centre2 = centre1 + np.array([cell, 0, 0])
    centre3 = centre1 + np.array([0, cell, 0])
    centre4 = centre1 + np.array([cell, cell, 0])

    # top layer (Z = 1.0 + cell)
    centre5 = centre1 + np.array([0, 0, cell])
    centre6 = centre2 + np.array([0, 0, cell])
    centre7 = centre3 + np.array([0, 0, cell])
    centre8 = centre4 + np.array([0, 0, cell])
    centre9 = 5*centre4 + np.array([0, 0, cell])

    # Combine all cubes
    centres = np.array([centre1, centre2, centre3, centre4,
                        centre5, centre6, centre7, centre8, centre9])

    # o = centres.mean(axis=0)

    points = proc.decorate(centres)
    points_flat = proc.decorate(centres).reshape(-1, 3)

    n = np.array([1., 1., 1.])
    n /= np.linalg.norm(n)

    m = np.array([-1., 1., 0.])
    m /= np.linalg.norm(m)

    l = np.cross(n, m)

    distances = np.dot(points_flat - o, n)
    projected_3d = points_flat - np.outer(distances[:, None], n)

    R = np.vstack([l, m, n])
    coords = (projected_3d - o) @ R.T

    pp, ff, dd, jj, cc = proc.project(centres, n, m, o)
    print(pp.shape, ff.shape, dd.shape, jj.shape, cc.shape)
    assert np.all(pp - points == 0)
    assert np.all(ff - points_flat == 0)
    assert np.all(dd - distances == 0)
    assert np.all(jj - projected_3d == 0)
    assert np.all(cc - coords == 0)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.scatter(points_flat[:, 0], points_flat[:, 1], points_flat[:, 2])
    ax.scatter(projected_3d[:, 0], projected_3d[:, 1], projected_3d[:, 2], marker="X")

    ax.scatter(centres[:, 0], centres[:, 1], centres[:, 2])
    ax.scatter(ff[:, 0], ff[:, 1], ff[:, 2])
    ax.scatter(jj[:, 0], jj[:, 1], jj[:, 2], marker="X")

    for p in pp:
        faces = [[p[0], p[1], p[5], p[4]],
                 [p[2], p[3], p[7], p[6]],
                 [p[0], p[3], p[7], p[4]],
                 [p[1], p[2], p[6], p[5]],
                 [p[0], p[1], p[2], p[3]],
                 [p[4], p[5], p[6], p[7]]
        ]
        poly = Poly3DCollection(faces, alpha=0.3, edgecolor="k")
        ax.add_collection3d(poly)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.axis("equal")

    fig = plt.figure()
    ax = fig.add_subplot()
    for c in coords.reshape(points.shape):
        ax.scatter(c[:, 0], c[:, 1])
    print(coords.reshape(points.shape).shape)
    for c in cc.reshape(points.shape):
        ax.scatter(c[:, 0], c[:, 1])

    for c in coords.reshape(points.shape):
        polygon = Polygon(c[:, :2]).convex_hull
        polya = polygon.area
        vertices = np.array(polygon.exterior.coords)

        poly = proc.polygon(c[:, :2])
        area = proc.polygon_area(c[:, :2])
        vert = proc.polygon_vertices(c[:, :2])

        print(polya - area)
        # print(vert - vertices)

        ax.scatter(vertices[:, 0], vertices[:, 1])
        ax.fill(vertices[:, 0], vertices[:, 1], alpha=0.2, edgecolor='k', zorder=0)

        ax.scatter(vert[:, 0], vert[:, 1], marker="x")
        ax.fill(vert[:, 0], vert[:, 1], alpha=0.2, edgecolor='r', zorder=0)

    polygons = [Polygon(c[:, :2]).convex_hull for c in coords.reshape(points.shape)]
    union_poly = unary_union(polygons)
    union_area = union_poly.area
    print("Union area:", union_area)

    proc.shape = points.shape
    proc.union_polygon(proc.projected_2_polygons(coords))
    aa = proc.union_area()
    print(aa - union_area)

    if union_poly.geom_type == 'Polygon':
        print("joint projection")
        vertices = np.array(union_poly.exterior.coords)
        print(vertices[0: -1].shape)

        ax.fill(vertices[:,0], vertices[:,1], alpha=.1,
                edgecolor='red', facecolor='none', linewidth=2, label='Union', zorder=-1)
        
        vv = proc.union_vertices()
        ax.fill(vv[:,0], vv[:,1], alpha=.1,
                edgecolor='red', facecolor='none', linewidth=2, label='Union', zorder=-1)

    elif union_poly.geom_type == 'MultiPolygon':
        print("disjoint projection")
        for poly in union_poly.geoms:
            verts = np.array(poly.exterior.coords)
            # print(verts[0: -1].shape)
            ax.fill(verts[:, 0], verts[:, 1], alpha=0.1, edgecolor='red',
                    facecolor='none', linewidth=2, label='Union', zorder=-1)
            

        vv = proc.union_vertices()
        for v in vv:
            ax.fill(v[:, 0], v[:, 1], alpha=0.1, edgecolor='blue',
                    facecolor='none', linewidth=2, label='Union', zorder=-1)

    ax.axis("equal")
    plt.show()


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