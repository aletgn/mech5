import os
import sys
sys.path.append('../../src/')

from typing import Tuple, Union, List
from itertools import product

import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.ops import unary_union

from tqdm import tqdm

from mech5.manager import H5File, SegmentedDatasetH5File
from sklearn.decomposition import IncrementalPCA


class GeometryPostProcessor:
    """
    Post-processing utilities for voxel-based geometries.

    This class provides methods to decorate voxel centres into cube vertices,
    project voxel geometries onto planes, construct planar polygons, and compute
    areas and unions in projected space. All geometric quantities are expressed
    in physical units defined by ``cell_size``.
    """
    def __init__(self, cell_size: float = 1.) -> None:
        """
        Initialise the geometry post-processor.

        Parameters
        ----------
        cell_size : float, optional
            Physical size of a voxel edge. All coordinates and areas are
            expressed in units derived from this scale.
        """
        self.cell_size = cell_size
        self.C_pix = np.zeros(shape=(3, ))
        self.C_unit = self.C_pix * self.cell_size
        self.R = np.eye(3, 3)
        self.shape = None


    def decorate(self, points: np.ndarray) -> np.ndarray:
        """
        Decorate voxel centres with their cube vertices.

        Each input point is interpreted as a voxel centre in voxel units.
        The method returns the eight cube vertices per voxel, scaled to
        physical coordinates using ``cell_size``.

        Parameters
        ----------
        points : ndarray of shape (N, 3)
            Voxel centre coordinates in voxel units.

        Returns
        -------
        ndarray of shape (N, 8, 3)
            Physical coordinates of the cube vertices for each voxel.
        """
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
        return self.C_unit + ((self.cell_size * points[:, np.newaxis, :] + offsets[np.newaxis, :, :] - self.C_unit) @ self.R)


    def make_tree(self, point_cloud: np.ndarray) -> None:
        """
        Construct a KD-tree from a point cloud.

        Parameters
        ----------
        point_cloud : ndarray of shape (N, 3)
            Reference point cloud in voxel units.
        """
        self.tree = cKDTree(point_cloud)
        print("tree done")


    def pca(self, array: np.ndarray, batch_size: int, prior: List) -> None:
        self.C_pix = array.mean(axis = 0)
        self.C_unit = self.cell_size * self.C_pix
        Q = array - self.C_pix

        ipca = IncrementalPCA(n_components=3, batch_size=int(batch_size))
        for start in tqdm(range(0, Q.shape[0], ipca.batch_size), desc="Incremental PCA"):
            end = start + ipca.batch_size
            ipca.partial_fit(Q[start:end])

        axes = ipca.components_

        if prior is not None:
            if sorted(prior) != [0, 1, 2]:
                raise ValueError("Prior must be a permutation of [0, 1, 2].")
            axes = axes[prior]

        self.R = axes.T

        print(f"Rotation matrix: {self.R}")
        print(f"Det(R): {np.linalg.det(self.R)}")

        if np.linalg.det(self.R) < 0:
            print("Negative determinant. Swapping one axis")
            self.R[:, -1] *= -1
        else:
            print("Positive determinant. Keeping axes")


    def distance(self, point_cloud: np.ndarray, array: np.ndarray) -> Tuple[np.ndarray]:
        """
        Compute nearest-neighbour distances using the KD-tree.

        Distances are returned in physical units.

        Parameters
        ----------
        point_cloud : ndarray of shape (N, 3)
            Reference point cloud in voxel units.
        array : ndarray of shape (M, 3)
            Query points in voxel units.

        Returns
        -------
        distances : ndarray of shape (M,)
            Euclidean distances to nearest neighbours in physical units.
        indices : ndarray of shape (M,)
            Indices of the nearest neighbours in the reference cloud.
        nearest : ndarray of shape (M, 3)
            Coordinates of the nearest neighbours in voxel units.
        """
        distances, indices = self.tree.query(array)
        distances = distances * self.cell_size
        nearest = point_cloud[indices]
        return distances, indices, nearest


    def project(self, points: np.ndarray,
                n: np.ndarray, m: np.ndarray,
                o: np.ndarray = np.zeros(shape=(3, ))) -> Tuple[np.ndarray]:
        """
        Project decorated voxel geometry onto a plane.

        The plane is defined by an origin ``o`` and an orthonormal basis
        constructed from vectors ``n`` (normal) and ``m`` (in-plane).

        Parameters
        ----------
        points : ndarray of shape (N, 3)
            Voxel centres in voxel units.
        n : ndarray of shape (3,)
            Plane normal vector.
        m : ndarray of shape (3,)
            In-plane direction vector orthogonal to ``n``.
        o : ndarray of shape (3,), optional
            Origin of the projection plane in physical coordinates.

        Returns
        -------
        decorated : ndarray of shape (N, 8, 3)
            Decorated voxel vertices in physical coordinates.
        decorated_flat : ndarray of shape (8N, 3)
            Flattened decorated vertices.
        distances : ndarray of shape (8N,)
            Signed distances of vertices from the plane.
        projected : ndarray of shape (8N, 3)
            Orthogonal projection of vertices onto the plane.
        plane_coords : ndarray of shape (8N, 3)
            Coordinates in the plane reference frame.
        """
        cols = points.shape[1]
        assert cols == 3

        # Plane normal
        _n = n/np.linalg.norm(n)

        # m to form a plane basis with n
        _m = m/np.linalg.norm(m)

        assert np.inner(_n, _m) < 1e-10

        # l to complete the basis
        l = np.cross(_n, _m)
        _l = l/np.linalg.norm(l)

        # Rotation matrix
        R = np.vstack([_l, _m, _n])

        decorated = self.decorate(points)
        decorated_flat = decorated.reshape(-1, cols)
        distances = np.dot(decorated_flat - o, _n)
        projected = decorated_flat - np.outer(distances[:, None], _n)
        plane_coords = (projected - o) @ R.T

        self.shape = decorated.shape
        return decorated, decorated_flat, distances, projected, plane_coords


    def polygon(self, points: np.ndarray):
        """
        Construct a convex hull polygon from planar points.

        Parameters
        ----------
        points : ndarray of shape (N, 2)
            Planar coordinates.

        Returns
        -------
        shapely.geometry.Polygon
            Convex hull polygon.
        """
        cols = points.shape[1]
        assert cols == 2
        return Polygon(points).convex_hull


    def polygon_area(self, points: np.ndarray) -> float:
        """
        Compute the area of a planar polygon.

        Parameters
        ----------
        points : ndarray of shape (N, 2)
            Planar polygon vertices.

        Returns
        -------
        float
            Polygon area in physical units squared.
        """
        return self.polygon(points).area


    def polygon_vertices(self, points: np.ndarray) -> np.ndarray:
        """
        Return polygon vertices without closure repetition.

        Parameters
        ----------
        points : ndarray of shape (N, 2)
            Planar polygon vertices.

        Returns
        -------
        ndarray of shape (M, 2)
            Polygon vertices with the closing vertex removed.
        """
        return np.array(self.polygon(points).exterior.coords)[0: -1]


    def projected_2_polygons(self, points: np.ndarray) -> np.ndarray:
        """
        Reshape flattened projected points back to per-voxel polygons.

        Parameters
        ----------
        points : ndarray of shape (8N, 2 or 3)
            Flattened projected coordinates.

        Returns
        -------
        ndarray
            Reshaped array of projected voxel polygons.
        """
        return points.reshape(self.shape)


    def union_polygon(self, points: np.ndarray) -> None:
        """
        Compute the union of multiple projected polygons.

        Parameters
        ----------
        points : ndarray of shape (N, M, 2)
            Array of planar polygons, one per voxel.
        """
        self.polygons = [self.polygon(p[:, :2]) for p in points]
        self.union = unary_union(self.polygons)


    def union_area(self) -> float:
        """
        Return the area of the polygon union.

        Returns
        -------
        float
            Union area in physical units squared.
        """
        return self.union.area


    def union_vertices(self) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Return vertices of the union polygon.

        For simple polygons, vertices are returned directly. For disjoint
        projections, a list of vertex arrays is returned.

        Returns
        -------
        ndarray or list of ndarray
            Union polygon vertices without closure repetition.
        """
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
    """
    Geometry post-processing utilities specialised for voxelised pore data.

    This class extends ``GeometryPostProcessor`` by interfacing with an HDF5
    segmented dataset, providing voxel-aware distance computations and planar
    projections of pore geometries. All geometric operations are internally
    performed in physical units derived from ``cell_size``.
    """


    def __init__(self, h5: SegmentedDatasetH5File, cell_size: float = 1.) -> None:
        """
        Initialise the voxel geometry post-processor.

        Parameters
        ----------
        h5 : SegmentedDatasetH5File
            Handle to the segmented HDF5 dataset containing voxel and pore data.
        cell_size : float, optional
            Physical size of a voxel edge.
        """
        super().__init__(cell_size)
        self.h5 = h5
        self.surface = None


    def make_tree(self, samples: int = None) -> None:
        """
        Construct a KD-tree from surface voxels.

        Optionally subsamples the surface voxels before building the tree.
        Surface coordinates are assumed to be expressed in voxel units.

        Parameters
        ----------
        samples : int, optional
            Number of surface voxels to randomly sample. If ``None``, all
            surface voxels are used.
        """
        surface = self.h5.read(f"{self.h5._surface}/voxels")

        if samples is not None and samples < len(surface):
            idx = np.random.choice(len(surface), size=samples, replace=False)
            self.surface = surface[idx]

        else:
            self.surface = surface

        return super().make_tree(self.surface)


    def distance(self, pore_id: int) -> Tuple[np.ndarray]:
        """
        Compute distance from a pore centre to the nearest surface voxel.

        The pore centre is read in voxel units and distances are returned
        in physical units.

        Parameters
        ----------
        pore_id : int
            Identifier of the pore.

        Returns
        -------
        distances : ndarray
            Distance to the nearest surface voxel in physical units.
        indices : ndarray
            Index of the nearest surface voxel.
        nearest : ndarray
            Coordinates of the nearest surface voxel in voxel units.
        """
        pore = self.h5.query_pore(pore_id)
        array = np.asarray([pore["cx_pix"], pore["cy_pix"], pore["cz_pix"]])
        return super().distance(self.surface, array)


    def all_distances(self) -> None:
        """
        Compute distances for all pores in the dataset.

        Distances are written to the HDF5 file both in voxel units and
        physical units.
        """
        pore_id = self.h5.read(f"{self.h5._root}/pores/ID")
        distances = []
        indices = []
        nearest = []

        for p in tqdm(pore_id, desc="Computing distances"):
            d, idx, n = self.distance(p)
            distances.append(d)
            indices.append(idx)
            nearest.append(n)

        self.h5.write(f"{self.h5._pores}/distance_pix", np.asarray(distances))
        self.h5.write(f"{self.h5._pores}/distance_unit", np.asarray(distances)*self.cell_size)
        self.h5.write(f"{self.h5._pores}/nearest", np.vstack(nearest))


    def pca(self, batch_size: int, prior: List):
        voxels = self.h5.read(f"{self.h5._surface}/voxels")
        super().pca(voxels, batch_size, prior)
        self.h5.write(f"{self.h5._surface}/R", self.R)
        self.h5.write(f"{self.h5._surface}/C_pix", self.C_pix)
        self.h5.write(f"{self.h5._surface}/C_unit", self.C_unit)


    def check_pca(self):
        try:
            self.C_pix = self.h5.read(f"{self.h5._surface}/C_pix")
            self.C_unit = self.h5.read(f"{self.h5._surface}/C_unit")
            self.R = self.h5.read(f"{self.h5._surface}/R")
            print("Found PCA.")
        except:
            print("PCA did not run.")


    def project_pore(self, pore_id: int, n: np.ndarray, m: np.ndarray, o: np.ndarray = None) -> float:
        """
        Project a pore voxel geometry onto a plane and compute its area.

        The pore is decorated into voxel cubes, projected onto the plane
        defined by ``n`` and ``m``, and the union area of the projected
        polygons is returned.

        Parameters
        ----------
        pore_id : int
            Identifier of the pore.
        n : ndarray of shape (3,)
            Plane normal vector.
        m : ndarray of shape (3,)
            In-plane direction vector orthogonal to ``n``.
        o : ndarray of shape (3,), optional
            Origin of the projection plane. If ``None``, the pore centroid
            is used.

        Returns
        -------
        float
            Projected pore area in physical units squared.
        """
        self.check_pca()
        pore, _ = self.h5.query_pore_voxels(pore_id)
        if o is None:
            o = pore.mean(axis=0)
        decorated, decorated_flat, distances, projected, plane_coords = self.project(pore, n, m, o)
        self.union_polygon(self.projected_2_polygons(plane_coords))
        return self.union_area() # unit!


    def all_project_pore(self, n: np.ndarray, m: np.ndarray, o: np.ndarray = None) -> List[float]:
        """
        Project all pores onto a plane and store projected areas.

        Areas are stored both in physical units and voxel units. The square
        root of the area is also stored as a characteristic projected length.

        Parameters
        ----------
        n : ndarray of shape (3,)
            Plane normal vector.
        m : ndarray of shape (3,)
            In-plane direction vector orthogonal to ``n``.
        o : ndarray of shape (3,), optional
            Origin of the projection plane.

        Returns
        -------
        list of float
            Projected pore areas in physical units squared.
        """
        pore_id = self.h5.read(f"{self.h5._root}/pores/ID")
        normal_str = np.array2string(n, separator='_')
        area = []
        for p in tqdm(pore_id, desc=f"Projecting {normal_str}"):
            area.append(self.project_pore(p, n, m, o))
        self.h5.write(f"{self.h5._pores}/proj_unit_{normal_str}", np.asarray(area))
        self.h5.write(f"{self.h5._pores}/mur_unit_{normal_str}", np.asarray(area)**0.5)
        self.h5.write(f"{self.h5._pores}/proj_pix_{normal_str}", np.asarray(area)/(self.cell_size**2))
        self.h5.write(f"{self.h5._pores}/mur_pix_{normal_str}", (np.asarray(area)**0.5)/(self.cell_size**2))
        return area


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

    proc = GeometryPostProcessor(cell_size=3.)
    o = np.array([0., 0., 0.])

    cell = 1  # size of each cube
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

    # centres = np.array([centre1, centre2])

    # o = centres.mean(axis=0)

    points = proc.decorate(centres)
    points_flat = proc.decorate(centres).reshape(-1, 3)

    n = np.array([0., 1., 0.])
    m = np.array([0., 0., 1.])

    # n = np.array([1., 1., 1.])
    _n = n/np.linalg.norm(n)
    _m = m/np.linalg.norm(m)
    assert np.inner(_n, _m) < 1e-10

    # m = np.array([-1., 1., 0.])
    # m /= np.linalg.norm(m)

    l = np.cross(_n, _m)
    _l = l/np.linalg.norm(l)

    distances = np.dot(points_flat - o, _n)
    projected_3d = points_flat - np.outer(distances[:, None], _n)

    R = np.vstack([_l, _m, _n])
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

    ax.scatter(centres[:, 0]*proc.cell_size, centres[:, 1]*proc.cell_size, centres[:, 2]*proc.cell_size)
    ax.scatter(ff[:, 0], ff[:, 1], ff[:, 2])
    ax.scatter(jj[:, 0], jj[:, 1], jj[:, 2], marker="o")

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
    print("Union area:", aa)
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


def test_project_pores():
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    h5 = SegmentedDatasetH5File(filename="/home/ale/Desktop/example/cyl_3.7_27_v3.h5",
                                mode="r")
    v = VoxelGeometryPostProcessor(h5, 3.7)
    with h5 as h:
        area = v.project_pore(2, np.array([1., 0., 0.]), np.array([0., 1., 0.]))
        v.all_project_pore(np.array([1., 0., 0.]), np.array([0., 1., 0.]))
        v.all_project_pore(np.array([0., 1., 0.]), np.array([0., 0., 1.]))
        v.all_project_pore(np.array([0., 0., 1.]), np.array([1., 0., 0.]))
        # print(area.shape)
        # fig = plt.figure()
        # ax = fig.add_subplot(projection="3d")
        # # ax.scatter(area[:, 0], area[:, 1])
        # for p in area:
        #     faces = [[p[0], p[1], p[5], p[4]],
        #             [p[2], p[3], p[7], p[6]],
        #             [p[0], p[3], p[7], p[4]],
        #             [p[1], p[2], p[6], p[5]],
        #             [p[0], p[1], p[2], p[3]],
        #             [p[4], p[5], p[6], p[7]]
        #     ]
        #     poly = Poly3DCollection(faces, alpha=0.3, edgecolor="k")
        #     ax.add_collection3d(poly)
        # plt.show()


def test_pca():
    h5 = SegmentedDatasetH5File(filename="/home/ale/Desktop/example/fiji.h5",
                                mode="r")
    v = GeometryPostProcessor(3.7)
    with h5 as h:
        voxels = h.read("/ct/surface/voxels")
        v.pca(voxels, 1e5, [1, 2, 0])

    v = VoxelGeometryPostProcessor(h5, 3.7)
    with h5 as h:
        v.pca(1e5, [1, 2, 0])

    # fig = plt.figure()
    # ax = fig.add_subplot(projection="3d", proj_type="ortho")

    # idx = np.random.choice(voxels.shape[0], size=10000, replace=False)
    # sample =  ((voxels[idx] - v.C_pix) @ v.R)
    # ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s = 2)
    # ax.axis("equal")
    # plt.show()


def test_pca_merged():
    h5 = SegmentedDatasetH5File(filename="/home/ale/Desktop/example/Fiji-2.0-10.h5",
                                mode="r")

    ha = SegmentedDatasetH5File(filename="/home/ale/Desktop/example/Fiji-2.0a-10.h5",
                                mode="r")

    hb = SegmentedDatasetH5File(filename="/home/ale/Desktop/example/Fiji-2.0b-10.h5",
                                mode="r")

    with h5 as h:
        vx = h.read("/ct/surface/voxels")
        print(vx.shape, vx.min(axis=0), vx.max(axis=0))
    idx = np.random.choice(vx.shape[0], size=1000, replace=False)

    with ha as a:
        va = ha.read("/ct/surface/voxels")
        print(va.shape, va.min(axis=0), va.max(axis=0))
    ida = np.random.choice(va.shape[0], size=1000, replace=False)

    with hb as b:
        vb = hb.read("/ct/surface/voxels")
        print(vb.shape, vb.min(axis=0), vb.max(axis=0))
    idb = np.random.choice(vb.shape[0], size=1000, replace=False)

    v = VoxelGeometryPostProcessor(h5, 3.7)
    with h5 as h:
        v.pca(1e5, [1, 2, 0])

    vb[:, 2] += va[:, 2].max()

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d", proj_type="ortho")

    sx =  ((vx[idx] - v.C_pix) @ v.R)
    va =  ((va[ida] - v.C_pix) @ v.R)
    vb =  ((vb[idb] - v.C_pix) @ v.R)

    ax.scatter(sx[:, 0], sx[:, 1], sx[:, 2], s = 2, c="r")
    ax.scatter(va[:, 0], va[:, 1], va[:, 2], s = 2, c="g")
    ax.scatter(vb[:, 0], vb[:, 1], vb[:, 2], s = 2, c="b")
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
    # test_polycube()
    # test_project_pores()
    # test_pca()
    # test_pca_merged()
    ...