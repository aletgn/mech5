import os
import sys
sys.path.append('../../src/')
from pathlib import Path

from typing import Union, List

import numpy as np
import pandas as pd

import yaml
from tqdm import tqdm

import pyvista as pv

from mech5.manager import H5File, SegmentedDatasetH5File


class SpreadsheetToH5File:
    """
    Load a spreadsheet into a DataFrame and write selected columns to an HDF5 file.

    Attributes
    ----------
    h5 : H5File
        HDF5 file wrapper object for reading/writing datasets.
    spreadsheet : str
        Path to the spreadsheet file.
    reader : callable
        Function to read the spreadsheet (e.g., pd.read_csv, pd.read_excel).
    data : pd.DataFrame
        DataFrame containing the spreadsheet data.
    _dict : dict or None
        Optional storage for column data.
    """


    def __init__(self, h5: H5File, spreadsheet: str, reader: callable,  **rargs) -> None:
        """
        Initialise the SpreadsheetToH5File object.

        Parameters
        ----------
        h5 : H5File
            HDF5 file wrapper object.
        spreadsheet : str
            Path to the spreadsheet file.
        reader : callable
            Function to read the spreadsheet into a DataFrame.
        **rargs
            Additional keyword arguments passed to the reader function.
        """
        self.h5 = h5
        self.spreadsheet = spreadsheet
        self.reader: callable = reader
        self.data: pd.DataFrame = reader(spreadsheet, **rargs)
        self._dict = None


    def set_dict(self, key_path: str) -> None:
        """
        Load a YAML file and assign its contents to the internal dictionary.

        Parameters
        ----------
        key_path : str
            Path to the YAML file containing the mapping to load.

        Returns
        -------
        None
        """
        with open(key_path, "r") as f:
            self._dict = yaml.safe_load(f)


    def get_columns(self, col_names: Union[str, List], query: str = "") -> np.ndarray:
        """
        Return one or more columns from the DataFrame as a NumPy array,
        optionally filtered by a query string.

        Parameters
        ----------
        col_names : str or list of str
            Column(s) to extract.
        query : str, optional
            Query string to filter rows, by default "" (no filtering).

        Returns
        -------
        np.ndarray
            Array of column data. For multiple columns, shape is (n_rows, n_columns).
        """
        series: pd.Series = self.data.query(query) if query else self.data
        return series[col_names].to_numpy()


    def to_h5(self, path: str, dataset_name: str, column: np.ndarray) -> None:
        """
        Write a single column array to an HDF5 file, using an optional name mapping.

        Parameters
        ----------
        path : str
            Base path within the HDF5 file where the dataset will be written.
        dataset_name : str
            Name of the dataset to write, optionally remapped via the internal dictionary.
        column : np.ndarray
            Array containing the column data.

        Returns
        -------
        None
        """
        dataset_name = dataset_name.replace("/", "")
        if self._dict is None:
            self.h5.write(path + dataset_name, column)
            return

        if dataset_name not in self._dict:
            return

        self.h5.write(f"{path}/{self._dict[dataset_name]}", column)


    def columns_to_h5(self, path: str, dataset_name: str, col_names: Union[str, List], query: str = "") -> None:
        """
        Write one or more columns to an HDF5 file at the specified path.

        Parameters
        ----------
        path : str
            Base path within the HDF5 file where the dataset will be written.
        dataset_name : str
            Name of the dataset under which the column data will be stored.
        col_names : str or list of str
            Column name or list of column names to write.
        query : str, optional
            Query string used to filter rows prior to writing, by default "".

        Returns
        -------
        None
        """
        series = self.get_columns(col_names, query)
        self.to_h5(path, dataset_name, series)


    def all_columns_to_h5(self, path: str, query: str = "") -> None:
        """
        Write all columns in the DataFrame to an HDF5 file as separate datasets.

        Parameters
        ----------
        path : str
            Base path within the HDF5 file where the datasets will be written.
        query : str, optional
            Query string used to filter rows prior to writing, by default "".

        Returns
        -------
        None
        """
        col_names = self.data.columns
        for c in col_names:
            series = self.get_columns(c, query)
            self.to_h5(path, c, series)


class FijiSegmentedDataToH5File(SpreadsheetToH5File):
    """
    Class for writing segmented Fiji spreadsheet data to an HDF5 file.

    Inherits from `SpreadsheetToH5File` and adds handling for pores and surface voxels.
    """


    def __init__(self, h5, spreadsheet, reader: callable) -> None:
        """
        Initialise with an HDF5 file, spreadsheet, and reader.

        Parameters
        ----------
        h5 : SegmentedDatasetH5File
            HDF5 file object to write data to. Must be a SegmentedDatasetH5File.
        spreadsheet : Any
            Spreadsheet data source.
        reader : callable
            Object capable of reading spreadsheet data.

        Raises
        ------
        NotImplementedError
            If `h5` is not an instance of `SegmentedDatasetH5File`.
        """
        super().__init__(h5, spreadsheet, reader)

        if not isinstance(h5, SegmentedDatasetH5File):
            raise NotImplementedError(f"{h5.__class__.__name__} not allowed.")

        self.surface_label = None

        config_path = Path(__file__).resolve().parents[2] / "config" / "fiji-keys.yaml"
        self.set_dict(config_path)


    def set_surface_label(self, surface_label: int) -> None:
        """
        Set the surface label and write it to the HDF5 file.

        Parameters
        ----------
        surface_label : int
            Label value representing the surface.

        Returns
        -------
        None
        """
        self.surface_label = surface_label
        self.h5.write(self.h5._surface + "/surface_label", surface_label)


    def check_surface_label(self):
        if self.surface_label is None:
            try:
                self.surface_label = self.h5.read(self.h5._surface + "/surface_label")
                print(f"Found surface label in dataset: {self.surface_label}.")
            except:
                raise Exception("Surface label not found.")
        else:
            print(f"Surface label: {self.surface_label}")


    def write_all_pore_descriptors(self) -> None:
        """
        Write all pore descriptor columns to the HDF5 file,
        excluding the surface label.

        Returns
        -------
        None
        """
        self.all_columns_to_h5(self.h5._pores, "Label != @self.surface_label")


    def write_pore_voxels(self) -> None:
        """
        Write voxel coordinates for each pore to the HDF5 file.

        - Computes offsets for each pore's voxel block.
        - Writes X, Y, Z columns and the voxel offsets.
        - Excludes voxels with the surface label.

        Returns
        -------
        None
        """
        self.check_surface_label()
        pore_id = self.h5.read(self.h5._pores + "/ID")
        voxel_stack = []
        voxel_lengths = []
        voxel_offsets = []
        for p in tqdm(pore_id, desc="Processing pores"):
            voxels = self.data.query("Label == @p")[["X", "Y", "Z"]].to_numpy()
            voxel_stack.append(voxels)
            voxel_lengths.append(voxels.shape[0])

        offsets = np.zeros(len(voxel_lengths) + 1, dtype=int)
        offsets[1:] = np.cumsum(voxel_lengths)
        self.columns_to_h5(self.h5._pores, "/voxels", ["X", "Y", "Z"], "Label != @self.surface_label")
        self.h5.write(self.h5._pores + "/voxels_offsets", offsets)


    def write_surface_voxels(self) -> None:
        """
        Write voxel coordinates corresponding to the surface label to the HDF5 file.

        - Writes X, Y, Z columns for surface voxels only.

        Returns
        -------
        None
        """
        self.check_surface_label()
        self.columns_to_h5(self.h5._surface, "/voxels", ["X", "Y", "Z"], "Label == @self.surface_label")


class ArrayToVTK:
    """
    Convert arrays of 3D points into VTK cube-based grids.

    This class provides two conversion strategies depending on array size:
    explicit cube merging for small arrays, and vectorised unstructured
    hexahedral grid construction for large arrays.
    """


    def __init__(self, cell_side: float = 1.0) -> None:
        """
        Initialise the converter.

        Parameters
        ----------
        cell_side : float, optional
            Edge length of each cubic cell.
        """
        self.cell_side = cell_side
        self.C_pix = np.zeros(shape=(3, ))
        self.C_unit = self.C_pix * self.cell_side
        self.R = np.eye(3, 3)


    def small_array_to_cubes(self, array: np.ndarray) -> pv.PolyData:
        """
        Convert a small array of 3D points into merged cube geometry.

        Each point is used as the centre of an axis-aligned cube. Individual
        cube meshes are created and merged into a single PolyData object.

        Parameters
        ----------
        array : numpy.ndarray
            Array of shape (N, 3) containing cube centre coordinates.

        Returns
        -------
        pyvista.PolyData
            Merged cube geometry representing all input points.
        """
        assert array.shape[1] == 3
        grids = []
        for a in array:
            cube = pv.Cube(center=(a[0], a[1], a[2]),
                        x_length= 2 * self.cell_side/2,
                        y_length= 2 * self.cell_side/2,
                        z_length= 2 * self.cell_side/2)
            grids.append(cube)
        grid = pv.merge(grids)
        return grid


    def large_array_to_cubes(self, array: np.ndarray, samples: int = None) -> pv.UnstructuredGrid:
        """
        Convert a large array of 3D points into an unstructured hexahedral grid.

        Cubes are constructed implicitly by defining shared point coordinates
        and explicit hexahedral cell connectivity, providing a memory-efficient
        representation suitable for large datasets. Optional random subsampling
        may be applied prior to grid construction.

        Parameters
        ----------
        array : numpy.ndarray
            Array of shape (N, 3) containing cube centre coordinates.
        samples : int or None, optional
            Number of points to randomly sample from the input array. If None,
            all points are used.

        Returns
        -------
        pyvista.UnstructuredGrid
            Unstructured grid containing hexahedral cells centred on the input
            points.
        """
        N = array.shape[0]
        assert array.shape[1] == 3

        if samples is not None:
            assert samples < N
            mask = np.random.choice(N, samples, replace=False)
            array = array[mask]
            N = array.shape[0]
            print(array.shape)

        # cell geometry
        vertices = np.array([[-1, -1, -1],
                             [ 1, -1, -1],
                             [ 1,  1, -1],
                             [-1,  1, -1],
                             [-1, -1,  1],
                             [ 1, -1,  1],
                             [ 1,  1,  1],
                             [-1,  1,  1]]) * self.cell_side/2
        points = ((array[:, None, :]*self.cell_side + vertices - self.C_unit) @ self.R).reshape(-1, 3)
        base = np.arange(0, 8 * N, 8)
        cells = np.c_[np.full(N, 8),
                      base + 0, base + 1, base + 2, base + 3,
                      base + 4, base + 5, base + 6, base + 7]

        grid = pv.UnstructuredGrid(cells,
                                   np.full(N, pv.CellType.HEXAHEDRON),
                                   points)

        return grid


class SegmentedH5FileToVTK(ArrayToVTK):
    """
    Convert segmented HDF5 voxel datasets into VTK unstructured grids.

    This class adapts voxel-based segmentation data stored in an HDF5 file
    into VTK-compatible cube representations, optionally attaching per-voxel
    metadata as cell data.
    """

    def __init__(self, h5: SegmentedDatasetH5File, cell_side: float = 1.):
        """
        Initialise the converter for a segmented HDF5 dataset.

        Parameters
        ----------
        h5 : SegmentedDatasetH5File
            HDF5 file interface providing access to segmented voxel data.
        cell_side : float, optional
            Edge length of each cubic voxel cell.
        """
        super().__init__(cell_side)
        self.h5 = h5


    def surface_voxels_to_vtu(self, samples: int) -> pv.UnstructuredGrid:
        """
        Convert surface voxels to a VTK unstructured grid.

        Surface voxel centres are read from the HDF5 file and converted into
        hexahedral cells. Optional random subsampling is applied prior to grid
        construction.

        Parameters
        ----------
        samples : int
            Number of surface voxels to randomly sample.

        Returns
        -------
        pyvista.UnstructuredGrid
            Unstructured grid representing the sampled surface voxels.
        """
        voxels = self.h5.read(f"{self.h5._surface}/voxels")
        return self.large_array_to_cubes(voxels, samples=samples)


    def pore_voxels_to_vtu(self, add_fields: bool = True) -> pv.UnstructuredGrid:
        """
        Convert pore voxels to a VTK unstructured grid.

        Pore voxel centres are converted into hexahedral cells. When enabled,
        per-object pore attributes stored in the HDF5 file are expanded to
        per-voxel values using voxel offsets and attached as cell data.

        Parameters
        ----------
        add_fields : bool, optional
            If True, expand and attach pore-level datasets as per-voxel cell
            data.

        Returns
        -------
        pyvista.UnstructuredGrid
            Unstructured grid representing pore voxels, optionally enriched
            with cell data fields.
        """
        voxels = self.h5.read(f"{self.h5._pores}/voxels")
        grid = self.large_array_to_cubes(voxels)

        if add_fields:
            counts = np.diff(self.h5.read(f"{self.h5._pores}/voxels_offsets"))
            for name in self.h5.list_datasets(self.h5._pores):
                print(f"Saving dataset to vtu: {name}")
                if "voxel" in name:
                    ...
                elif "nearest" in name:
                    ...
                else:
                    obj_data = self.h5.read(name)
                    # expand per-voxel
                    voxel_data = np.repeat(obj_data, counts)
                    if voxel_data.size != grid.n_cells:
                        raise ValueError("Expanded field size does not match number of voxels")
                    grid.cell_data[name] = voxel_data

        return grid


import surfalize
class PluxToH5File:
    """Uses surfalize package."""

    def __init__(self, h5: H5File, surface: surfalize.Surface):
        self.h5 = h5
        self.surface = surface


    def write_surface(self):
        self.h5.write("/roughness/raster/surface", self.surface.data)
        self.h5.write("/roughness/raster/pix_x", self.surface.step_x)
        self.h5.write("/roughness/raster/pix_y", self.surface.step_y)

        print("Pixel size along x- and y-axis", self.surface.step_x, self.surface.step_x)


class DarkFieldXrayMicroscopyH5FileToVTK:
    
    def __init__(self, h5file, name: str="Untitled"):
        self.h5 = h5file
        self.data_n_idx = None
        self.grid = None

        self.tol = 1e-5
        self.lower_threshold = -np.inf # Lower bound of voxel values
        self.upper_threshold = +np.inf # Lower bound of voxel values
        self.spacing = (1., 1., 10.) # Voxel size
        self.slice_offset = 0

        self.off_screen = False
        self.show_edges = False
        self.cmap = "viridis"
        self.clim = None
        self.scalar_bar_args = {}
        # Example
        # self.scalar_bar_args={"title": "Orientation",
        #                  "vertical": True,     # vertical colorbar
        #                  "position_x": 0,   # x-position in figure (0 left, 1 right)
        #                  "position_y": 0,    # y-position in figure (0 bottom, 1 top)
        #                  "height": 0.08,        # fraction of figure height
        #                  "width": 0.8,        # fraction of figure width
        #                  "title_font_size": 35,
        #                  "label_font_size": 12}
        self.rgb = False
        self.bounds = None
        self.show_bounds = False

        # BBox properties
        self.tick_pos = "outside"
        self.minor_ticks = False
        self.all_edges = True
        self.use_2d = False
        self.use_3d_text = True
        self.bbox_grid = None
        self.x_title = "X"
        self.y_title = "Y"
        self.z_title = "Z"
        self.n_xlabels = 5
        self.n_ylabels = 5
        self.n_zlabels = 5
        self.fmt = "%2.f"
        self.location = "outer"
        self.font_size = None
        self.font_family = None

        # Axis widget
        self.show_axes = True

        # Camera settings
        self.set_camera = False
        self.camera_position = "iso" # xy
        self.camera_roll = 270
        self.camera_azimuth = 360

        # Centre camera -- not implemented yet
        # center = [(bounds[0]+bounds[1])/2,
        #   (bounds[2]+bounds[3])/2,
        #   (bounds[4]+bounds[5])/2]

        # Set camera to tightly frame the bounds
        # pl.camera_position = [
        #     (center[0] + 1.5*(bounds[1]-bounds[0]),
        #      center[1] + 1.5*(bounds[3]-bounds[2]),
        #      center[2] + 1.5*(bounds[5]-bounds[4])),  # camera location
        #     center,  # focal point
        #     (0, 0, 1)  # view up
        # ]

        # Screenshot parameters
        self.trasparent = False
        self.window_size = (800, 800)
        self.scale = 1

        # Export extension
        # - .vti for image data (uniform grid)
        # - .vtk for unstructured grids
        # - .vtu for unstructured grids (XML-based format)
        self.ext = None


    def initialise_grid(self, dataset, transpose=(1,2,0)):
        data = self.h5.read(dataset).transpose(transpose)
        self.data_n_idx = len(data.shape)
        px, py, pz = self.spacing
        mask = ~np.isnan(data)
        
        self.x, self.y, self.z = np.nonzero(mask)
        values = data[self.x, self.y, self.z]
        n = len(values)

        offsets = np.array([[0,0,0],
                            [px,0,0],
                            [px,py,0],
                            [0,py,0],
                            [0,0,pz],
                            [px,0,pz],
                            [px,py,pz],
                            [0,py,pz]])
        

        points = (np.repeat(np.array([self.x*px, self.y*py, self.z*(pz+self.slice_offset)]).T, 8, axis=0) \
                  + np.tile(offsets, (n,1))).astype(float)

        # Cell indices (8 = vertices in a hexahedron)
        cells = np.arange(n*8).reshape(n, 8)
        cells = np.hstack([np.full((n,1),8), cells])

        # Initialise grid (next line: 12 = VTK_HEXAHEDRON)
        self.grid = pv.UnstructuredGrid(cells, np.full(n, 12), points)
        self.ext = ".vtu"


    def dataset2grid(self, dataset: str, transpose=(1,2,0)):
        print(f"Add {dataset} dataset")
        data = self.h5.read(dataset).transpose(transpose)
        values = data[self.x, self.y, self.z]
        self.grid.cell_data[dataset] = values


    def make_unstructured_grid(self, dataset: np.ndarray, transpose=(1,2,0), label="Data"):
        data = self.h5.read(dataset).transpose(transpose)
        self.data_n_idx = len(data.shape)
        px, py, pz = self.spacing

        if self.data_n_idx == 3:
            # Keep voxels with value above the threshold
            print("Data without RGB")
            # mask = data > self.lower_threshold
            mask = ~np.isnan(data)

        elif self.data_n_idx == 4:
            # Keep voxels with value above the threshold (intensity, non-black) and
            # exclude voxels having color 1, 1, 1, i.e. 255, 255, 255 (white)
            # mask = (np.linalg.norm(data, axis=-1) > self.lower_threshold) & \
            #     (~np.all(np.abs(data - self.upper_threshold) < self.tol, axis=-1))
            mask = ~np.isnan(data).any(axis=-1)
            print("Data with RGB")
        else:
            raise Exception("Non-voxel array.")

        # Generate cells upon masked voxels
        x, y, z = np.nonzero(mask)
        values = data[x, y, z]
        n = len(values)

        # Voxel vertices
        offsets = np.array([[0,0,0],
                            [px,0,0],
                            [px,py,0],
                            [0,py,0],
                            [0,0,pz],
                            [px,0,pz],
                            [px,py,pz],
                            [0,py,pz]])

        points = (np.repeat(np.array([x*px, y*py, z*(pz+self.slice_offset)]).T, 8, axis=0) \
                  + np.tile(offsets, (n,1))).astype(float)

        # Cell indices (8 = vertices in a hexahedron)
        cells = np.arange(n*8).reshape(n, 8)
        cells = np.hstack([np.full((n,1),8), cells])

        # Initialise grid (next line: 12 = VTK_HEXAHEDRON)
        self.grid = pv.UnstructuredGrid(cells, np.full(n, 12), points)

        # Populate grid
        if self.data_n_idx == 3:
            self.grid.cell_data[label] = values
        else:
            colors = (data[x, y, z] * 255).astype(np.uint8)
            self.grid.cell_data[label] = colors


        self.bounds = self.grid.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
        self.ext = ".vtu"

        # If opacity array is present, consider this
        # opacity = np.ones_like(values)
        # plotter.add_mesh(grid, scalars="values", opacity=opacity, cmap="viridis")
    

    def view(self):
        self.pl = pv.Plotter(off_screen=self.off_screen)
        self.pl.add_mesh(self.grid,
                            show_edges=self.show_edges,
                            cmap=self.cmap,
                            clim=self.clim,
                            scalar_bar_args=self.scalar_bar_args,
                            rgb=self.rgb)

        if self.show_bounds:
            # _ = pl.add_bounding_box(line_width=1, color='black')
            self.pl.show_bounds(ticks=self.tick_pos, minor_ticks=self.minor_ticks,
                                all_edges=self.all_edges, use_2d=self.use_2d, use_3d_text=self.use_3d_text,
                                xtitle=self.x_title, ytitle=self.y_title, ztitle=self.z_title,
                                n_xlabels=self.n_xlabels, n_ylabels=self.n_ylabels, n_zlabels=self.n_zlabels,
                                fmt=self.fmt, location=self.location,
                                bounds=self.bounds,
                                axes_ranges=self.bounds,
                                font_size=self.font_size, font_family=self.font_family,
                                # bold=bold,# color=color
                                )

        if self.show_axes:
            self.pl.show_axes()

        if self.set_camera:
            self.pl.camera_position = self.camera_position
            self.pl.camera.roll = self.camera_roll
            self.pl.camera.azimuth = self.camera_azimuth

        self.pl.show()

    
    def export_vt_format(self, path: str):
        self.grid.save(path + self.ext)

TEST_H5 = "/home/ale/Desktop/example/test.h5"
TEST_FILE = "/home/ale/Desktop/example/measure.csv"
YAML_FILE = "../../config/fiji-keys.yaml"


def test_h5file_open_close():
    h5 = SegmentedDatasetH5File(TEST_H5, "a", overwrite=True)
    h5.open()
    print("File opened:", isinstance(h5.file, type(h5.file)))
    h5.close()
    print("File closed:", h5._file is None)


def test_spreadsheet_to_h5():
    h5 = SegmentedDatasetH5File(TEST_H5, "w", overwrite=True)
    s = SpreadsheetToH5File(h5, TEST_FILE, pd.read_csv)
    s.set_dict(YAML_FILE)
    with h5 as h5:
        s.all_columns_to_h5("/")
        print("Inspecting HDF5 file hierarchy:")
        h5.inspect()


def test_read_written_column():
    h5 = SegmentedDatasetH5File(TEST_H5, "r")
    h5.open()
    try:
        data = h5.read("cx_pix")
        print("Read cx_pix:", data)
    except KeyError:
        print("Column 'cx_pix' not found in HDF5.")
    h5.close()


def test_cleanup():
    h5 = SegmentedDatasetH5File(TEST_H5, "r")
    try:
        h5.delete_file()
        print("HDF5 file deleted:", not os.path.exists(TEST_H5))
    except Exception as e:
        print("Error deleting file:", e)


TEST_MEASURE = "/home/ale/Desktop/example/measure.csv"
TEST_VOXELS = "/home/ale/Desktop/example/voxels.csv"
TEST_FIJI = "../../config/fiji-keys.yaml"

def read_descriptors():
    h5 = SegmentedDatasetH5File(TEST_H5, "w", overwrite=True)
    s = FijiSegmentedDataToH5File(h5, TEST_MEASURE, pd.read_csv)
    s.set_dict(TEST_FIJI)

    with h5 as h:
        s.set_surface_label(2)
        h.write_created()
        s.write_all_pore_descriptors()
        h.write_name("Test-Dataset")
        h.write_modified()
        h.inspect()
        print(h.read("common/modified"))
        print(h.read("common/created"))


def read_voxels():
    h5 = SegmentedDatasetH5File(TEST_H5, "a", overwrite=False)
    s = FijiSegmentedDataToH5File(h5, TEST_VOXELS, pd.read_csv)
    s.set_dict(TEST_FIJI)

    with h5 as h:
        s.set_surface_label(2)
        s.write_pore_voxels()
        s.write_surface_voxels()
        h.write_modified()
        h.inspect()


def validate_voxels():
    h5 = SegmentedDatasetH5File(TEST_H5, "a", overwrite=False)
    pd_voxels = pd.read_csv(TEST_VOXELS)
    print(pd_voxels.query("Label != 2").shape)
    print(pd_voxels.query("Label == 2").shape)

    with h5 as h:
        h.inspect()
        print(h.read("ct/pores/voxels").shape)
        print(h.read("ct/surface/voxels").shape)
        print(h.read("ct/pores/voxels_offsets").shape)

        pore_id = list(h.read("ct/pores/ID"))
        voxels = h.read("ct/pores/voxels")
        offsets = h.read("ct/pores/voxels_offsets")

        for p in pore_id:
            idx = pore_id.index(p)
            start = offsets[idx]
            end = offsets[idx + 1]

            vox = voxels[start: end]

            assert p in pd_voxels.query("Label != 2")["Label"].to_list()
            pd_vox = pd_voxels.query("Label == @p")[["X", "Y", "Z"]].to_numpy()
            diff = vox - pd_vox
            # print(diff)
            assert np.all(diff == 0)

        for p in pore_id:
            vox = h5.query_pore(p)["voxels"]
            assert p in pd_voxels.query("Label != 2")["Label"].to_list()
            pd_vox = pd_voxels.query("Label == @p")[["X", "Y", "Z"]].to_numpy()
            diff = vox - pd_vox
            print(diff)
            assert np.all(diff == 0)


if __name__ == "__main__":
    # print("=== Test open/close ===")
    # test_h5file_open_close()

    # print("\n=== Test writing spreadsheet columns to HDF5 ===")
    # test_spreadsheet_to_h5()

    # print("\n=== Test reading written column ===")
    # test_read_written_column()

    # print("\n=== Test cleanup ===")
    # test_cleanup()

    # print("\n=== Read descriptors ===")
    # read_descriptors()

    # print("\n=== Read voxels ===")
    # read_voxels()

    # print("\n=== Validate voxels ===")
    # validate_voxels()
    ...