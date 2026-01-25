import os
import sys
sys.path.append('../../src/')

from mech5.manager import H5File, SegmentedDatasetH5File

from typing import Union, List

import numpy as np
import pandas as pd

import yaml

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
        pore_id = self.h5.read(self.h5._pores + "/ID")
        voxel_stack = []
        voxel_lengths = []
        voxel_offsets = []
        for p in pore_id:
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
        self.columns_to_h5(self.h5._surface, "/voxels", ["X", "Y", "Z"], "Label == @self.surface_label")


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