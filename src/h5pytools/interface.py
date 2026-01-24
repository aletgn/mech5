import os
import sys
sys.path.append('../../src/')

from h5pytools.manager import H5File

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
        Write a column to HDF5 using _dict mapping if available.

        Parameters
        ----------
        name : str
            Original column name.
        column : np.ndarray
            Column data to write.

        Notes
        -----
        - If _dict is None, the column is written with its original name.
        - If _dict is defined but does not contain 'name', the column is skipped.
        """
        with open(key_path, "r") as f:
            self._dict = yaml.safe_load(f)


    def get_columns(self, columns: Union[str, List], query: str = "") -> np.ndarray:
        """
        Return one or more columns from the DataFrame as a NumPy array,
        optionally filtered by a query string.

        Parameters
        ----------
        columns : str or list of str
            Column(s) to extract.
        query : str, optional
            Query string to filter rows, by default "" (no filtering).

        Returns
        -------
        np.ndarray
            Array of column data. For multiple columns, shape is (n_rows, n_columns).
        """
        series: pd.Series = self.data.query(query) if query else self.data
        return series[columns].to_numpy()


    def columns_to_h5(self, path: str, columns: Union[str, List], query: str = "") -> None:
        """
        Write selected column(s) to the HDF5 file at the specified path.

        Parameters
        ----------
        path : str
            Path in the HDF5 file to store the column(s).
        columns : str or list of str
            Column(s) to write.
        query : str, optional
            Query string to filter rows before writing, by default "".
        """
        series = self.get_columns(columns, query)
        self.to_h5(path, series)


    def to_h5(self, path: str, column: np.ndarray) -> None:
        """
        Write a single column array to the HDF5 file.

        Parameters
        ----------
        path : str
             in the HDF5 file.
        column : np.ndarray
            Array containing the column data.
        """
        path = path
        if self._dict is None:
            self.h5.write(self.h5._root + path, column)
            return

        if path not in self._dict:
            return

        self.h5.write(self.h5._root + self._dict[path], column)


    def all_columns_to_h5(self) -> None:
        """
        Write all columns in the DataFrame to the HDF5 file as separate datasets.
        """
        cols = self.data.columns
        for c in cols:
            series = self.get_columns(c)
            self.to_h5(c, series)


class FijiSegmentedDataToH5File(SpreadsheetToH5File):


    def __init__(self, h5, spreadsheet, reader):
        super().__init__(h5, spreadsheet, reader)
        self.h5._root = "/ct/"
        self.h5._pores = self.h5._root + "pores/"
        self.h5._surface = self.h5._root + "surface/"


    def set_voxel_size(self, ):
        ...


    def set_surface_label(self,):
        ...




TEST_H5 = "/home/ale/Desktop/example/test.h5"
TEST_FILE = "/home/ale/Desktop/example/measure.csv"
YAML_FILE = "../../config/fiji-keys.yaml"


def test_h5file_open_close():
    h5 = H5File(TEST_H5, "a", overwrite=True)
    h5.open()
    print("File opened:", isinstance(h5.file, type(h5.file)))
    h5.close()
    print("File closed:", h5._file is None)


def test_spreadsheet_to_h5():
    h5 = H5File(TEST_H5, "a", overwrite=True)
    s = SpreadsheetToH5File(h5, TEST_FILE, pd.read_csv)
    s.set_dict(YAML_FILE)
    with h5 as h5:
        s.all_columns_to_h5()
        print("Inspecting HDF5 file hierarchy:")
        h5.inspect()


def test_read_written_column():
    h5 = H5File(TEST_H5, "r")
    h5.open()
    try:
        data = h5.read("cx_pix")
        print("Read cx_pix:", data)
    except KeyError:
        print("Column 'cx_pix' not found in HDF5.")
    h5.close()


def test_cleanup():
    h5 = H5File(TEST_H5, "r")
    try:
        h5.delete_file()
        print("HDF5 file deleted:", not os.path.exists(TEST_H5))
    except Exception as e:
        print("Error deleting file:", e)


if __name__ == "__main__":
    # print("=== Test open/close ===")
    # test_h5file_open_close()

    # print("\n=== Test writing spreadsheet columns to HDF5 ===")
    # test_spreadsheet_to_h5()

    # print("\n=== Test reading written column ===")
    # test_read_written_column()

    # print("\n=== Test cleanup ===")
    # test_cleanup()
    ...