import sys
sys.path.append('../../src/')

from h5pytools.manager import H5File

from typing import Union, List

import numpy as np
import pandas as pd

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


    def set_dict(self):
        ...

    
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
        with self.h5 as h5:
            self.to_h5(path, series)


    def to_h5(self, name: str, column: np.ndarray) -> None:
        """
        Write a single column array to the HDF5 file.

        Parameters
        ----------
        name : str
            Path or dataset name in the HDF5 file.
        column : np.ndarray
            Array containing the column data.
        """
        self.h5.write(self.h5._root + name, column)


    def all_columns_to_h5(self) -> None:
        """
        Write all columns in the DataFrame to the HDF5 file as separate datasets.
        """
        cols = self.data.columns
        with self.h5 as h5:
            for c in cols:
                series = self.get_columns(c)
                self.to_h5(c, series)


class FijiSegmentedDataToH5File(SpreadsheetToH5File):


    def __init__(self, h5, spreadsheet, reader):
        super().__init__(h5, spreadsheet, reader)
        self.h5._root = "/ct/"
        self.h5._pores = self.h5._root + "pores/"
        self.h5._surface = self.h5._root + "surface/"


if __name__ == "__main__":
    TEST_H5 = "/home/ale/Desktop/example/test.h5"
    TEST_FILE = "/home/ale/Desktop/example/measure.csv"
    
    h5 = H5File(TEST_H5, "a", overwrite=True)
    s = FijiSegmentedDataToH5File(h5, TEST_FILE, pd.read_csv)

    # s.columns_to_h5("centroid", ["CX (pix)", "CX (pix)"], )
    # s.column_to_h5("CX (unit)")

    s.all_columns_to_h5()


    h5.open()
    h5.inspect()

    # print(h5.read("Name"))
    h5.close()
    h5.delete_file()
