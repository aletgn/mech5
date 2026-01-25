import sys
sys.path.append('../../src/')

import os

import h5py
from typing import Optional, Any, List, Dict

import numpy as np

from mech5.util import utc


class H5File:
    """
    Context-managed HDF5 file wrapper with convenient read/write and inspection methods.

    Attributes
    ----------
    filename : str
        Path to the HDF5 file.
    mode : str
        File mode ('r', 'w', 'a', etc.).
    overwrite : bool
        If True, allows overwriting existing files when opening in write mode.
    _root : str
        Root path for reading/writing datasets.
    _file : Optional[h5py.File]
        Internal HDF5 file handle.
    """


    def __init__(self, filename: str, mode: str, overwrite: str = False) -> None:
        """
        Initialise H5File object.

        Parameters
        ----------
        filename : str
            Path to the HDF5 file.
        mode : str
            File mode ('r', 'w', 'a', etc.).
        overwrite : bool, optional
            If True, allows overwriting existing files when opening in write mode, by default False.
        """
        self.filename = filename
        self.mode = mode
        self.overwrite = overwrite
        self._file: Optional[h5py.File] = None


    def open(self) -> None:
        """
        Open the HDF5 file.

        Raises
        ------
        FileExistsError
            If file exists in write mode and overwrite is False.
        """
        if self._file is not None:
            return

        if self.mode in ("w", "w-") and os.path.exists(self.filename):
            if not self.overwrite:
                raise FileExistsError(
                    f"HDF5 file '{self.filename}' exists. "
                    "Set overwrite=True to allow destruction.")

        self._file = h5py.File(self.filename, self.mode)


    def close(self) -> None:
        """
        Close the HDF5 file if it is open.
        """
        if self._file is not None:
            self._file.close()
            self._file = None


    def __enter__(self):
        """
        Enter context manager by opening the file.

        Returns
        -------
        H5File
            Self.
        """
        self.open()
        return self


    def __exit__(self, exc_type, exc, tb):
        """
        Exit context manager by closing the file.
        """
        self.close()


    @property
    def file(self) -> h5py.File:
        """
        Access the underlying HDF5 file object.

        Returns
        -------
        h5py.File
            Open HDF5 file handle.

        Raises
        ------
        RuntimeError
            If the file is not open.
        """
        if self._file is None:
            raise RuntimeError("HDF5 file is not open")
        return self._file


    def read(self, path: str) -> Any:
        """
        Read a dataset from the file.

        Parameters
        ----------
        path : str
            Path to the dataset.

        Returns
        -------
        Any
            Data contained in the dataset.
        """
        return self.file[path][()]


    def write(self, path: str, data: Any) -> None:
        """
        Write a dataset to the file, replacing it if it exists.

        Parameters
        ----------
        path : str
            Path to the dataset.
        data : Any
            Data to write.
        """
        if path in self.file:
            del self.file[path]
        self.file.create_dataset(path, data=data)


    def write_name(self, name: str) -> None:
        """Wrap write method and set name."""
        self.write("common/name", name)


    def write_created(self) -> None:
        """Wrap write method and set creation timestamp if not already present."""
        group = self.file.require_group("common")
        if "created" not in group:
            group.create_dataset("created", data=utc())


    def write_modified(self) -> None:
        """Wrap write method and set modification date."""
        self.write("common/modified", utc())


    def delete(self, path: str) -> None:
        """
        Delete a dataset or group from the HDF5 file.

        Parameters
        ----------
        path : str
            Absolute or relative path to the dataset or group to delete.

        Returns
        -------
        None
        """
        if path in self.file:
            del self.file[path]


    def inspect(self) -> None:
        """
        Print the hierarchy of the HDF5 file.
        """
        def print_name(name: str, obj) -> None:
            obj_type = "Group" if isinstance(obj, h5py.Group) else "Dataset"
            print(f"{obj_type}: {name}")

        self.file.visititems(print_name)


    def list_datasets(self, gpr_name: str) -> List[str]:
        """
        List all dataset names in a given group.

        Parameters
        ----------
        gpr_name : str
            Name of the HDF5 group.

        Returns
        -------
        List[str]
            List of dataset paths within the group.
        """
        group = self.file[gpr_name]
        return [gpr_name + "/" + key for key, item in group.items() if isinstance(item, h5py.Dataset)]


    def read_datasets_in_group(self, grp_name: str) -> Dict[str, np.ndarray]:
        """
        Read all datasets in a group into a dictionary.

        Parameters
        ----------
        name : str
            Group name.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary mapping dataset names to their values.
        """
        group = self.file[grp_name]
        out = {}
        for key, item in group.items():
            if isinstance(item, h5py.Dataset):
                out[key] = item[()]
        return out


    def delete_file(self) -> None:
        """
        Permanently delete the HDF5 file from disk.

        Raises
        ------
        RuntimeError
            If the file is currently open.
        FileNotFoundError
            If the file does not exist.
        """
        if self._file is not None:
            raise RuntimeError(
                "Cannot delete HDF5 file while it is open. "
                "Close the file first."
            )

        if not os.path.exists(self.filename):
            raise FileNotFoundError(self.filename)

        os.remove(self.filename)


TEST_FILE =  "./testh5.h5"


def test_open_and_close():
    h5 = H5File(TEST_FILE, "w", overwrite=True)
    h5.open()
    print("File opened:", isinstance(h5.file, type(h5.file)))
    h5.close()
    print("File closed:", h5._file is None)


def test_write_and_read():
    data = np.arange(5)
    with H5File(TEST_FILE, "a", overwrite=True) as h5:
        h5.write("my_dataset", data)
        read_data = h5.read("my_dataset")
        h5.inspect()
    print("Original data:", data)
    print("Read data:", read_data)
    assert np.array_equal(data, read_data)


def test_list_datasets():
    with H5File(TEST_FILE, "w", overwrite=True) as h5:
        h5.write("group1/ds1", np.array([1]))
        h5.write("group1/ds2", np.array([2]))
        h5.inspect()
        datasets = h5.list_datasets("group1")
    print("Datasets in group1:", datasets)


def test_delete_dataset():
    with H5File(TEST_FILE, "w", overwrite=True) as h5:
        h5.write("ct/groupa/a", np.array([10, 20]))
        h5.write("ct/groupa/b", np.array([30, 40]))
        assert "ct/groupa/a" in h5.file
        assert "ct/groupa/b" in h5.file
        h5.delete("ct/groupa/a")
        assert "ct/groupa/a" not in h5.file
        assert "ct/groupa/b" in h5.file
        h5.inspect()


def test_read_datasets_in_group():
    with H5File(TEST_FILE, "w", overwrite=True) as h5:
        h5.write("ct/groupa/a", np.array([10, 20]))
        h5.write("ct/groupa/b", np.array([30, 40]))
        h5.inspect()
        data_dict = h5.read_datasets_in_group("ct/groupa")
    print("Datasets read from group1:", data_dict)


def test_delete_file():
    h5 = H5File(TEST_FILE, "w", overwrite=True)
    h5.open()
    h5.close()
    h5.delete_file()
    print("File exists after delete:", os.path.exists(TEST_FILE))


if __name__ == "__main__":
    print("=== Test open and close ===")
    test_open_and_close()

    print("\n=== Test write and read ===")
    test_write_and_read()

    print("\n=== Test list datasets ===")
    test_list_datasets()

    print("\n=== Test delete dataset ===")
    test_delete_dataset()

    print("\n=== Test read datasets in group ===")
    test_read_datasets_in_group()

    print("\n=== Test delete file ===")
    test_delete_file()
    ...