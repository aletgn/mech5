import sys
sys.path.append('../../src/')

import os

import h5py
from typing import Optional, Any, List, Dict, Tuple, Union

import numpy as np

from mech5.util import utc, Mask, Criterion, TrueMask


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
        self.query_queue = TrueMask()
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


    def read(self, path: str) -> np.ndarray:
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


    def list_groups(self, gpr_name: str) -> List[str]:
        """
        List all subgroup names within a given HDF5 group.

        Parameters
        ----------
        gpr_name : str
            Name of the parent HDF5 group.

        Returns
        -------
        List[str]
            List of full paths to subgroups within the group.
        """
        group = self.file[gpr_name]
        return [gpr_name + "/" + key for key, item in group.items() if isinstance(item, h5py.Group)]


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


    def query(self, path_to_group: str, protocol: Union[callable, Criterion, Mask]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query a dataset in an HDF5 group and return the filtered data and mask.

        Parameters
        ----------
        path_to_group : str
            Path to the HDF5 group or dataset to query.
        protocol : callable or class
            class with __call__ method to provide a mask (see util module).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple containing:
            - Filtered data as a NumPy array.
            - Boolean mask indicating which rows satisfy the criterion.
        """
        data = self.read(path_to_group)
        mask = protocol(data)
        return data[mask], mask


    def load_query(self, protocol: Union[callable, Criterion, Mask]) -> None:
        self.query_queue = protocol


    def concatenate(self, dataset: str, other: h5py.File, destination: str):
        """other/destination and MUST BE OPEN already in read/append mode"""
        print(f"Merging: {dataset}")
        arr_self = self.read(dataset)
        arr_other = other.read(dataset)
        print(arr_self.shape)
        print(arr_other.shape)
        arr_destination = np.concatenate([arr_self, arr_other])
        assert arr_destination.shape[0] == arr_self.shape[0] + arr_other.shape[0]
        return arr_destination
        # destination.write(dataset, arr_destination)


    def merge_all_datasets(self, grp: str, other: h5py.File, destination: str):
        for g in self.list_datasets(grp):
            array = self.concatenate(g, other, destination)
            destination.write(g, array)


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


class SegmentedDatasetH5File(H5File):
    """
    HDF5 file specialised for segmented datasets containing pores and optional surface data.

    Provides utilities for locating pores, querying pore data, and reading voxel blocks.
    """

    def __init__(self, filename: str, mode: str, overwrite: bool = False) -> None:
        super().__init__(filename, mode, overwrite)
        """
        Initialise the segmented dataset HDF5 file and set default groups.

        Parameters
        ----------
        filename : str
            Path to the HDF5 file.
        mode : str
            File mode, e.g., 'r', 'w', 'a'.
        overwrite : bool, optional
            If True, overwrite existing file, by default False.
        """
        # default groups
        self._root = "ct"
        self._pores = f"{self._root}/pores"
        self._surface =  f"{self._root}/surface"


    def locate_pore(self, pore_id: int) -> int:
        """
        Find the index of a pore in the pore ID array.

        Parameters
        ----------
        pore_id : int
            The unique pore identifier.

        Returns
        -------
        int
            Index of the pore in the ID array.
        """
        ids = self.read(self._pores + "/ID")
        return np.where(ids == pore_id)[0].squeeze()


    def query_pore_voxels(self, pore_id: int = None, pore_loc: int = None) -> Tuple[np.ndarray, List[int]]:
        """
        Retrieve the voxel coordinates for a single pore.

        Parameters
        ----------
        pore_id : int, optional
            ID of the pore. Required if `pore_loc` is not provided.
        pore_loc : int, optional
            Index of the pore. Overrides `pore_id` if provided.

        Returns
        -------
        tuple[np.ndarray, list[int]]
            - Array of voxel coordinates for the pore (X, Y, Z).
            - List containing the start and end indices in the voxels dataset.
        """
        loc = pore_loc if pore_loc is not None else self.locate_pore(pore_id)
        off = self.read(self._pores + "/voxels_offsets")
        start = off[loc]
        end = off[loc+1]
        return self.read(self._pores + "/voxels")[start: end], [start, end]


    def query_pore(self, pore_id: int) -> Dict[str, np.ndarray]:
        """
        Retrieve all stored attributes and voxel data for a single pore.

        Parameters
        ----------
        pore_id : int
            ID of the pore to query.

        Returns
        -------
        dict[str, Any]
            Dictionary containing pore data:
            - Standard attributes are keyed by their dataset names.
            - Voxel coordinates are stored under 'voxels'.
            - Voxel offsets are stored under 'voxels_offsets'.
        """
        loc = self.locate_pore(pore_id)
        pore = {}
        for g in self.list_datasets(self._pores):
            if "voxels" in g:
                pore["voxels"], pore["voxels_offsets"] = self.query_pore_voxels(pore_loc=loc)
            else:
                pore[g.split("/")[-1]] = self.read(g)[loc]
        return pore


    def index_pores(self, mask: np.array) -> Tuple[np.ndarray, np.ndarray]:
        """
        Re-index pore voxel data after masking.

        Given a boolean mask over pores, this method filters the voxel data and
        recomputes the corresponding voxel offsets such that the remaining pores
        are stored contiguously.

        Parameters
        ----------
        mask : np.array
            Boolean array indicating which pores to retain. Its length must match
            the number of pores encoded by the voxel offsets.

        Returns
        -------
        voxels : np.ndarray
            Stacked array of voxel data corresponding to the retained pores.
        offsets : np.ndarray
            One-dimensional array of voxel offsets for the retained pores, with
            length equal to the number of retained pores plus one.

        Raises
        ------
        ValueError
            If no pores are selected by the mask.
        """
        voxels = self.read(f"{self._pores}/voxels")
        offsets = self.read(f"{self._pores}/voxels_offsets")
        filtered_voxels = []
        filtered_offsets = [0]
        for keep, start, end in zip(mask, offsets[: -1], offsets[1: ]):
            if keep:
                filtered_voxels.append(voxels[start: end])
                filtered_offsets.append(filtered_offsets[-1] + voxels[start:end].shape[0])
        return np.vstack(filtered_voxels), np.array(filtered_offsets)


    def mask_pores(self, path: str, criterion: callable) -> None:
        """
        Apply a mask to all pore-related datasets and re-index voxel data.

        This method queries pore-level data using a user-defined criterion,
        applies the resulting mask to all associated datasets, and rebuilds
        the voxel and voxel offset arrays to remain consistent with the masked
        pore set.

        Parameters
        ----------
        path : str
            Path used to query pore-level data on which the masking criterion
            is applied.
        criterion : callable
            Function used to determine whether a pore is retained. It is passed
            to the query mechanism and must be compatible with it.

        Returns
        -------
        None
        """
        masked_data, mask = self.query(path, criterion)
        for g in self.list_datasets(f"{self._pores}"):
            if "voxels" in g:
                # skip this and re-index below
                ...
            else:
                # print(g, self.read(g)[mask].shape)
                self.write(g, self.read(g)[mask])
                ...

        # re-index pores
        voxels, offsets = self.index_pores(mask)
        self.write(f"{self._pores}/voxels", voxels)
        self.write(f"{self._pores}/voxels_offsets", offsets)


    def merge_segmented_datasets(self, other, destination):

        translation =  [self._pores + "/" + t for t in ["cz_pix", "cz_unit", "zmin_pix", "zmax_pix"]]
        for g in self.list_datasets(f"{self._pores}"):
            print(g)
            arr_bot = self.read(g)
            arr_top = other.read(g)

            if g in translation:
                print("Merging with translation.")
                arr_full = np.concatenate([arr_bot, arr_top + arr_bot.max()])

            elif g == f"{self._pores}/voxels":
                print("Merging voxels.")
                arr_top_shifted = arr_top.copy()
                arr_top_shifted[:, 2] += arr_bot[:, 2].max()
                arr_full = np.concatenate([arr_bot, arr_top_shifted])

            elif g == f"{self._pores}/voxels_offsets":
                print("Merging offsets.")
                arr_full = np.concatenate([arr_bot, arr_top[1: ] + arr_bot[-1] + 1])

            else:
                print("Merging normally.")
                arr_full = np.concatenate([arr_bot, arr_top])

            destination.write(g, arr_full)

        for g in self.list_datasets(f"{self._surface}"):
            arr_bot = self.read(g)
            arr_top = other.read(g)

            if g == f"{self._surface}/voxels":
                arr_top_shifted = arr_top.copy()
                arr_top_shifted[:, 2] += arr_bot[:, 2].max()
                arr_full = np.concatenate([arr_bot, arr_top_shifted])

            elif g == f"{self._surface}/surface_label":
                arr_full = np.array(1)

            else:
                ...
                # arr_full = np.concatenate([arr_bot, arr_top])

            destination.write(g, arr_full)

        # Re-label pores based on surface label (1) -- start from 2
        new_ID = np.arange(2, destination.read(f"{self._pores}/ID").shape[0] + 2)
        destination.write(f"{destination._pores}/ID", new_ID)


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


def test_query():
    with SegmentedDatasetH5File("/home/ale/Desktop/example/test.h5", "r") as h5:
        list_ = [False]*39330
        list_[0] = True
        print(h5.query("ct/pores/volume_pix", Criterion(lambda x: x > 27)))
        print(h5.query("ct/pores/volume_pix", Mask(list_)))


def test_true_query():
    with SegmentedDatasetH5File("/home/ale/Desktop/example/test.h5", "r") as h5:
        print(h5.query("ct/pores/volume_pix", TrueMask()))


def test_query_pore():
    with SegmentedDatasetH5File("/home/ale/Desktop/example/test.h5", "r") as h5:
        h5.inspect()
        print(h5.query_pore(10))


def test_list_groups():
    with H5File("/home/ale/Desktop/example/test.h5", "r", overwrite=True) as h5:
        h5.inspect()
        print(h5.list_groups("/ct"))


def test_merge():

    hb = H5File("/home/ale/Desktop/example/Fiji-2.0b-10.h5", "r", overwrite=True)
    hc = H5File("/home/ale/Desktop/example/Fiji-2.0-10.h5", "a", overwrite=True)
    hb.open()
    hc.open()

    with H5File("/home/ale/Desktop/example/Fiji-2.0a-10.h5", "r", overwrite=True) as ha:
        cat = ha.concatenate("/ct/pores/ID", hb, hc)
        print(ha.read("/ct/pores/ID").shape)
        print(hb.read("/ct/pores/ID").shape)
        print(hc.read("/ct/pores/ID").shape)
        # ha.merge_all_datasets("/ct/pores", hb, hc)
        # hc.inspect()

    hb.close()
    hc.close()


def test_merge_segmented():

    hb = SegmentedDatasetH5File("/home/ale/Desktop/example/Fiji-2.0b-10.h5", "r", overwrite=True)
    hc = SegmentedDatasetH5File("/home/ale/Desktop/example/Fiji-2.0-10.h5", "w", overwrite=True)
    hb.open()
    hc.open()

    with SegmentedDatasetH5File("/home/ale/Desktop/example/Fiji-2.0a-10.h5", "r", overwrite=True) as ha:
        ha.merge_segmented_datasets(hb, hc)
        hc.inspect()

    hb.close()
    hc.close()

if __name__ == "__main__":
    # print("=== Test open and close ===")
    # test_open_and_close()

    # print("\n=== Test write and read ===")
    # test_write_and_read()

    # print("\n=== Test list datasets ===")
    # test_list_datasets()

    # print("\n=== Test delete dataset ===")
    # test_delete_dataset()

    # print("\n=== Test read datasets in group ===")
    # test_read_datasets_in_group()

    # print("\n=== Test delete file ===")
    # test_delete_file()

    # print("\n=== Test query file ===")
    # test_query()

    # print("\n=== Test true query file ===")
    # test_true_query()

    # print("\n=== Test query pore ===")
    # test_query_pore()

    # print("\n=== Test list groups ===")
    # test_list_groups()

    # print("\n=== Test merge files ===")
    # test_merge()

    print("\n=== Test segmented datasets ===")
    test_merge_segmented()
    ...