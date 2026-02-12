import os
import sys
sys.path.append('../../src/')

from typing import Union, List

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

FONT_SIZE = 14
FONT_FAMILY = "sans-serif"
USE_LATEX = False
INTERACTIVE = False
matplotlib.rcParams['font.size'] = FONT_SIZE
matplotlib.rcParams['font.family'] = FONT_FAMILY
matplotlib.rcParams['text.usetex'] = USE_LATEX
matplotlib.rcParams["interactive"] = INTERACTIVE

from mech5.manager import H5File, SegmentedDatasetH5File
from mech5.util import Mask, Criterion, TrueMask


class H5Plot:

    def __init__(self, *h5file: Union[List[H5File], H5File, SegmentedDatasetH5File]) -> None:
        # common
        self.h5: Union[List[H5File], H5File, SegmentedDatasetH5File] = h5file
        self.x_label = None
        self.y_label = None
        self.z_label = None
        self.x_scale = None
        self.y_scale = None
        self.z_scale = None
        self.x_lim = None
        self.y_lim = None
        self.z_lim = None
        self.labels = [None]*len(self.h5)
        self.alpha = 0.5
        self.edgecolor = "k"
        self.cmap = "RdYlBu_r"
        
        # histograms
        self.bins = None
        self.density = True
        
        # output
        self.dpi = 300
        self.format = "pdf"
        self.frontend = None
        self.save = False
        self.folder = "./"
        self.plot_name = "Untitled"


    def query_datasets(self, path: str) -> np.ndarray:
        data = []; mask = []
        for h in self.h5:
            d, m = h.query(path, h.query_queue)
            data.append(d); mask.append(m)
        return data, mask
    

    def histogram(self, path: str) -> None:
        data, _ = self.query_datasets(path)
        fig, ax = plt.subplots(dpi=self.dpi)
        for d, l in zip(data, self.labels):
            ax.hist(d, bins=self.bins, density=self.density,
                    edgecolor=self.edgecolor, label=l)

        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        ax.set_xlim(self.x_lim)
        ax.set_ylim(self.y_lim)
        ax.tick_params("both", right=1, top=1, direction="in")

        plt.legend()
        plt.tight_layout()
        if self.save:
            plt.savefig(f"{self.folder}/{self.plot_name}.{self.format}",
                        dpi=self.dpi, format=self.format, bbox_inches="tight")
        else:
            plt.show()


    def scatter_2d(self, path_1: str, path_2: str,
                   path_size: str = None, path_color: str = None) -> None:
        data_1, _ = self.query_datasets(path_1)
        data_2, _ = self.query_datasets(path_2)
        size = None
        color = None

        if path_size is not None:
            size = self.query_datasets(path_size)

        if path_color is not None:
            color = self.query_datasets(path_color)
        
        fig, ax = plt.subplots(dpi=self.dpi)
        for d_1, d_2, l in zip(data_1, data_2, self.labels):
            ax.scatter(d_1, d_2, s=size, c=color, cmap=self.cmap,
                       edgecolor=self.edgecolor, label=l)

        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        ax.set_xlim(self.x_lim)
        ax.set_ylim(self.y_lim)
        ax.tick_params("both", right=1, top=1, direction="in")
        
        plt.tight_layout()
        plt.legend()
        if self.save:
            plt.savefig(f"{self.folder}/{self.plot_name}.{self.format}",
                        dpi=self.dpi, format=self.format, bbox_inches="tight")
        else:
            plt.show()


    def scatter_3d(self, path_1: str, path_2: str, path_3: str,
                   path_size: str = None, path_color: str = None) -> None:
        data_1, _ = self.query_datasets(path_1)
        data_2, _ = self.query_datasets(path_2)
        data_3, _ = self.query_datasets(path_3)
        size = None
        color = None

        if path_size is not None:
            size = self.query_datasets(path_size)

        if path_color is not None:
            color = self.query_datasets(path_color)
        
        fig, ax = plt.subplots(projection="3d")
        for d_1, d_2, d_3, l in zip(data_1, data_2, data_3, self.labels):
            ax.scatter(d_1, d_2, d_3, s=size, c=color, cmap=self.cmap, label=l)

        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        ax.set_ylabel(self.z_label)
        ax.set_xlim(self.x_lim)
        ax.set_ylim(self.y_lim)
        ax.set_ylim(self.z_lim)
        
        plt.tight_layout()
        plt.legend()
        if self.save:
            plt.savefig(f"{self.folder}/{self.plot_name}.{self.format}",
                        dpi=self.dpi, format=self.format, bbox_inches="tight")
        else:
            plt.show()


def test_query_data():
    h5 = SegmentedDatasetH5File("/home/ale/Desktop/example/test.h5", "r")
    v = H5Plot(h5)
    with h5 as h:
        print(v.query_datasets("ct/pores/volume_pix"))


def test_histogram_data():
    h5 = SegmentedDatasetH5File("/home/ale/Desktop/example/test.h5", "r")
    v = H5Plot(h5)
    with h5 as h:
        print(v.query_datasets("ct/pores/volume_pix"))
        v.histogram("ct/pores/volume_pix")


if __name__ == "__main__":
    # test_query_data()
    test_histogram_data()
    ...