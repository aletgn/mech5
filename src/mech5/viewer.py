import os
import sys
sys.path.append('../../src/')

from typing import Union, List

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

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
        self.x_scale = "linear"
        self.y_scale = "linear"
        self.z_scale = "linear"
        self.x_lim = None
        self.y_lim = None
        self.z_lim = None
        self.axis = None
        self.labels = [None]*len(self.h5)
        self.alpha = 0.5
        self.edgecolor = "k"
        self.cmap = "RdYlBu_r"
        self.point_scale = 1e-3
        self.elev = None
        self.azim = None
        self.proj = "persp"

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
                    edgecolor=self.edgecolor, label=l, alpha=self.alpha)

        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        ax.set_xlim(self.x_lim)
        ax.set_ylim(self.y_lim)
        ax.set_xscale(self.x_scale)
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

        if path_size is not None:
            size, _ = self.query_datasets(path_size)
            size = [s*self.point_scale for s in size]
        else:
            size = [None] * len(self.h5)

        if path_color is not None:
            color, _ = self.query_datasets(path_color)
        else:
            color = [None] * len(self.h5)

        fig, ax = plt.subplots(dpi=self.dpi)
        for d_1, d_2, s, c, l in zip(data_1, data_2, size, color, self.labels):
            ax.scatter(d_1, d_2, s=s, c=c, cmap=self.cmap,
                       edgecolor=self.edgecolor, alpha=self.alpha, label=l)

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

        if path_size is not None:
            size, _ = self.query_datasets(path_size)
            size = [s*self.point_scale for s in size]
        else:
            size = [self.point_scale] * len(self.h5)

        if path_color is not None:
            color, _ = self.query_datasets(path_color)
        else:
            color = [None] * len(self.h5)

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        for d_1, d_2, d_3, s, l in zip(data_1, data_2, data_3, size, self.labels):
            ax.scatter(d_1, d_2, d_3, s=s, cmap=self.cmap, label=l, edgecolors=self.edgecolor)

        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        ax.set_ylabel(self.z_label)
        ax.set_xlim(self.x_lim)
        ax.set_ylim(self.y_lim)
        ax.set_ylim(self.z_lim)
        ax.axis(self.axis)
        ax.view_init(elev=self.elev, azim=self.azim)
        ax.set_proj_type(self.proj)

        plt.tight_layout()
        plt.legend()
        if self.save:
            plt.savefig(f"{self.folder}/{self.plot_name}.{self.format}",
                        dpi=self.dpi, format=self.format, bbox_inches="tight")
        else:
            plt.show()


class H5PlotRoughness(H5Plot):

    def __init__(self, *h5file):
        super().__init__(*h5file)


    def scatter_2d(self, path: str, x: int=0, y: int=1, color: bool = False, samples: int=1000):
        z = list({0, 1, 2} - {x, y})[0]
        data, _ = self.query_datasets(path)

        fig, ax = plt.subplots(dpi=self.dpi)
        for d, l in zip(data, self.labels):

            if samples is not None and samples < len(d):
                idx = np.random.choice(len(d), size=samples, replace=False)
                d = d[idx]

            c = d[:, z] if color else None
            ax.scatter(d[:, x], d[:, y], s=self.point_scale, c=c, cmap=self.cmap,
                       edgecolor=self.edgecolor, alpha=self.alpha, label=l)

        ax.axis(self.axis)
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


    def scatter_3d(self, path: str, color=None, samples: int=1000):
        data, _ = self.query_datasets(path)

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        for d, l in zip(data, self.labels):

            if samples is not None and samples < len(d):
                idx = np.random.choice(len(d), size=samples, replace=False)
                d = d[idx]
            c = d[:, color] if color is not None else None

            ax.scatter(d[:, 0], d[:, 1], d[:, 2], s=self.point_scale, c=c, cmap=self.cmap,
                       edgecolor=self.edgecolor, alpha=self.alpha, label=l)

        ax.axis(self.axis)
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


    def inspect_partition(self, path: str, ID: str = None,
                          three: bool = False, samples: int = 1000):

        points = []

        for h in self.h5:
            if ID is None:
                for II in h.read("roughness/partitions/ID"):
                    points.append(h.query_raster_partition(II)["points"])
            else:
                points.append(h.query_raster_partition(ID)["points"])

        n = len(points)
        if not three:
            
            fig, axes = plt.subplots(1, max(n, 1), figsize=(4*n, 4), dpi=self.dpi)
            if n == 1:
                axes = np.array([axes])
            for i, ax in enumerate(axes):
                if i < n:
                    p = points[i]
                    n = min(samples, len(p))
                    idx = np.random.choice(len(p), size=n, replace=False)
                    im = ax.imshow(p)
                    ax.axis(self.axis)
                    ax.set_xlabel(self.x_label)
                    ax.set_ylabel(self.y_label)
                    ax.set_xlim(self.x_lim)
                    ax.set_ylim(self.y_lim)
                    ax.set_title(f"Raster {i}")
                else:
                    ax.axis('off')
            ax.axis(self.axis)
            ax.set_xlabel(self.x_label)
            ax.set_ylabel(self.y_label)
            ax.set_xlim(self.x_lim)
            ax.set_ylim(self.y_lim)

        else:
            fig, axes = plt.subplots(1, max(n, 1), figsize=(4*n, 4), subplot_kw={'projection': '3d'} if n > 0 else None)
            if n == 1:
                axes = np.array([axes])
            for i in range(len(axes)):
                ax = axes[i]
                if i < n:
                    arr = points[i]
                    ny, nx = arr.shape
                    x = np.arange(nx)
                    y = np.arange(ny)
                    X, Y = np.meshgrid(x, y)
                    surf = ax.plot_surface(X, Y, arr, cmap=self.cmap, linewidth=0, antialiased=True)
                    ax.set_title(f"Raster {i}")
                else:
                    ax.axis('off')

            ax.axis(self.axis)
            ax.set_xlabel(self.x_label)
            ax.set_ylabel(self.y_label)
            ax.set_zlabel(self.y_label)
            ax.set_xlim(self.x_lim)
            ax.set_ylim(self.y_lim)
            ax.set_zlim(self.y_lim)

        ax.tick_params("both", right=1, top=1, direction="in")

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