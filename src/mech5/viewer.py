import os
import sys
sys.path.append('../../src/')

from typing import Union, List

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LogNorm, SymLogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.ticker import FormatStrFormatter, LogFormatterSciNotation

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


class H5PlotDarkFieldXrayMicroscopy:
    """This class shall be deprecated in the future."""
    def __init__(self, h5: H5File):
        self.h5 = h5

        self.com_phi_min = None
        self.com_phi_max = None

        self.com_chi_min = None
        self.com_chi_max = None

        self.mosa_min = None
        self.mosa_max = None

        self.mis_min = None
        self.mis_max = None

        self.gnd_min = None
        self.gnd_max = None


    def plot_layer(self, layer):
        l = self.h5.query_layer(layer)
        fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(16,10))

        phi = ax[0, 0].imshow(np.where(l["morph_phi"], l["com_phi"], np.nan),
                             vmin=self.com_phi_min,
                             vmax=self.com_phi_max,cmap="viridis")

        chi = ax[0, 1].imshow(np.where(l["morph_chi"], l["com_chi"], np.nan),
                             vmin=self.com_phi_min,
                             vmax=self.com_phi_max,cmap="viridis")

        img = np.where(np.repeat(l["morph_chi"][..., None], 3, axis=-1),
                       l["mosaicity_radial"], np.nan)

        mos = ax[0, 2].imshow(img)
        cax = inset_axes(ax[0, 2], width="25%", height="25%", loc="lower right")
        col = cax.imshow(l["mosaicity_colorbar"], origin="lower",
                         extent=(self.h5.read("/dfxm/processed/min_phi"),
                                 self.h5.read("/dfxm/processed/max_phi"),
                                 self.h5.read("/dfxm/processed/min_chi"),
                                 self.h5.read("/dfxm/processed/max_chi"),))


        mis = ax[1, 0].imshow(np.where(l["morph_phi"], l["misorientation"], np.nan),
                             vmin=self.mis_min,
                             vmax=self.mis_max, cmap="RdYlBu_r")

        gnd = ax[1, 1].imshow(np.where(l["morph_phi"], l["gnd"], np.nan),
                             norm=LogNorm(vmin=self.gnd_min,
                                          vmax=self.gnd_max), cmap="magma")


        try:
            strain = ax[1, 2].imshow(np.where(l["morph_phi"], self.h5.read("dfxm/processed/strain")[0], np.nan))
        except:
            print("Strain data not found")

        fig.colorbar(phi, ax=ax[0, 0])
        fig.colorbar(chi, ax=ax[0, 1])
        fig.colorbar(mis, ax=ax[1, 0])
        fig.colorbar(gnd, ax=ax[1, 1])

        plt.tight_layout()
        plt.show()


    def plot_mask(self, layer):
        l = self.h5.query_layer(layer)

        fig, ax = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(16,10))
        ax[0, 0].imshow(l["com_phi_raw"])
        ax[0, 1].imshow(l["morph_phi"])
        a0 = ax[0, 2].imshow(l["com_phi"])

        ax[1, 0].imshow(l["com_chi_raw"])
        ax[1, 1].imshow(l["morph_chi"])
        a1 = ax[1, 2].imshow(l["com_chi"])

        ax[2, 0].imshow(l["mosaicity_raw"])
        ax[2, 1].imshow(l["morph_mos"][:, :, 2])
        a2 = ax[2, 2].imshow(l["mosaicity"])

        for i, im in enumerate([a0, a1, a2]):
            if im is not None:
                fig.colorbar(im, ax=ax[i, 2])

        plt.tight_layout()
        plt.show()


class H5PlotDarkField:

    def __init__(self, h5: H5File):
        self.h5 = h5
        self.inset = False
        self._min = None
        self._max = None
        self.pix_x = None
        self.pix_y = None
        self.iextent = None
        self.xlim = None
        self.ylim = None
        self.xlabel = None
        self.ylabel = None
        self.ixlabel = None
        self.iylabel = None
        self.cbar = False
        self.clabel = None
        self.cscale = None
        self.cori = "horizontal"
        self.cshrink = 1
        self.cmap = "viridis"
        self.cpad = 1
        self.origin = "lower"
        self.iloc = "upper right"
        self.folder = None
        self.name = None
        self.save = False
        self.format = "png"
        self.dpi = 300
        self.norm_ori = None
        self.ticks_off = None
        self.scale_bar = None
        self.figsize = (10, 12)


    def plot(self, dataset, layer, imap=None, dist=None):
        image = self.h5.read(dataset)[layer]

        fig, ax = plt.subplots()
        if self.pix_x is None and self.pix_y is None:
            extent = None
        else:
            extent = (0, image.shape[0]*self.pix_x, 0, image.shape[1]*self.pix_y)

        if self.cscale == "log":
            im = ax.imshow(image, cmap=self.cmap, norm=LogNorm(vmin=self._min, vmax=self._max),
                           extent=extent, origin=self.origin)
        else:
            im = ax.imshow(image, cmap=self.cmap, vmin=self._min, vmax=self._max,
                           extent=extent, origin=self.origin)

        if self.scale_bar is not None:
            scalebar = AnchoredSizeBar(transform=ax.transData, **self.scale_bar)
            ax.add_artist(scalebar)

        if self.inset:
            # axins = inset_axes(ax, width="35%", height="35%", loc=self.iloc, borderpad=0)
            w, h = 0.3, 0.3
            axins = ax.inset_axes([1 - w, 1 - h, w, h], transform=ax.transAxes)
            axins.set_anchor('NE')
            axins.tick_params(axis="both", direction="in", top=True, right=True)
            if imap is not None:
                axins.imshow(imap, extent=self.iextent, origin=self.origin)
                axins.set_xlabel(self.ixlabel)
                axins.set_ylabel(self.iylabel)

            if dist is not None:
                ori = dist[2] if self.norm_ori is None else self.norm_ori(dist[2])
                axins.contour(dist[0], dist[1], ori, cmap="jet", levels=10)

        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)

        if self.ticks_off:
            ax.axis('off')
        else:
            ax.tick_params(axis="both", direction="in", top=True, right=True)

        if self.cbar:
            cbar = fig.colorbar(im, label=self.clabel, orientation=self.cori, shrink=self.cshrink)
            cbar.ax.tick_params(axis="both", direction="in", left=True, right=True, top=True, bottom=True)
            cbar.ax.tick_params(which='minor', direction="in", left=True, right=True, top=True, bottom=True)
            # cbar.ax.minorticks_off()

        plt.tight_layout()
        if self.save:
            plt.savefig(self.folder+self.name, format=self.format, dpi=self.dpi,
                        bbox_inches="tight", pad_inches=0)
        else:
            plt.show()


    def plot_layer(self, com_mu=None, com_phi=None,
                   mosaicity=None, mosaicity_map=None,
                   ori_mu=None, ori_phi=None, ori_dist=None,
                   misorientation=None, gnd=None, strain=None, layer=0):

        fig, ax = plt.subplots(2, 3, dpi=300)

        if com_mu is not None:
            mu = self.h5.read(com_mu)[layer]
            imu = ax[0, 0].imshow(mu, cmap="viridis", vmin=0)
            ax[0, 0].set_title("CoM mu")
            fig.colorbar(imu, ax=ax[0, 0])

        if com_phi is not None:
            phi = self.h5.read(com_phi)[layer]
            iphi = ax[0, 1].imshow(phi, cmap="viridis", vmin=0)
            ax[0, 1].set_title("CoM phi")
            fig.colorbar(iphi, ax=ax[0, 1])

        if mosaicity is not None:
            mos = self.h5.read(mosaicity)[layer]
            axins = inset_axes(ax[0, 2], width="35%", height="35%", loc="lower right")

            imos = ax[0, 2].imshow(mos)
            ax[0, 2].set_title("Mosaicity")
            ins_extent = None

            if ori_dist is not None:
                ori = self.h5.read(ori_dist)[layer]
                ori_mu = self.h5.read(ori_mu)[layer]
                ori_phi = self.h5.read(ori_phi)[layer]
                axins.contour(ori_mu, ori_phi, ori, cmap="jet", levels=5)
                ins_extent = (np.nanmin(ori_mu), np.nanmax(ori_mu), np.nanmin(ori_phi), np.nanmax(ori_phi))

            if mosaicity_map is not None:
                _map = self.h5.read(mosaicity_map)[layer]
                axins.imshow(_map, extent=ins_extent)


        if misorientation is not None:
            mis = self.h5.read(misorientation)[layer]
            imis = ax[1, 0].imshow(mis, cmap="RdYlBu_r", vmin=0, vmax=3.5)
            ax[1, 0].set_title("Misorientation")
            fig.colorbar(imis, ax=ax[1, 0])

        if gnd is not None:
            gnd = self.h5.read(gnd)[layer]
            ignd = ax[1, 1].imshow(gnd, cmap="magma", norm=LogNorm(vmin=0.1, vmax=10))
            ax[1, 1].set_title("GND")
            fig.colorbar(ignd, ax=ax[1, 1])

        if strain is not None:
            strain = self.h5.read(strain)[layer]
            istr = ax[1, 2].imshow(strain, cmap="jet")
            ax[1, 2].set_title("Strain")
            fig.colorbar(istr, ax=ax[1, 2])

        for a in ax.flat:
            a.tick_params(axis="both", direction="in", top=True, right=True)

        plt.tight_layout()
        plt.show()


    def plot_tiles(self, dataset, rows=4, cols=4, imap=None, dist=None):
        image = self.h5.read(dataset)

        vmin = np.nanmin(image)
        vmax = np.nanmax(image)

        fig = plt.figure(figsize=self.figsize)

        left = 0.05
        right = 0.95
        bottom = 0.1
        top = 0.95

        wspace = 0.02
        hspace = 0.02

        tile_w = (right - left - (cols - 1) * wspace) / cols
        tile_h = (top - bottom - (rows - 1) * hspace) / rows

        axes = []

        for r in range(rows):
            for c in range(cols):
                x0 = left + c * (tile_w + wspace)
                y0 = top - (r + 1) * tile_h - r * hspace
                axes.append(fig.add_axes([x0, y0, tile_w, tile_h]))

        n_layers = image.shape[0]
        filled_rows = (n_layers + cols - 1) // cols
        last_row_filled = n_layers % cols

        im = None

        for i, ax in enumerate(axes):
            if i >= n_layers:
                ax.set_visible(False)
                continue

            row = i // cols
            col = i % cols

            if row == filled_rows - 1 and last_row_filled != 0:
                offset = (cols - last_row_filled) / 2
                col = col + offset

            x0 = left + col * (tile_w + wspace)
            y0 = top - (row + 1) * tile_h - row * hspace
            ax.set_position([x0, y0, tile_w, tile_h])

            imi = image[i]

            if self.pix_x is None or self.pix_y is None:
                extent = None
            else:
                try:
                    nx, ny = imi.shape
                except ValueError:
                    nx, ny, _ = imi.shape
                extent = (0, nx * self.pix_x, 0, ny * self.pix_y)

            if self.cscale == "log":
                im = ax.imshow(imi, cmap=self.cmap, norm=LogNorm(vmin=self._min, vmax=self._max),
                            extent=extent, origin=self.origin)
            else:
                im = ax.imshow(imi, cmap=self.cmap, vmin=self._min, vmax=self._max,
                            extent=extent, origin=self.origin)

            ax.set_title(f"{i + 1}th layer", size=10, pad=2)

            if self.ticks_off:
                ax.axis('off')
            else:
                ax.tick_params(axis="both", direction="in", top=True, right=True)

            if self.scale_bar is not None:
                scalebar = AnchoredSizeBar(transform=ax.transData, **self.scale_bar)
                ax.add_artist(scalebar)

            if self.inset:
                axins = inset_axes(ax, width="15%", height="15%", loc=self.iloc, borderpad=0)
                axins.tick_params(axis="both", direction="in", top=True, right=True, labelsize=8)
                if imap is not None:
                    axins.imshow(imap[i], extent=self.iextent, origin=self.origin)
                    axins.set_xlabel(self.ixlabel, size=8)
                    axins.set_ylabel(self.iylabel, size=8)

                if dist is not None:
                    ori = dist[2][i] if self.norm_ori is None else self.norm_ori(dist[2][i])
                    axins.contour(dist[0][i], dist[1][i], ori, cmap="jet", levels=10)

        if self.cbar:
            cbar_ax = fig.add_axes([0.25, 0.1, 0.5, 0.02])
            cbar_ax.set_xlabel(self.clabel, labelpad=1)
            fig.colorbar(im, cax=cbar_ax, orientation="horizontal", label= self.clabel, pad=self.cpad)

        plt.tight_layout()
        if self.save:
            plt.savefig(self.folder+self.name, format=self.format, dpi=self.dpi,
                        bbox_inches="tight", pad_inches=0)
        else:
            plt.show()


    def plot_colorbar(self, dataset):
        image = self.h5.read(dataset)
        import pylab as pl
        import matplotlib as mpl

        fig = plt.figure(figsize=(6, 1))
        cax = fig.add_axes([0.1, 0.45, 0.8, 0.2])

        if self.cscale == "log":
            norm = mpl.colors.LogNorm(vmin=self._min, vmax=self._max)
        else:
            norm = mpl.colors.Normalize(vmin=self._min, vmax=self._max)

        sm = mpl.cm.ScalarMappable(norm=norm, cmap=self.cmap)
        sm.set_array([])

        cbar = fig.colorbar(sm, cax=cax, orientation=self.cori)
        cbar.set_label(self.clabel)
        cbar.ax.tick_params(which="both", direction="in", right=True, top=True,
                            bottom=True, left=True)

        if self.cscale == "log":
            cbar.ax.xaxis.set_major_formatter(LogFormatterSciNotation())
        else:
            cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        if self.save:
            plt.savefig(self.folder+"df_cbar_"+self.name, format=self.format, dpi=self.dpi,
                        bbox_inches="tight", pad_inches=0)
        else:
            plt.show()


class H5EBSDPlot:

    def __init__(self, h5):
        self.h5 = h5
        self._min = None
        self._max = None
        self.saturate_max = True
        self.saturate_min = True
        self.xlabel = None
        self.ylabel = None
        self.xlim = None
        self.ylim = None
        self.origin = "lower"
        self.pix_x = None
        self.pix_y = None
        self.cbar = True
        self.clabel = None
        self.cscale = None
        self.linthresh = 1e-3
        self.cori = "vertical"
        self.cshrink = 0.8
        self.cmap = "viridis"
        self.cpad = 1
        self.title = None
        self.folder = None
        self.name = None
        self.save = False
        self.format = "png"
        self.dpi = 300
        self.ticks_off = None
        self.scale_bar = None


    def plot(self, dataset):
        image = self.h5.read(dataset)

        if self._min is None:
            self._min = np.nanmin(image)
        
        if self._max is None:
            self._max = np.nanmax(image)

        if self.saturate_max:
            image[image > self._max] = np.nan

        if self.saturate_min:
            image[image < self._min] = np.nan

        fig, ax = plt.subplots()

        if self.pix_x is None and self.pix_y is None:
            extent = None
        else:
            extent = (0, image.shape[1]*self.pix_x, 0, image.shape[0]*self.pix_y)

        if self.cscale == "log":
            norm = LogNorm(vmin=self._min, vmax=self._max)
        elif self.cscale == "symlog":
            norm = SymLogNorm(linthresh=self.linthresh, vmin=self._min, vmax=self._max,)
        else:
            norm = Normalize(vmin=self._min, vmax=self._max)

        im = ax.imshow(image, extent=extent, cmap=self.cmap, norm=norm)

        if self.ticks_off:
            ax.axis('off')
        else:
            ax.tick_params(axis="both", direction="in", top=True, right=True)

        if self.scale_bar is not None:
            scalebar = AnchoredSizeBar(transform=ax.transData, **self.scale_bar)
            ax.add_artist(scalebar)

        if self.cbar:
            cbar = fig.colorbar(im, label=self.clabel, orientation=self.cori, shrink=self.cshrink)
            cbar.ax.tick_params(axis="both", direction="in", left=True, right=True, top=True, bottom=True)
            cbar.ax.tick_params(which='minor', direction="in", left=True, right=True, top=True, bottom=True)
        

        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_title(self.title)

        plt.tight_layout()
        if self.save:
            plt.savefig(self.folder+self.name, format=self.format, dpi=self.dpi,
                        bbox_inches="tight", pad_inches=0)
            print(f"Saved {self.folder+self.name}")
        else:
            plt.show()


    def plot_image(self, dataset):
        image = self.h5.read(dataset)

        if self.pix_x is None and self.pix_y is None:
            extent = None
        else:
            extent = (0, image.shape[1]*self.pix_y, 0, image.shape[0]*self.pix_x)

        fig, ax = plt.subplots()
        ax.imshow(image, extent=extent, cmap=self.cmap)

        if self.ticks_off:
            ax.axis('off')
        else:
            ax.tick_params(axis="both", direction="in", top=True, right=True)

        if self.scale_bar is not None:
            scalebar = AnchoredSizeBar(transform=ax.transData, **self.scale_bar)
            ax.add_artist(scalebar)

        plt.tight_layout()
        if self.save:
            plt.savefig(self.folder+self.name, format=self.format, dpi=self.dpi,
                        bbox_inches="tight", pad_inches=0)
            print(f"Saved {self.folder+self.name}")
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
