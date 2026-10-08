import sys
sys.path.append('../../src/')
from typing import List

import numpy as np
from scipy import ndimage

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import skimage
from skimage import measure
from skimage.transform import rescale
from skimage.morphology import remove_small_holes

from mech5.manager import H5File
from mech5.util import normalise
from mech5.util import cartesian2polar as c2p

class DarkFieldProcessor(H5File):

    def __init__(self, filename, mode, overwrite = False, pix_x=1., pix_y=1.):
        super().__init__(filename, mode, overwrite)
    # def __init__(self, pix_x=1., pix_y=1.):
        self.pix_x = pix_x
        self.pix_y = pix_y


    def create_mask(self, image, min_size_to_close, replaced_value=0, show=False, criterion=np.isnan, **cargs):
        mask = criterion(image, **cargs)
        closed_mask = ~remove_small_holes(mask, min_size_to_close)
        masked_data = np.where(closed_mask, image, replaced_value)

        if show:
            fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
            ax[0].imshow(image)
            ax[1].imshow(closed_mask)
            ax[2].imshow(masked_data)
            plt.show()

        return masked_data, closed_mask


    def apply_mask(self, image, mask, replaced_value=np.nan, show=False):
        if image.ndim > mask.ndim:
            mask = mask[..., None]

        masked = np.where(mask, image, replaced_value)

        if show:
            fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
            ax[0].imshow(image)
            ax[1].imshow(mask)
            ax[2].imshow(masked)
            plt.show()

        return masked


    def normalise_to_zero(self, image, show=False):
        """Zeroing motor coordinates."""
        zeroed = image - np.nanmin(image)

        if show:
            fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
            im = ax[0].imshow(image)
            ze = ax[1].imshow(zeroed)
            fig.colorbar(im)
            fig.colorbar(ze)
            plt.show()

        return zeroed


    @staticmethod
    def gradient_module(x_dx, x_dy, y_dx, y_dy):
        return np.sqrt(x_dx**2 + x_dy**2 + y_dx**2 + y_dy**2)


    def misorientation(self, image_x, image_y, axis=None, show=False):
        x_dx, x_dy = np.gradient(image_x, self.pix_x, self.pix_y, axis=axis)
        y_dx, y_dy = np.gradient(image_y, self.pix_x, self.pix_y, axis=axis)
        mis = self.gradient_module(x_dx, x_dy, y_dx, y_dy)

        if show:
            fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
            ax[0].imshow(image_x)
            ax[1].imshow(image_y)
            ax[2].imshow(mis)
            plt.show()

        return mis


    def gnd(self, image, b, thk, show=False):
        gnd = np.deg2rad(image)/(thk*b)/1e14

        if show:
            fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
            ax[0].imshow(image)
            ax[1].imshow(gnd)
            plt.show()

        return gnd


    def strain(self, image, ref=None, show=False):
        if ref is None:
            ref = np.deg2rad(np.nanmedian(image))

        image = np.deg2rad(image)
        strain = (image - ref) / np.tan(ref)

        if show:
            fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
            im = ax[0].imshow(image)
            st = ax[1].imshow(strain)

            fig.colorbar(im)
            fig.colorbar(st)
            plt.show()

        return strain


    @staticmethod
    def saturate_count(N, fn=np.log10):
        N = np.where(N == 0, np.nan, N)
        N_valid = N[~np.isnan(N)]
        N_clipped = np.clip(N, np.percentile(N_valid, 2), np.percentile(N_valid, 98))

        if fn is not None:
            return fn(N_clipped)
        else:
            return N_clipped


    def ori_dist(self, arr_x, arr_y, dist, levels=10, show=None):
        X, Y = np.meshgrid(arr_x, arr_y)
        dist_sat = self.saturate_count(dist, None)

        if dist is not None and show:
            fig, ax = plt.subplots()
            ax.contour(X, Y, dist_sat, levels=levels, cmap="jet")
            plt.show()

        return X, Y, dist_sat


    @staticmethod
    def percentiles(x):
        # x = x[~np.isnan(x)]
        x_min = np.nanpercentile(x, 2)
        x_max = np.nanpercentile(x, 98)
        return x_min, x_max


    @staticmethod
    def cartesian2polar(x, y, x0, y0):
        radius = np.sqrt((x - x0)**2 + (y - y0)**2)
        angle = np.arctan2(y - y0, x - x0) + np.pi # chi = y, phi = x
        return radius, angle


    @staticmethod
    def to_rgb(x, x_min, x_max, y, y_min, y_max):
        from matplotlib.colors import hsv_to_rgb
        H = np.clip(normalise(x, x_min, x_max), 0, 1)
        S = np.clip(normalise(y, y_min, y_max), 0, 1)
        V = np.ones_like(H)
        HSV = np.stack([H, S, V], axis=-1)
        return hsv_to_rgb(HSV)


    def mosaicity(self, image_x, image_y, x_min=None, x_max=None, y_min=None, y_max=None, N=100, show=False):

        if x_min is None and x_max is None and y_min is None and y_max is None:
            x_min, x_max = self.percentiles(image_x)
            y_min, y_max = self.percentiles(image_y)

        x0 = 0.5 * (x_min + x_max)
        y0 = 0.5 * (y_min + y_max)

        # mosaicity map of the image
        radius, angle = self.cartesian2polar(image_x, image_y, x0, y0)

        # mosaicity colorbar
        x_mesh = np.linspace(x_min, x_max, N)
        y_mesh = np.linspace(y_min, y_max, N)
        X, Y = np.meshgrid(x_mesh, y_mesh)
        radius_mesh, angle_mesh = self.cartesian2polar(X, Y, x0, y0)

        radius_max = max(np.nanmax(radius), np.nanmax(radius_mesh))
        rgb = self.to_rgb(angle, 0., 2*np.pi, radius, 0., np.nanmax(radius_mesh))
        rgb_mesh = self.to_rgb(angle_mesh, 0., 2*np.pi, radius_mesh, 0., radius_max)

        if show:
            fig, ax = plt.subplots(2, 2)
            ax[0, 0].imshow(rgb)
            ax[0, 1].imshow(rgb_mesh, origin="lower", extent=(x_min, x_max, y_min, y_max))

            x = ax[1, 0].imshow(image_x)
            y = ax[1, 1].imshow(image_y)

            fig.colorbar(x)
            fig.colorbar(y)
            plt.show()

        return rgb, rgb_mesh


    def load_layers(self):
        self.layers = self.read("/dfxm/common/layers")


    def load_pixel_size(self, attr, pix):
        setattr(self, attr, pix)


    def write_com(self, dataset_in, dataset_out, norm_to_zero=True):
        stack = self.read(dataset_in)
        com_stack = []

        for s in stack:
            if norm_to_zero:
                print(f"Write {dataset_out} - Normalised")
                com_stack.append(self.normalise_to_zero(s))
            else:
                print(f"Write {dataset_out} - Not-normalised")
                com_stack.append(s)

        self.write(dataset_out, np.asarray(com_stack))


    def write_mask(self, dataset_in, dataset_out, mask_out, min_size_to_close, replaced_value=0., criterion=np.isnan, **cargs):
        stack = self.read(dataset_in)
        data_stack = []
        mask_stack = []

        if not isinstance(min_size_to_close, list):
            min_size_to_close = [min_size_to_close]*stack.shape[0]

        for s, m in zip(stack, min_size_to_close):
            print(f"Masking {dataset_in} - Size {m}")
            md, cm = self.create_mask(s, m, replaced_value, False, criterion, **cargs)
            data_stack.append(md)
            mask_stack.append(cm)


        self.write(dataset_out, np.asarray(data_stack))

        if mask_out is not None:
            self.write(mask_out, np.asarray(mask_stack))


    def write_masked_array(self, dataset_in, mask_in, dataset_out, replaced_value=np.nan):
        image = self.read(dataset_in)
        mask = self.read(mask_in)

        masked = self.apply_mask(image, mask, replaced_value)
        self.write(dataset_out, masked)


    def write_misorientation(self, dataset_x, dataset_y, dataset_out, axis=None):
        image_x = self.read(dataset_x)
        image_y = self.read(dataset_y)
        print(f"Write {dataset_out}")
        self.write(dataset_out, self.misorientation(image_x, image_y, axis))


    def write_gnd(self, dataset_in, dataset_out, b, thk):
        stack = self.read(dataset_in)
        print(f"Write {dataset_out}")
        self.write(dataset_out, self.gnd(stack, b, thk))


    def write_mosaicity(self, dataset_x, dataset_y, dataset_out, x_min=None, x_max=None, y_min=None, y_max=None):
        stack_x = self.read(dataset_x)
        stack_y = self.read(dataset_y)
        rgb_stack = []
        rgb_map_stack = []

        for x, y in zip(stack_x, stack_y):
            rgb, rgb_map = self.mosaicity(x, y, x_min, x_max, y_min, y_max)
            rgb_stack.append(rgb)
            rgb_map_stack.append(rgb_map)

        print(f"Write {dataset_out}")
        self.write(dataset_out, np.asarray(rgb_stack))
        self.write(dataset_out + "_map", np.asarray(rgb_map_stack))


    def write_ori_dist(self, dataset_in_x, dataset_in_y, dataset_in_dist,
                       dataset_out_x, dataset_out_y, dataset_out_dist, norm=True):

        stack_x = self.read(dataset_in_x)
        stack_y = self.read(dataset_in_y)
        dist = self.read(dataset_in_dist)

        if norm:
            stack_x = self.normalise_to_zero(stack_x)
            stack_y = self.normalise_to_zero(stack_y)

        X_stack = []
        Y_stack = []
        D_stack = []

        for x, y, d, in zip(stack_x, stack_y, dist):
            X, Y, D = self.ori_dist(x, y, d)
            X_stack.append(X)
            Y_stack.append(Y)
            D_stack.append(D)

        print("Write orientation distribution")
        self.write(dataset_out_x, np.asarray(X_stack))
        self.write(dataset_out_y, np.asarray(Y_stack))
        self.write(dataset_out_dist, np.asarray(D_stack))


    def write_strain(self, dataset_in, dataset_out, ref=None):
        stack = self.read(dataset_in)
        print(f"Write {dataset_out}")
        self.write(dataset_out, self.strain(stack, ref))


class Image2Profile(H5File):

    def __init__(self, filename, mode, overwrite = False):
        super().__init__(filename, mode, overwrite)
        self.data = None
        self.nan_mask = None
        self.selected_mask = None
        self.selected_mask_closed = None
        self.boundary = None
        self.profile = None
        self.C = None


    def map_2_mask(self, dataset: str):
        self.data = self.read(dataset)
        self.nan_mask = np.isnan(self.data)

        self.labeled, self.n_regions = ndimage.label(self.nan_mask)

        if self.n_regions == 0:
            raise ValueError("No mask were found.")

        sizes = ndimage.sum(self.nan_mask, self.labeled, range(1, self.n_regions + 1))
        self.order = np.argsort(sizes)[::-1]
        self.sizes_sorted = sizes[self.order]
        self.labels_sorted = np.arange(1, self.n_regions + 1)[self.order]

        print(f"Regions: {self.n_regions}")
        # print(f"Sizes: {len(self.sizes_sorted)}")


    def select_mask(self, idx=0):
        if self.n_regions == 0:
            raise ValueError("No regions available.")

        if idx is None:
            idx = 0

        if idx < 0 or idx >= self.n_regions:
            raise IndexError(f"idx {idx} out of range [0, {self.n_regions - 1}]")

        label = self.labels_sorted[idx]
        self.selected_mask = (self.labeled == label)

        print(f"Selected region {idx} -> label {label} " f"with size {int(self.sizes_sorted[idx])}")


    def close_selected_mask(self, iterations: int=2):
        if self.selected_mask is None:
            raise ValueError("No selected mask available.")

        closed = ndimage.binary_closing(self.selected_mask, iterations=iterations)
        self.selected_mask_closed = ndimage.binary_fill_holes(closed)
        self.selected_mask = self.selected_mask_closed


    def contours_selected_mask(self, level=0.5):
        if self.selected_mask is None:
            raise ValueError("No selected mask available.")

        contours = measure.find_contours(self.selected_mask.astype(float), level=level)

        if len(contours) == 0:
            raise ValueError("No contours found for selected mask.")

        self.contours = contours
        self.boundary = max(contours, key=len)


    def inspect_mask(self):
        fig, ax = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False)
        if self.data is not None:
            ax[0, 0].imshow(self.data)
        if self.nan_mask is not None:
            ax[0, 1].imshow(self.nan_mask)
        if self.select_mask is not None:
            ax[1, 0].imshow(self.selected_mask)
        if self.selected_mask_closed is not None:
            ax[1, 1].imshow(self.selected_mask_closed)
        if self.boundary is not None:
            ax[0, 0].plot(self.boundary[:, 1], self.boundary[:, 0], color='red', linewidth=1.8)
        plt.show()


    def get_profile(self, window=1):
        if self.boundary is None:
            raise ValueError("No contour available.")

        coords = np.round(self.boundary).astype(int)
        h, w = self.data.shape
        profile = []

        half = window // 2
        for y, x in coords:
            y0 = max(y - half, 0)
            y1 = min(y + half + (window % 2), h)
            x0 = max(x - half, 0)
            x1 = min(x + half + (window % 2), w)

            patch = self.data[y0:y1, x0:x1]
            profile.append(np.nanmean(patch))

        self.profile = np.array(profile)


    def cartesian2polar(self, phase=0., pix_x=1., pix_y=1., show=False):
        if self.boundary is None:
            raise ValueError("No boundary available.")

        y = -self.boundary[:, 0] * pix_y # chance convention image -> points
        x = self.boundary[:, 1] * pix_x

        if self.C is None:
            self.C = np.array([x.mean(), y.mean()])
        print(f"Centroid is {self.C}")

        r, theta, order = c2p(x, y, C=self.C, phase=phase)

        self.r = r[order]
        self.theta = theta[order]
        if self.profile is not None:
            self.profile = self.profile[order]
        if show:
            fig, ax = plt.subplots(figsize=(6, 6))

            ax.scatter(x, y, s=15, label="boundary points")
            ax.scatter(*self.C, color="red", marker="x", s=100, label="centroid")

            r_max = self.r.max()
            phase_rad = np.deg2rad(phase)
            ax.plot([self.C[0], self.C[0] + r_max * np.cos(phase_rad)],
                    [self.C[1], self.C[1] + r_max * np.sin(phase_rad)],
                    color="k", linestyle="--", label=f"phase = {phase}°")

            ax.set_aspect("equal")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            plt.show()


    def theta_profile(self, show=False):
        if show:
            fig, ax = plt.subplots()
            ax.plot(self.theta, self.r)
            plt.show()
        return self.theta, self.profile


class TomoMask3DXRD:

    def __init__(self, tomo, pix_tomo, scan, pix_scan):
        self.pix_tomo = pix_tomo
        self.pix_scan = pix_scan
        self.ratio = self.pix_tomo / self.pix_scan

        self.tomo_orig = tomo
        self.scan_orig = scan
        self.ht_orig, self.wt_orig = self.tomo_orig.shape
        self.hs_orig, self.ws_orig = self.scan_orig.shape

        self.tomo = self.tomo_orig
        self.scan = self.scan_orig
        self.ht, self.wt = self.tomo.shape
        self.hs, self.ws = self.scan.shape

        self.mask = None
        self.mask_path = None

        print(f"pix_tomo = {self.pix_tomo}")
        print(f"pix_scan = {self.pix_scan}")
        print(f"ratio = {self.ratio}")
        print(f"tomo shape = {self.tomo_orig.shape}")
        print(f"scan shape = {self.scan_orig.shape}")


    def plot_orig(self) -> None:
        """Plot original tomographic and 3DXRD scans"""
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(self.tomo_orig, cmap="gray", extent=(0, self.wt_orig * self.pix_tomo, self.ht_orig * self.pix_tomo, 0))
        ax[0].set_title(f"Tomo ({self.ht_orig} X {self.wt_orig} px)")
        ax[1].imshow(self.scan_orig, cmap="viridis", extent=(0, self.ws_orig * self.pix_scan, self.hs_orig * self.pix_scan, 0))
        ax[1].set_title(f"Scan ({self.hs_orig} X {self.ws_orig} px)")
        plt.show()


    def plot(self) -> None:
        """Plot current state of tomographic and 3DXRD scans"""
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(self.tomo, cmap="gray", extent=(0, self.wt * self.pix_tomo, self.ht * self.pix_tomo, 0))
        ax[0].set_title(f"Tomo ({self.ht} X {self.wt} px)")
        ax[1].imshow(self.scan, cmap="viridis", extent=(0, self.ws * self.pix_scan, self.hs * self.pix_scan, 0))
        ax[1].set_title(f"Scan ({self.hs} X {self.ws} px)")
        plt.show()


    def register_tomo(self, scale: bool=False, order:int =1, transpose: bool=False, flip: bool=False, k: bool=0) -> None:
        """
        Register the tomography image by applying scaling, transposition,
        horizontal flipping, and rotation.

        Parameters
        ----------
        scale : bool, default=False
            If True, rescale the original tomography image using ``self.ratio``.
        order : int, default=1
            Interpolation order used for rescaling. ``0`` uses nearest-neighbour
            interpolation and preserves the original data type; higher orders
            produce a ``float32`` output.
        transpose : bool, default=False
            If True, transpose the tomography image.
        flip : bool, default=False
            If True, flip the tomography image horizontally.
        k : int, default=0
            Number of 90-degree counter-clockwise rotations to apply. Only
            ``k % 4`` is relevant.

        Returns
        -------
        None
            The registered tomography is stored in ``self.tomo``. Its height and
            width are stored in ``self.ht`` and ``self.wt``, respectively.
        """
        out = self.tomo_orig
        if scale:
            print(f"Scaling tomo: ratio={self.ratio}, order={order}")
            out = rescale(self.tomo_orig, self.ratio, order=order, preserve_range=True, anti_aliasing=(self.ratio < 1))
            out = out.astype(self.tomo_orig.dtype if order == 0 else np.float32)
        if transpose:
            print("Transposing tomo")
            out = out.T
        if flip:
            print("Flipping tomo horizontally")
            out = out[:, ::-1]
        if k % 4:
            print(f"Rotating tomo: k={k}")
            out = np.rot90(out, k)
        tomo = np.ascontiguousarray(out)
        print(f"shape: {tomo.shape}")
        self.tomo = tomo
        self.ht, self.wt = self.tomo.shape


    def roi(self, center=None, offset: List=[0,0], sigma=None, threshold: callable=skimage.filters.threshold_li, show=True):
        """
        Extract an ROI from the registered tomography and generate a binary mask.

        The ROI is centred on the tomography by default, or at a user-defined
        position with an optional offset. A Gaussian filter can be applied before
        extracting the ROI. The ROI is then thresholded to generate a binary
        mask, which is stored in ``self.mask``.

        Parameters
        ----------
        center : tuple or list of int, optional
            ROI centre as ``(cy, cx)`` in ``(row, column)`` coordinates. If None,
            the centre of ``self.tomo`` is used.
        offset : list of int, default=[0, 0]
            Offset applied to the ROI centre as ``[dy, dx]``.
        sigma : float, optional
            Standard deviation of the Gaussian filter applied to the tomography
            before ROI extraction. If None, no filtering is applied.
        threshold : callable, default=skimage.filters.threshold_li
            Thresholding function applied to the finite ROI values. The function
            must accept a NumPy array and return a scalar threshold value.
        show : bool, default=True
            If True, display the registered tomography, extracted ROI, masks, and
            Dice coefficient.

        Returns
        -------
        numpy.ndarray
            Boolean ROI mask with shape ``(self.hs, self.ws)``. Pixels with values
            above the computed threshold are True, while all other pixels are
            False. The mask is also stored in ``self.mask``.
        """
        # get tomo
        tomo = self.tomo

        # roi edges
        h_roi, w_roi = self.hs, self.ws

        # roi centre
        if center is None:
            cy, cx = tomo.shape[0] // 2, tomo.shape[1] // 2
        else:
            cy, cx = center

        cy += offset[0]
        cx += offset[1]
        print(f"ROI centre {cx, cy}")

        y0 = round(cy - h_roi // 2)
        x0 = round(cx - w_roi // 2)

        y1 = y0 + h_roi
        x1 = x0 + w_roi

        # blur tomo if needed with gaussian kernel
        if sigma is not None:
            print(f"Gaussian filter with {sigma} sigma")
            tomo = skimage.filters.gaussian(tomo, sigma=sigma, preserve_range=True)

        # Extract ROI, keeping exactly the scan dimensions
        core = np.zeros((h_roi, w_roi), dtype=tomo.dtype)

        ys, xs = max(y0, 0), max(x0, 0)
        ye, xe = min(y1, tomo.shape[0]), min(x1, tomo.shape[1])

        if ye > ys and xe > xs:
            core[ys - y0:ye - y0, xs - x0:xe - x0] = tomo[ys:ye, xs:xe]

        # threshold
        print(f"Thesholding with {threshold.__name__}")
        thr = threshold(core[np.isfinite(core)])
        roi_mask = np.isfinite(core) & (core > thr)
        scan_mask = np.isfinite(self.scan) & (self.scan != 0)
        inter = (roi_mask & scan_mask).sum()
        dice = 2 * inter / (roi_mask.sum() + scan_mask.sum() + 1e-9)
        # masked_scan = np.where(roi_mask, self.scan, np.nan)

        if show:
            fig, ax = plt.subplots(2, 2, figsize=(8, 8))
            ax[0, 0].imshow(tomo, cmap="gray")
            ax[0, 0].add_patch(Rectangle((x0, y0), w_roi, h_roi, fill=False, ec="r", lw=2))
            ax[0, 0].set_title(f"Registered Tomo {tomo.shape}\nROI {h_roi} X {w_roi} px")

            ax[0, 1].set_title(f"ROI {core.shape}")
            ax[0, 1].imshow(core, cmap="grey")

            ax[1, 0].set_title(f"Scan {self.scan.shape}")
            ax[1, 0].imshow(self.scan, cmap="inferno")
            ax[1, 0].contour(roi_mask, linewidth=1, cmap="viridis")

            masked_scan = np.where(roi_mask & scan_mask, self.scan, np.nan)
            ax[1, 1].imshow(masked_scan, cmap="viridis")
            ax[1, 1].set_title(f"DICE = {dice:.2f}")

            inset = ax[1, 1].inset_axes([0.72, 0.72, 0.22, 0.22])
            inset.imshow(roi_mask, cmap="gray")
            inset.axis("off")

            plt.tight_layout()
            plt.show()

        self.mask = roi_mask
        return self.mask


    def to_array(self):
        """Save mask to file."""
        print(self.mask)
        print(f"Mask saved to {self.mask_path}")
        np.save(self.mask_path, self.mask)
