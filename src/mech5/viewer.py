import os
import sys
sys.path.append('../../src/')

from typing import Union, List

import numpy as np
import matplotlib.pyplot as plt

from mech5.manager import H5File, SegmentedDatasetH5File
from mech5.util import Mask, Criterion, TrueMask


class H5Plot:

    def __init__(self, *h5file: Union[List[H5File], H5File, SegmentedDatasetH5File]) -> None:
        self.h5 = h5file
        self.dpi = 300
        self.format = "pdf"
        self.frontend = None
        self.x_label = None
        self.y_label = None
        self.z_label = None
        self.x_scale = None
        self.y_scale = None
        self.z_scale = None
        self.x_lim = None
        self.y_lim = None
        self.z_lim = None
        self.bins = None
        self.density = None
        self.forced_labels = []
        self.save = False
        self.plot_name = "Untitled"


    def query_datasets(self, mask, criterion) -> np.ndarray:
        return None


    def histogram(self, path, protocol):
        ...
        plt.tight_layout()
        if self.save:
            ...
        else:
            plt.show()


    def scatter_2d(self):
        ...
        plt.tight_layout()
        if self.save:
            ...
        else:
            plt.show()


    def scatter_3d(self):
        ...
        plt.tight_layout()
        if self.save:
            ...
        else:
            plt.show()