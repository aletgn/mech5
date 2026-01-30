# MECH5: MEChanical H5

This package provides utilities to manage `H5` datasets for collating data from various types of mechanical characterisation methods.

## Architecture

The package is structured as follows:

- `manager.py`: Contains classes for managing access (read/write/query) to generic `H5` files. Subclasses are tailored for specific datasets.  
- `interface.py`: Includes classes to convert spreadsheet files into `H5` files, whether generic or specific. The module also include classes to convert datasets in `H5` files into VTK for visualisation purposes.
- `view.py`: Collates some utilities to plot datasets in `H5` files, e.g.,  histograms, and scatter plots. 

## Features

`MECH5` currently supports:

- I/O for generic `H5` files.  
- I/O for computed tomography (CT) `H5` files.  
- Conversion from spreadsheets into the above files.
