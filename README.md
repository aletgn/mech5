# MECH5: MECHanical H5

This package provides utilities to manage `H5` datasets for collating data from various types of mechanical characterisation methods.

## Architecture

The package is structured as follows:

- `manager.py`: Contains classes for managing access (read/write/query) to generic `H5` files. Subclasses are tailored for specific datasets.  
- `interface.py`: Includes classes to convert spreadsheet files into `H5` files, whether generic or specific.

## Features

`MECH5` currently supports:

- I/O for generic `H5` files.  
- I/O for computed tomography (CT) `H5` files.  
- Conversion from spreadsheets into the above files.
