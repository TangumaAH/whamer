"""
utils.py
File management utilities and GROMACS XVG format handler.

Author: Alejandro H. Tanguma
Version: 0.1.0
"""

import os
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


# Header written to every XVG file produced by Whamer
_WHAMER_COMMENT: str = (
    "# Computed by Whamer\n"
    "# visit: https://github.com/TangumaAH/whamer\n"
)

# Default physical units
_DEFAULT_POSITION_UNITS: str = "nm"
_DEFAULT_ENERGY_UNITS: str = "kJ/mol"


@dataclass
class FileManager:
    """Handles basic file operations and existence checks."""
    name: str
    read: bool = True

    def __post_init__(self):
        if self.read and not os.path.isfile(self.name):
            raise FileNotFoundError(f"File '{self.name}' does not exist.")

    def find_line(self, tag: str) -> Optional[str]:
        """Returns the first line in the file containing the specified tag."""
        with open(self.name, 'r') as f:
            for line in f:
                if tag in line:
                    return line
        return None

    def read_lines(
            self,
            remove_empty: bool = True,
            remove_start: List[str] = [],
    ) -> List[str]:
        """Reads all lines, optionally filtering empty lines or lines starting
        with any of the characters in remove_start."""
        with open(self.name, 'r') as f:
            lines = f.readlines()

        if remove_empty:
            lines = [line.strip() for line in lines if line.strip()]

        if remove_start:
            lines = [
                line for line in lines
                if not any(line.startswith(s) for s in remove_start)
            ]

        return lines

    def write_file(self, lines: List[str], eol: str = ''):
        """Writes a list of strings to the file."""
        with open(self.name, 'w') as f:
            for line in lines:
                f.write(line + eol)


@dataclass
class XVG(FileManager):
    """Handler for GROMACS .xvg file format."""

    def read_xvg(
            self,
            column: int = 1,
            verbose: bool = False,
    ) -> Tuple[np.ndarray, str, np.ndarray, str]:
        """
        Reads an XVG file and returns the X and Y data along with axis labels.

        Parameters:
            column: Index of the Y column to read (default 1, i.e. second column).
            verbose: If True, prints the filename being read.

        Returns:
            xdata: 1D array with X values.
            xaxis: Label of the X axis.
            ydata: 1D array with Y values.
            yaxis: Label of the Y axis.
        """
        if verbose:
            print(f"Reading file: {self.name}")

        lines = self.read_lines(remove_empty=True, remove_start=['#'])
        xaxis, yaxis = '', ''
        xdata, ydata = [], []

        for line in lines:
            if line.startswith('@'):
                if 'xaxis label' in line:
                    xaxis = line.split('"')[1]
                elif 'yaxis label' in line:
                    yaxis = line.split('"')[1]
                continue

            parts = line.split()
            if len(parts) > column:
                xdata.append(float(parts[0]))
                ydata.append(float(parts[column]))

        return np.array(xdata), xaxis, np.array(ydata), yaxis

    def write_xvg(
            self,
            *arrays: np.ndarray,
            x_units: str = _DEFAULT_POSITION_UNITS,
            y_units: str = _DEFAULT_ENERGY_UNITS,
    ):
        """
        Writes multiple NumPy arrays into a multi-column XVG file.

        Rows containing any NaN value are silently skipped.
        The file header includes a Whamer attribution comment and XVG-style
        axis labels derived from x_units and y_units.

        Parameters:
            *arrays: NumPy arrays to write as columns (first = X axis).
            x_units: Physical unit label for the X axis (default: 'nm').
            y_units: Physical unit label for the Y axis (default: 'kJ/mol').
        """
        # Build XVG header
        header_lines = (
            _WHAMER_COMMENT
            + f'@ xaxis label "{x_units}"\n'
            + f'@ yaxis label "{y_units}"\n'
        )

        data_lines = []
        for row in zip(*arrays):
            if any(np.isnan(val) for val in row):
                continue
            data_lines.append("\t".join(f"{val:.6f}" for val in row))

        with open(self.name, 'w') as f:
            f.write(header_lines)
            for line in data_lines:
                f.write(line + '\n')
