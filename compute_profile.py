# ---------------------------------------------------------
# DISCLAIMER: This code has been typed and commented by Gemini.
# ---------------------------------------------------------

"""
compute_profile.py
Script to process simulation data and compute PMF profiles using WHAM.

Author: Alejandro H. Tanguma
Version: 0.1.0
"""

import os
import argparse
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union, Any
from scipy.signal import argrelextrema
from scipy.stats import norm
from scipy.optimize import curve_fit
from whamer import wham


def get_args() -> argparse.Namespace:
    """
    Defines and parses command-line arguments for PMF profile computation.
    """
    parser = argparse.ArgumentParser(
        description='Process trajectory data to compute and plot PMF profiles using WHAM.'
    )

    # --- WHAM algorithm options ---
    wham_group = parser.add_argument_group('WHAM Options')
    wham_group.add_argument('-il', '--log', type=str, required=True,
                            help='Path to a text file containing the list of all .log files (to read init positions and k).')
    wham_group.add_argument('-ix', '--pullx', type=str, required=True,
                            help='Path to a text file containing the list of all pullx.xvg files (trajectory data).')
    wham_group.add_argument('-t', '--temperature', type=float, default=300.0,
                            help='Simulation temperature in Kelvin (default: 300).')
    wham_group.add_argument('-kb', '--boltzmann', type=float, default=0.0083144621,
                            help='Boltzmann constant value (default in kJ/mol*K: 0.0083144621).')
    wham_group.add_argument('-nb', '--bins', type=int, default=100,
                            help='Number of bins to discretize the reaction coordinate (default: 100).')
    wham_group.add_argument('-wi', '--maxiter', type=int, default=1000000,
                            help='Maximum number of iterations for the WHAM solver (default: 1,000,000).')
    wham_group.add_argument('-wt', '--tol', type=float, default=1.0e-07,
                            help='Convergence tolerance for free energy iterations (default: 1e-07).')

    # --- Profile output and formatting options ---
    prof_group = parser.add_argument_group('Profile Options')
    prof_group.add_argument('-pro', '--profile', type=str, default='profile.xvg',
                            help='Output filename for the computed PMF profile (default: profile.xvg).')
    prof_group.add_argument('-his', '--histo', type=str, default='histo.xvg',
                            help='Output filename for the reconstructed histograms (default: histo.xvg).')
    prof_group.add_argument('-ex', '--extrema', type=float, default=None,
                            help='Position to crop the profile (removes data beyond this value).')
    prof_group.add_argument('-sm', '--set_min', action='store_true',
                            help='Shift the entire PMF profile so the global minimum is at 0.')
    prof_group.add_argument('-hpoints', '--histo_points', type=int, default=None,
                            help='Number of points to interpolate the output histograms.')

    # --- Free Energy (Delta G) analysis options ---
    dg_group = parser.add_argument_group('Analysis Options')
    dg_group.add_argument('-dg', '--show_dg', action='store_true',
                          help='Calculate and display the free energy difference (Delta G).')
    dg_group.add_argument('-max', '--max', type=float, default=None,
                          help='Specific x-position to define the maximum state for Delta G calculation.')
    dg_group.add_argument('-minf', '--min_from', type=float, default=None,
                          help='Lower bound to search for the local minimum.')
    dg_group.add_argument('-minb', '--min_before', type=float, default=None,
                          help='Upper bound to search for the local minimum.')

    return parser.parse_args()


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
    
    def read_lines(self, remove_empty: bool = True, remove_start: List[str] = []) -> List[str]:
        """Reads all lines, optionally filtering empty lines or specific starting characters."""
        with open(self.name, 'r') as f:
            lines = f.readlines()
        
        if remove_empty:
            lines = [line.strip() for line in lines if line.strip()]
        
        if remove_start:
            lines = [line for line in lines if not any(line.startswith(s) for s in remove_start)]
            
        return lines
    
    def write_file(self, lines: List[str], eol: str = ''):
        """Writes a list of strings to the file."""
        with open(self.name, 'w') as f:
            for line in lines:   
                f.write(line + eol)


@dataclass
class XVG(FileManager):
    """Specific handler for GROMACS .xvg file formats."""
    def read_xvg(self, column: int = 1, verbose: bool = False) -> Tuple[np.ndarray, str, np.ndarray, str]:
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
    
    def write_xvg(self, *arrays: np.ndarray):
        """Writes multiple NumPy arrays into a multi-column XVG file."""
        lines = []
        for row in zip(*arrays):
            if any(np.isnan(val) for val in row):
                continue
            lines.append("\t".join(f"{val:.6f}" for val in row))
        self.write_file(lines, eol='\n')


@dataclass
class Profile:
    """Represents a PMF profile and provides analysis methods."""
    bins: np.ndarray
    pmf: np.ndarray
    max_pos: Optional[float] = None
    min_left: Optional[float] = None
    min_right: Optional[float] = None

    def remove_extrema(self, max_lim: Optional[float] = None):
        """Crops the profile data up to a specified maximum x-limit."""
        if max_lim is not None:
            idx = np.argmin(np.abs(self.bins - max_lim))
            self.bins = self.bins[:idx]
            self.pmf = self.pmf[:idx]

    @property
    def dg(self) -> float:
        """Calculates Free Energy Difference (Delta G)."""
        return round(float(self.max_val - self.min_val), 4)
    
    @property
    def min_idx(self) -> int:
        """Finds the index of the minimum value within specified bounds."""
        data = self.pmf
        left_offset = 0
        
        # Apply bounds if provided
        if self.min_left is not None:
            left_offset = np.argmin(np.abs(self.bins - self.min_left))
            data = data[left_offset:]
        if self.min_right is not None:
            right_idx = np.argmin(np.abs(self.bins - self.min_right))
            data = data[:right_idx - left_offset]
        
        try:
            # Use local extrema detection
            idx_min = argrelextrema(data, np.less, order=5)[0][0]
        except (IndexError, ValueError):
            # Fallback to absolute minimum
            idx_min = np.nanargmin(data)
            
        return left_offset + idx_min
    
    @property
    def min_val(self) -> float:
        return self.pmf[self.min_idx]
    
    @property
    def max_val(self) -> float:
        """Finds the maximum value, either at a specific position or globally."""
        if self.max_pos is not None:
            idx_max = np.argmin(np.abs(self.bins - self.max_pos))
            return self.pmf[idx_max]
        return np.nanmax(self.pmf)

    @property
    def x(self) -> np.ndarray: return self.bins
    
    @property
    def y(self) -> np.ndarray: return self.pmf


@dataclass
class Histo:
    """Represents a single Gaussian histogram distribution."""
    mu: float
    std: float
    h: Optional[float] = 1.0

    def map_histogram(self, x_array: np.ndarray) -> np.ndarray:
        """Generates Y values for the Gaussian curve over x_array."""
        height = self.h if self.h is not None else 1.0
        return norm.pdf(x_array, self.mu, self.std) * height


@dataclass
class HistoArray:
    """Manages a collection of histograms."""
    mu_list: List[float]
    std_list: List[float]
    h_list: Optional[List[float]] = None
    x: Optional[np.ndarray] = None
    maps: List[np.ndarray] = field(default_factory=list)

    def __post_init__(self):
        h_vals = self.h_list if self.h_list else [1.0] * len(self.mu_list)
        self.histo_list = [Histo(mu, std, h) for mu, std, h in zip(self.mu_list, self.std_list, h_vals)]
        
    def map_all(self, points: int = 100, x_array: Optional[np.ndarray] = None, redefine_x: bool = False):
        """Maps all individual histograms to a common X-axis."""
        if self.x is not None:
            x_temp = self.x
        elif x_array is not None:
            x_temp = x_array
        else:
            raise ValueError("No X values provided to map histograms.")

        if redefine_x:
            x_temp = np.linspace(np.min(x_temp), np.max(x_temp), points)
            self.x = x_temp

        self.maps = [h.map_histogram(x_temp) for h in self.histo_list]


def adjust_norm(data: np.ndarray, bins: int = 30) -> Tuple[float, float, float]:
    """Fits a Gaussian distribution to the data and returns (mu, sigma, height)."""
    hist, bin_edges = np.histogram(data, bins=bins, density=False)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    def gauss(x, a, mu, sigma):
        return a * np.exp(-(x - mu)**2 / (2 * sigma**2))

    p0 = [hist.max(), np.mean(data), np.std(data)]
    try:
        popt, _ = curve_fit(gauss, centers, hist, p0=p0)
        return round(popt[1], 6), round(popt[2], 6), popt[0]
    except RuntimeError:
        return round(np.mean(data), 6), round(np.std(data), 6), float(hist.max())


def read_input_locations(log_list_file: str, pullx_list_file: str) -> Tuple[List[FileManager], List[XVG]]:
    """Reads paths from index files and returns lists of FileHandlers."""
    logs = FileManager(log_list_file).read_lines()
    pullxs = FileManager(pullx_list_file).read_lines()
    
    log_handlers = [FileManager(path) for path in logs]
    pullx_handlers = [XVG(path) for path in pullxs]
    
    return log_handlers, pullx_handlers


def profile_computer(
    log_files: str, pullx_files: str, temperature: float = 300.0, k_B: float = 0.0083144621, 
    n_bins: int = 100, x_range: Optional[Tuple[float, float]] = None, tolerance: float = 1.0e-4, 
    max_iter: int = 10000, verbose: bool = False, set_min: bool = True,
    max_pmf: Optional[float] = None, min_left: Optional[float] = None, min_right: Optional[float] = None,
    rm_extrema: Optional[float] = None, save: bool = True, profile_name: str = 'profile.xvg',
    histo_name: str = 'histo.xvg', histo_points: Optional[int] = None
) -> Tuple[Profile, HistoArray]:
    """
    Main logic to read simulation data, run WHAM, and generate PMF/Histogram objects.
    """
    log_loc, pullx_loc = read_input_locations(log_files, pullx_files)
    
    mu_list, std_list, h_list = [], [], []
    sampled_pos, centers, force_const = [], [], []

    for l_file, p_file in zip(log_loc, pullx_loc):
        # Extract metadata from log
        init_val = float(l_file.find_line(' init ').split()[-1])
        k_val = float(l_file.find_line(' k ').split()[-1])
        
        # Extract data from XVG
        _, _, pullx_data, _ = p_file.read_xvg(verbose=verbose)
        
        centers.append(init_val)
        force_const.append(k_val)
        sampled_pos.append(pullx_data)
        
        # Fit Gaussian for histogram visualization
        mu, sigma, h = adjust_norm(pullx_data, bins=100)
        mu_list.append(mu)
        std_list.append(sigma)
        h_list.append(h)

    # Convert to arrays for WHAM
    sampled_pos_arr = np.array(sampled_pos, dtype=object) # Windows may have different lengths
    centers_arr = np.array(centers)
    force_const_arr = np.array(force_const)

    # Run WHAM
    b_centers, b_pmf = wham(
        sampled_pos_arr, centers_arr, force_const_arr, 
        temperature=temperature, k_B=k_B, n_bins=n_bins, 
        x_range=x_range, tolerance=tolerance, max_iter=max_iter, verbose=verbose
    )

    # Initialize Profile Object
    pmf_profile = Profile(b_centers, b_pmf, max_pos=max_pmf, min_left=min_left, min_right=min_right)
    
    if set_min:
        pmf_profile.pmf -= pmf_profile.min_val
        
    if rm_extrema:
        pmf_profile.remove_extrema(max_lim=rm_extrema)

    # Prepare Histograms
    histo_data = HistoArray(mu_list=mu_list, std_list=std_list, h_list=h_list, x=pmf_profile.x)

    if save:
        # Save PMF
        XVG(profile_name, read=False).write_xvg(pmf_profile.x, pmf_profile.y)
        # Save Histograms
        redefine = histo_points is not None
        histo_data.map_all(points=histo_points or 100, redefine_x=redefine)
        XVG(histo_name, read=False).write_xvg(histo_data.x, *histo_data.maps)

    return pmf_profile, histo_data


def main():
    args = get_args()
    pmf, histo = profile_computer(
        args.log, args.pullx, temperature=args.temperature,
        k_B=args.boltzmann, n_bins=args.bins, tolerance=args.tol,
        max_pmf=args.max, min_left=args.min_from, min_right=args.min_before,
        set_min=args.set_min, max_iter=args.maxiter, rm_extrema=args.extrema,
        save=True, profile_name=args.profile, histo_name=args.histo, 
        verbose=True, histo_points=args.histo_points
    )
    
    if args.show_dg:
        print(f"Calculated Delta G: {pmf.dg}")


if __name__ == '__main__':
    main()
    