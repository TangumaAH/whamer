# === compute_profile.py ===
from dataclasses import dataclass, field
import numpy as np
from scipy.signal import argrelextrema
from scipy.stats import norm
from scipy.optimize import curve_fit
from whamer import wham
import os
import argparse


def parser():
    parser = argparse.ArgumentParser(description ='Plot PMF profile')
    # wham options
    parser.add_argument('-il', '--log', help='File with location of all the log files to read init and k.',  type=str, required=True,)
    parser.add_argument('-ix', '--pullx',   help='File with location of all pullx.xvg files.',type=str, required=True,)
    parser.add_argument('-t', '--temperature',  help='Temperature',   type=float, default=300)
    parser.add_argument('-kb', '--boltzmann',  help='Boltzmann constant value', type=float, default=0.0083144621)
    parser.add_argument('-nb', '--bins',  help='Number of bins used in WHAM', type=int, default=100)
    parser.add_argument('-wi', '--maxiter',  help='Maximum WHAM iterations', type=int, default=1000000)
    parser.add_argument('-wt', '--tol',  help='Tolerance of convergence', type=float, default=1.0e-07)
    # profile options
    parser.add_argument('-pro', '--profile', help='Name of the profile xvg file', type=str, default='profile.xvg')
    parser.add_argument('-his', '--histo', help='Name of the hitograms xvg file', type=str, default='histo.xvg')
    parser.add_argument('-ex', '--extrema', help='Position of the extrema', type=float, default=None)
    parser.add_argument('-sm', '--set_min', help='Translate profile to min value', action='store_true')
    parser.add_argument('-hpoints', '--histo_points', help='Translate profile to min value', type=int, default=None)
    # DG options
    parser.add_argument('-dg', '--show_dg', help='Show free energy difference in the plot', action='store_true')
    parser.add_argument('-max', '--max', help=r'Position of the maximum to compute $\Delta$G', type=float, default=None)
    parser.add_argument('-minf', '--min_from', help='Look for minimum from this value to infinity.', type=float, default=None)
    parser.add_argument('-minb', '--min_before', help='Look for minimum from infinity to this value.', type=float, default=None)

    args = parser.parse_args()

    return args

@dataclass
class FileManager:
    name: str
    read: bool = True

    def __post_init__(self):
        if not self.__test_open(self.name) and self.read:
            raise FileNotFoundError(f"File '{self.name}' does not exist")

    def __test_open(self, filename):
        return os.path.isfile(filename)

    def find_line(self, tag):
        """
        Returns the first line including tag
        """
        with open(self.name, 'r') as f:
            for line in f:
                if tag in line:
                    return line
    
    def read_lines(self, remove_empty=True, remove_start: list = []):
        """
        Return all the lines of the file
        """
        with open(self.name, 'r') as f:
            lines = f.readlines()
        if remove_empty:
            lines = [k for k in lines if len(k) > 0]
        if len(remove_start) > 0:
            lines = [k for k in lines if k[0] not in remove_start]
        return lines
    
    def write_file(self, lines: list, eol=''):
        with open(self.name, 'w') as f:
            for line in lines:   
                f.write(line + eol)

@dataclass
class XVG(FileManager):
    def read_xvg(self, column=1, verbose=False):
        if verbose:
            print('Reading:', self.name) #
        if not self.read:
            if verbose:
                print('This file is not readable')
            return
        lines = self.read_lines(remove_empty=True, remove_start=['#'])
        # reading instructions and data
        xaxis = ''
        yaxis = ''
        xdata = []
        ydata = []
        for line in lines:
            if line[0] == '@':
                if 'xaxis' in line:
                    xaxis = line.split('"')[1]
                if 'yaxis' in line:
                    yaxis = line.split('"')[1]
                continue
            xdata.append(float(line.split()[0]))
            ydata.append(float(line.split()[column]))
        return np.array(xdata), xaxis, np.array(ydata), yaxis
    
    def write_xvg(self, *arrays):
        lines = []
        for a in zip(*arrays):
            if any([np.isnan(k) for k in a]):
                continue
            lines.append([f'{k}\t' for k in a])
        lines = [''.join(k) for k in lines]
        self.write_file(lines, eol='\n')

@dataclass
class Profile:
    bins: list
    pmf: list
    max_pos: float = None
    min_left: float = None
    min_right: float = None

    def remove_extrema(self, min_lim: float = None, max_lim: float = None):
        """
        Remove extrema of profile. The result is the middle part of the profile from min_lim to max_lim.
        """
        left_idx = 0
        right_idx = 10000  # stupid
        if min_lim is not None:
            diff = [[n, abs(k-min_lim)] for n, k in enumerate(np.nan_to_num(self.bins))]
            diff.sort(key=lambda x: x[1]) # sort using differences
            left_idx = diff[0][0]
        if max_lim is not None:
            diff = [[n, abs(k-max_lim)] for n, k in enumerate(np.nan_to_num(self.bins))]
            diff.sort(key=lambda x: x[1]) # sort using differences
            right_idx = diff[0][0]
        self.bins = self.bins[left_idx:right_idx]
        self.pmf = self.pmf[left_idx:right_idx]

    @property
    def dg(self):
        return round(self.max - self.min, 4)
    
    @property
    def min_idx(self):
        data = self.pmf
        left_idx = 0
        # cut left part of pmf
        if self.min_left is not None:
            diff = [[n, abs(k-self.min_left)] for n, k in enumerate(np.nan_to_num(self.bins))]
            diff.sort(key=lambda x: x[1]) # sort using differences
            left_idx = diff[0][0]
            data = data[diff[0][0]:]
        # cut left part of pmf
        if self.min_right is not None:
            diff = [[n, abs(k-self.min_right)] for n, k in enumerate(np.nan_to_num(self.bins))]
            diff.sort(key=lambda x: x[1]) # sort using differences
            data = data[:diff[0][0]]
        
        # get minima
        try:
            idx_min = argrelextrema(data, np.less, order=5)[0][0]
        except:
            idx_min = np.argmin(np.nan_to_num(data))
        return left_idx + idx_min
    
    @property
    def min(self):
        # get minima
        idx_min = self.min_idx
        return self.pmf[idx_min]
    
    @property
    def max(self):
        """
        Find maximum and return value
        """
        # how to find max?
        if self.max_pos is not None:
            # look for closest x position
            diff = [[n, abs(k-self.max_pos)] for n, k in enumerate(np.nan_to_num(self.bins))]
            diff.sort(key=lambda x: x[1]) # sort using differences
            idx_max = diff[0][0]
            return self.pmf[idx_max]
        return np.max(np.nan_to_num(self.pmf))

    @property
    def x(self):
        return self.bins
    
    @property
    def y(self):
        return self.pmf


@dataclass
class HistoArray:
    mu_list: list
    std_list: list
    h_list: list = None
    x: list = None

    def __post_init__(self):
        if self.h_list is None:
            h_temp = [None for k in self.mu_list]
        else:
            h_temp = self.h_list
        self.histo_list = []
        for mu, std, h in zip(self.mu_list, self.std_list, h_temp):
            self.histo_list.append(Histo(mu=mu, std=std, h=h))
        
    def map_all(self, start: float = None, end: float = None, points=100, x_array: np.ndarray = None, redefine_x=False):
        self.maps = []
        if self.x is not None:
            x_temp = self.x
        elif x_array is not None:
            x_temp = x_array
        elif start is not None and end is not None:
            x_temp = np.linspace(start, end, points)
            redefine_x = False
        else:
            raise ValueError("No X values received to map histograms.")
        if redefine_x:
            x_min = np.min(x_temp)
            x_max = np.max(x_temp)
            x_temp = np.linspace(x_min, x_max, points)
            self.x = x_temp
        for h in self.histo_list:
            y = h.map_histogram(x_array=x_temp)
            self.maps.append(y)

@dataclass
class Histo:
    mu: float
    std: float
    h: float = None

    def map_histogram(self, x_array: np.ndarray = None):
        """
        map histogram (Y) using X values from x_array
        """
        # check if h is None
        if self.h is None:
            h_temp = 1
        else:
            h_temp = self.h
        # create actual histogram
        x = x_array
        y = norm.pdf(x, self.mu, self.std) * h_temp
        return y

def adjust_norm(data, bins=30, decimals=6, density=False):
    # fit curve
    data = np.asarray(data)
    hist, bin_edges = np.histogram(data, bins=bins, density=density)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    def gauss(x, a, mu, sigma):
        return a * np.exp(-(x - mu)**2 / (2 * sigma**2))
    p0 = [hist.max(), data.mean(), data.std()]
    popt, _ = curve_fit(gauss, centers, hist, p0=p0)
    h, mu, sigma = popt
    return round(mu, decimals), round(sigma, decimals), h

def read_files(log_files: str, pullx_files: str):
    """Reads files and returns lists of FileManager objects"""
    # read file with all log and pullx locations
    log_loc = FileManager(log_files)
    log_loc = log_loc.read_lines()
    pullx_loc = FileManager(pullx_files)
    pullx_loc = pullx_loc.read_lines()
    # create FileManager per file in *_loc
    log_loc = [FileManager(k.strip()) for k in log_loc]
    pullx_loc = [XVG(k.strip()) for k in pullx_loc]
    return log_loc, pullx_loc

def profile_computer(log_files: str, pullx_files: str, 
                     temperature=300, k_B=0.0083144621, 
                     n_bins=100, x_range=None, tolerance=1.0e-4, 
                     max_iter=10000, verbose=False, set_min=True,
                     max_pmf=None, min_left=None, min_right=None,
                     rm_extrema=None, save=True, profile_name='profile.xvg',
                     histo_name='histo.xvg', histo_points=None):
    """
    Read files and compute profiles
    """
    # read file with all log and pullx locations
    log_loc, pullx_loc = read_files(log_files, pullx_files)
    # read all the configurations
    mu_list = []
    std_list = []
    h_list = []
    sampled_pos = []
    centers = []
    force_const = []
    for l, p in zip(log_loc, pullx_loc):
        # get init line
        init = float(l.find_line(tag=' init ').split()[-1])
        centers.append(init)
        # get k line
        pull_k = float(l.find_line(tag=' k ').split()[-1])
        force_const.append(pull_k)
        # get pullx and time
        time, _, pullx, _ = p.read_xvg(verbose=verbose)
        sampled_pos.append(pullx)
        # gaussian
        mu, sigma, h = adjust_norm(pullx, bins=100)
        mu_list.append(mu)
        std_list.append(sigma)
        h_list.append(h)
    # transform into numpy array
    sampled_pos = np.array(sampled_pos)
    centers = np.array(centers)
    force_const = np.array(force_const)
    # create profile from real data ---------
    bin_real, pmf_real = wham(sampled_pos, centers, force_const, temperature=temperature,
                        k_B=k_B, n_bins=n_bins, x_range=x_range,
                        tolerance=tolerance, max_iter=max_iter, verbose=verbose)
    if set_min:
        # get minima
        idx_min = argrelextrema(pmf_real, np.less, order=5)
        idx_min = idx_min[0][0]
        # translate to minima
        pmf_real -= pmf_real[idx_min]
    pmf_real = Profile(bin_real, pmf_real, max_pos=max_pmf, min_left=min_left, min_right=min_right)
    histo = HistoArray(mu_list=mu_list, std_list=std_list, h_list=h_list, x=pmf_real.x)
    # Remove the last part of the profile (if rm_extrema is not None)
    pmf_real.remove_extrema(max_lim=rm_extrema)
    # save xvg files
    if save:
        outfile = XVG(profile_name, read=False)
        outfile.write_xvg(pmf_real.x, pmf_real.y)
        outfile = XVG(histo_name, read=False)
        # make histograms
        redefine_x = False
        if histo_points is not None:
            redefine_x = True
        histo.map_all(points=histo_points, redefine_x=redefine_x)
        outfile.write_xvg(histo.x, *[k for k in histo.maps])
    return pmf_real, histo

def main():
    args = parser()
    pmf, histo = profile_computer(args.log, args.pullx, temperature=args.temperature,
                                k_B=args.boltzmann, n_bins=args.bins, tolerance=args.tol,
                                max_pmf=args.max, min_left=args.min_from, min_right=args.min_before,
                                set_min=args.set_min, max_iter=args.maxiter, rm_extrema=args.extrema,
                                save=True, profile_name=args.profile, histo_name=args.histo, 
                                verbose=True, histo_points=args.histo_points)


if __name__ == '__main__':
    main()
