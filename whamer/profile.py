"""
profile.py
Classes for representing and analysing PMF profiles and histogram distributions.

Author: Alejandro H. Tanguma
Version: 0.1.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from scipy.signal import argrelextrema, savgol_filter, medfilt


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

    def get_min_smoothing(
            self,
            window_med: int = 5,
            window_sg: int = 11,
            order_sg: int = 3,
    ) -> int:
        """
        Finds the index of the minimum by smoothing.

        Parameters:
            window_med: Size of the window for median filtering (must be odd).
            window_sg: Window size for Savitzky-Golay filter (must be odd).
            order_sg: Polynomial order for Savitzky-Golay filter.

        Returns:
            idx_min: Index of the smoothed minimum.
        """
        # Remove spikes with a median filter
        y_med = medfilt(self.y, kernel_size=window_med)
        # Smooth with Savitzky-Golay
        y_smooth = savgol_filter(y_med, window_length=window_sg, polyorder=order_sg)
        idx_min = np.argmin(y_smooth)
        return idx_min

    @property
    def dg(self) -> float:
        """Calculates the Free Energy Difference (Delta G)."""
        return round(float(self.max_val - self.min_val), 4)

    @property
    def min_idx(self) -> int:
        """
        Finds the index of the minimum value within the specified bounds.
        Tries smoothing via Savitzky-Golay first; falls back to argrelextrema,
        then to the absolute minimum ignoring NaN values.
        """
        try:
            idx_min = self.get_min_smoothing()
        except Exception:
            data = self.pmf
            left_offset = 0

            # Apply optional bounds
            if self.min_left is not None:
                left_offset = np.argmin(np.abs(self.bins - self.min_left))
                data = data[left_offset:]
            if self.min_right is not None:
                right_idx = np.argmin(np.abs(self.bins - self.min_right))
                data = data[:right_idx - left_offset]

            try:
                idx_min = argrelextrema(data, np.less, order=5)[0][0]
            except (IndexError, ValueError):
                idx_min = np.nanargmin(data)
            idx_min = left_offset + idx_min
        return idx_min

    @property
    def min_val(self) -> float:
        """Returns the PMF value at the minimum index."""
        return self.pmf[self.min_idx]

    @property
    def max_val(self) -> float:
        """Returns the maximum PMF value, either at max_pos or globally."""
        if self.max_pos is not None:
            idx_max = np.argmin(np.abs(self.bins - self.max_pos))
            return self.pmf[idx_max]
        return np.nanmax(self.pmf)

    @property
    def x(self) -> np.ndarray:
        """Returns the x dimension (bin centres)."""
        return self.bins

    @property
    def y(self) -> np.ndarray:
        """Returns the y dimension (PMF values)."""
        return self.pmf


@dataclass
class Histo:
    """Represents a single Gaussian histogram distribution."""
    mu: float
    std: float
    h: Optional[float] = 1.0

    def map_histogram(self, x_array: np.ndarray) -> np.ndarray:
        """Generates Y values for the Gaussian curve over x_array."""
        from scipy.stats import norm
        height = self.h if self.h is not None else 1.0
        return norm.pdf(x_array, self.mu, self.std) * height


@dataclass
class HistoArray:
    """Manages a collection of Gaussian histograms."""
    mu_list: List[float]
    std_list: List[float]
    h_list: Optional[List[float]] = None
    x: Optional[np.ndarray] = None
    maps: List[np.ndarray] = field(default_factory=list)

    def __post_init__(self):
        h_vals = self.h_list if self.h_list else [1.0] * len(self.mu_list)
        self.histo_list = [
            Histo(mu, std, h)
            for mu, std, h in zip(self.mu_list, self.std_list, h_vals)
        ]

    def map_all(
            self,
            points: int = 300,
            x_array: Optional[np.ndarray] = None,
            redefine_x: bool = False,
    ):
        """Maps all individual histograms onto a common X-axis."""
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
