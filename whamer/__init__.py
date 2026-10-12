"""
whamer
======
Simple WHAM implementation in Python.

Public API
----------
    wham            — Core WHAM solver.
    profile_computer — High-level PMF computation from GROMACS files.
    Profile         — PMF profile container and analysis helpers.
    Histo           — Single Gaussian histogram distribution.
    HistoArray      — Collection of Gaussian histograms.
    FileManager     — Generic file reader/writer.
    XVG             — GROMACS XVG file handler.
"""

from whamer.whamer import wham
from whamer.computer import profile_computer
from whamer.profile import Profile, Histo, HistoArray
from whamer.utils import FileManager, XVG

__all__ = [
    "wham",
    "profile_computer",
    "Profile",
    "Histo",
    "HistoArray",
    "FileManager",
    "XVG",
]

__version__ = "0.1.0"
__author__ = "Alejandro H. Tanguma"
