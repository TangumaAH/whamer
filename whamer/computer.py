"""
computer.py
Reads GROMACS simulation data and computes PMF profiles using WHAM.

Author: Alejandro H. Tanguma
Version: 0.1.0
"""

import warnings
import argparse
import numpy as np
from typing import List, Tuple, Optional
from scipy.optimize import curve_fit

from whamer.whamer import wham
from whamer.profile import Profile, Histo, HistoArray
from whamer.utils import (
    FileManager, XVG,
    _DEFAULT_POSITION_UNITS, _DEFAULT_ENERGY_UNITS,
)

# Reference value of k_B (kJ·mol⁻¹·K⁻¹). Used to detect non-default units.
_DEFAULT_K_B: float = 0.0083144621

_UNITS_WARNING: str = (
    "A non-default Boltzmann constant was supplied but no unit labels were given. "
    "Energy and position units may differ from the defaults "
    f"('{_DEFAULT_ENERGY_UNITS}' and '{_DEFAULT_POSITION_UNITS}'). "
    "Pass energy_units and position_units explicitly to suppress this warning."
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_args() -> argparse.Namespace:
    """Defines and parses command-line arguments for PMF profile computation."""
    parser = argparse.ArgumentParser(
        description=(
            'Process trajectory data to compute and plot PMF profiles using WHAM.'
        )
    )

    # --- Input sources ---
    input_group = parser.add_argument_group('Input Options')
    input_group.add_argument(
        '-il', '--log', type=str, default=None,
        help=(
            'Path to a text file listing .log files '
            '(used to read initial positions and force constants). '
            'Required unless --inits and --k-values are both provided.'
        ),
    )
    input_group.add_argument(
        '-ii', '--inits', type=str, default=None,
        help=(
            'Path to a plain-text file with initial positions (one value per line '
            'or space-separated). Replaces the init values from --log.'
        ),
    )
    input_group.add_argument(
        '-ik', '--k-values', type=str, default=None,
        help=(
            'Path to a plain-text file with force constants (one value per line '
            'or space-separated). Replaces the k values from --log.'
        ),
    )
    input_group.add_argument(
        '-ix', '--pullx', type=str, required=True,
        help='Path to a text file listing all pullx.xvg files (trajectory data).',
    )

    # --- WHAM options ---
    wham_group = parser.add_argument_group('WHAM Options')
    wham_group.add_argument(
        '-t', '--temperature', type=float, default=300.0,
        help='Simulation temperature in Kelvin (default: 300).',
    )
    wham_group.add_argument(
        '-kb', '--boltzmann', type=float, default=_DEFAULT_K_B,
        help=(
            f'Boltzmann constant (default: {_DEFAULT_K_B} kJ·mol⁻¹·K⁻¹). '
            'If changed, also set --E-units and --p-units.'
        ),
    )
    wham_group.add_argument(
        '-nb', '--bins', type=int, default=100,
        help='Number of bins to discretise the reaction coordinate (default: 100).',
    )
    wham_group.add_argument(
        '-wi', '--maxiter', type=int, default=1_000_000,
        help='Maximum WHAM iterations (default: 1,000,000).',
    )
    wham_group.add_argument(
        '-wt', '--tol', type=float, default=1.0e-07,
        help='Convergence tolerance for free-energy iterations (default: 1e-07).',
    )

    # --- Unit labels ---
    unit_group = parser.add_argument_group('Unit Options')
    unit_group.add_argument(
        '--E-units', type=str, default=None,
        help=(
            f'Energy unit label for output files (default: \'{_DEFAULT_ENERGY_UNITS}\'). '
            'Must be set when --boltzmann differs from the default.'
        ),
    )
    unit_group.add_argument(
        '--p-units', type=str, default=None,
        help=(
            f'Position unit label for output files (default: \'{_DEFAULT_POSITION_UNITS}\'). '
            'Must be set when --boltzmann differs from the default.'
        ),
    )

    # --- Profile output options ---
    prof_group = parser.add_argument_group('Profile Options')
    prof_group.add_argument(
        '-pro', '--profile', type=str, default='profile.xvg',
        help='Output filename for the PMF profile (default: profile.xvg).',
    )
    prof_group.add_argument(
        '-his', '--histo', type=str, default='histo.xvg',
        help='Output filename for the reconstructed histograms (default: histo.xvg).',
    )
    prof_group.add_argument(
        '-ex', '--extrema', type=float, default=None,
        help='Crop the profile beyond this x-position.',
    )
    prof_group.add_argument(
        '-sm', '--set_min', action='store_true',
        help='Shift the entire PMF so the global minimum is at 0.',
    )
    prof_group.add_argument(
        '-hpoints', '--histo_points', type=int, default=300,
        help='Number of interpolation points for the output histograms (default: 300).',
    )

    # --- Delta G analysis options ---
    dg_group = parser.add_argument_group('Analysis Options')
    dg_group.add_argument(
        '-dg', '--show_dg', action='store_true',
        help='Calculate and display the free energy difference (Delta G).',
    )
    dg_group.add_argument(
        '-max', '--max', type=float, default=None,
        help='X-position of the maximum state used for Delta G.',
    )
    dg_group.add_argument(
        '-minf', '--min_from', type=float, default=None,
        help='Lower bound for the local minimum search.',
    )
    dg_group.add_argument(
        '-minb', '--min_before', type=float, default=None,
        help='Upper bound for the local minimum search.',
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helper: read plain-text value files
# ---------------------------------------------------------------------------

def _read_values_file(path: str) -> List[float]:
    """
    Reads a plain-text file of numeric values separated by whitespace or newlines.

    Parameters:
        path: Path to the file.

    Returns:
        List of float values.
    """
    fm = FileManager(path)
    lines = fm.read_lines(remove_empty=True)
    values: List[float] = []
    for line in lines:
        for token in line.split():
            values.append(float(token))
    return values


# ---------------------------------------------------------------------------
# Helper: Gaussian fit for histogram visualisation
# ---------------------------------------------------------------------------

def adjust_norm(data: np.ndarray, bins: int = 30) -> Tuple[float, float, float]:
    """
    Fits a Gaussian distribution to the data.

    Parameters:
        data: 1D array of sampled positions.
        bins: Number of histogram bins used for the fit.

    Returns:
        Tuple (mu, sigma, height) of the fitted Gaussian.
    """
    hist, bin_edges = np.histogram(data, bins=bins, density=False)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    def gauss(x, a, mu, sigma):
        return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    p0 = [hist.max(), np.mean(data), np.std(data)]
    try:
        popt, _ = curve_fit(gauss, centers, hist, p0=p0)
        return round(popt[1], 6), round(popt[2], 6), popt[0]
    except RuntimeError:
        return round(np.mean(data), 6), round(np.std(data), 6), float(hist.max())


# ---------------------------------------------------------------------------
# Helper: read input file locations
# ---------------------------------------------------------------------------

def _read_input_locations(
        pullx_list_file: str,
        log_list_file: Optional[str] = None,
) -> Tuple[Optional[List[FileManager]], List[XVG]]:
    """
    Reads path lists from index files and returns file handler objects.

    Parameters:
        pullx_list_file: Path to the file listing pullx.xvg paths.
        log_list_file: Path to the file listing .log paths (optional).

    Returns:
        Tuple of (log_handlers or None, pullx_handlers).
    """
    pullxs = FileManager(pullx_list_file).read_lines()
    pullx_handlers = [XVG(path) for path in pullxs]

    log_handlers = None
    if log_list_file is not None:
        logs = FileManager(log_list_file).read_lines()
        log_handlers = [FileManager(path) for path in logs]

    return log_handlers, pullx_handlers


# ---------------------------------------------------------------------------
# Main API function
# ---------------------------------------------------------------------------

def profile_computer(
    pullx_files: str,
    log_files: Optional[str] = None,
    inits_file: Optional[str] = None,
    k_values_file: Optional[str] = None,
    temperature: float = 300.0,
    k_B: float = _DEFAULT_K_B,
    n_bins: int = 100,
    x_range: Optional[Tuple[float, float]] = None,
    tolerance: float = 1.0e-4,
    max_iter: int = 10000,
    verbose: bool = False,
    set_min: bool = True,
    max_pmf: Optional[float] = None,
    min_left: Optional[float] = None,
    min_right: Optional[float] = None,
    rm_extrema: Optional[float] = None,
    save: bool = True,
    profile_name: str = 'profile.xvg',
    histo_name: str = 'histo.xvg',
    histo_points: Optional[int] = None,
    energy_units: Optional[str] = None,
    position_units: Optional[str] = None,
) -> Tuple[Profile, HistoArray]:
    """
    Reads GROMACS simulation data, runs WHAM, and returns PMF and histogram objects.

    Centers and force constants can be provided in two ways (mutually exclusive):
      1. Via log_files: paths are read from .log files (original behaviour).
      2. Via inits_file + k_values_file: plain-text files with numeric values.

    If both are provided, inits_file and k_values_file take precedence.
    If neither is provided, a ValueError is raised.

    Parameters:
        pullx_files: Path to a file listing pullx.xvg trajectory files.
        log_files: Path to a file listing GROMACS .log files (optional).
        inits_file: Path to a plain-text file with initial positions (optional).
        k_values_file: Path to a plain-text file with force constants (optional).
        temperature: Simulation temperature in Kelvin.
        k_B: Boltzmann constant. If different from the default, also set
             energy_units and position_units to avoid a unit warning.
        n_bins: Number of histogram bins.
        x_range: Optional (min, max) tuple for the reaction coordinate range.
        tolerance: WHAM convergence tolerance.
        max_iter: Maximum WHAM iterations.
        verbose: If True, print WHAM iteration details.
        set_min: If True, shift the PMF so its minimum is at zero.
        max_pmf: X-position of the PMF maximum for Delta G computation.
        min_left: Lower bound for the minimum search.
        min_right: Upper bound for the minimum search.
        rm_extrema: Crop the profile beyond this x-position.
        save: If True, write the PMF and histogram XVG files.
        profile_name: Output filename for the PMF profile.
        histo_name: Output filename for the histograms.
        histo_points: Number of interpolation points for histograms.
        energy_units: Energy unit label for output XVG files.
        position_units: Position unit label for output XVG files.

    Returns:
        pmf_profile: Profile object with the computed PMF.
        histo_data: HistoArray object with the fitted histograms.

    Raises:
        ValueError: If neither (log_files) nor (inits_file + k_values_file) are given,
                    or if only one of inits_file / k_values_file is given.
    """

    # --- Unit warning when k_B is non-default ---
    if k_B != _DEFAULT_K_B and (energy_units is None or position_units is None):
        warnings.warn(_UNITS_WARNING, UserWarning, stacklevel=2)

    # Resolve final unit labels
    e_units = energy_units if energy_units is not None else _DEFAULT_ENERGY_UNITS
    p_units = position_units if position_units is not None else _DEFAULT_POSITION_UNITS

    # --- Validate input mode ---
    use_plain_files = (inits_file is not None) or (k_values_file is not None)

    if use_plain_files:
        if inits_file is None or k_values_file is None:
            raise ValueError(
                "--inits and --k-values must both be provided together."
            )
    else:
        if log_files is None:
            raise ValueError(
                "Provide either --log, or both --inits and --k-values."
            )

    # --- Read pullx trajectories ---
    log_handlers, pullx_handlers = _read_input_locations(
        pullx_files,
        log_list_file=None if use_plain_files else log_files,
    )

    # --- Build centers and force_constants arrays ---
    if use_plain_files:
        centers = _read_values_file(inits_file)
        force_const_list = _read_values_file(k_values_file)
    else:
        centers = []
        force_const_list = []

    mu_list, std_list, h_list = [], [], []
    sampled_pos: List[np.ndarray] = []

    for idx, p_file in enumerate(pullx_handlers):
        # Read trajectory data
        _, _, pullx_data, _ = p_file.read_xvg(verbose=verbose)
        sampled_pos.append(pullx_data)

        # If using log files, extract metadata here (one per window)
        if not use_plain_files:
            l_file = log_handlers[idx]
            init_val = float(l_file.find_line(' init ').split()[-1])
            k_val = float(l_file.find_line(' k ').split()[-1])
            centers.append(init_val)
            force_const_list.append(k_val)

        # Fit Gaussian for histogram visualisation
        mu, sigma, h = adjust_norm(pullx_data, bins=100)
        mu_list.append(mu)
        std_list.append(sigma)
        h_list.append(h)

    centers_arr = np.array(centers)
    force_const_arr = np.array(force_const_list)
    # Windows may have different lengths so dtype=object is required
    sampled_pos_arr = np.array(sampled_pos, dtype=object)

    # --- Run WHAM ---
    b_centers, b_pmf = wham(
        sampled_pos_arr, centers_arr, force_const_arr,
        temperature=temperature, k_B=k_B, n_bins=n_bins,
        x_range=x_range, tolerance=tolerance, max_iter=max_iter, verbose=verbose,
    )

    # --- Build Profile object ---
    pmf_profile = Profile(
        b_centers, b_pmf,
        max_pos=max_pmf, min_left=min_left, min_right=min_right,
    )

    if set_min:
        pmf_profile.pmf -= pmf_profile.min_val

    if rm_extrema:
        pmf_profile.remove_extrema(max_lim=rm_extrema)

    # --- Build HistoArray object ---
    histo_data = HistoArray(
        mu_list=mu_list, std_list=std_list, h_list=h_list, x=pmf_profile.x,
    )

    # --- Save outputs ---
    if save:
        XVG(profile_name, read=False).write_xvg(
            pmf_profile.x, pmf_profile.y,
            x_units=p_units, y_units=e_units,
        )
        redefine = histo_points is not None
        histo_data.map_all(
            points=histo_points or 3 * n_bins, redefine_x=redefine,
        )
        XVG(histo_name, read=False).write_xvg(
            histo_data.x, *histo_data.maps,
            x_units=p_units, y_units="counts",
        )

    return pmf_profile, histo_data


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Command-line entry point for the compute_profile command."""
    args = get_args()

    # --- Validate input mode ---
    use_plain = (args.inits is not None) or (getattr(args, 'k_values', None) is not None)
    k_values_arg = getattr(args, 'k_values', None)

    if use_plain:
        if args.inits is None or k_values_arg is None:
            raise SystemExit(
                "Error: --inits and --k-values must both be provided together."
            )
    else:
        if args.log is None:
            raise SystemExit(
                "Error: provide either --log, or both --inits and --k-values."
            )

    # --- Unit warning ---
    if args.boltzmann != _DEFAULT_K_B and (args.E_units is None or args.p_units is None):
        warnings.warn(_UNITS_WARNING, UserWarning, stacklevel=1)

    e_units = args.E_units if args.E_units is not None else _DEFAULT_ENERGY_UNITS
    p_units = args.p_units if args.p_units is not None else _DEFAULT_POSITION_UNITS

    pmf, _ = profile_computer(
        pullx_files=args.pullx,
        log_files=args.log,
        inits_file=args.inits,
        k_values_file=k_values_arg,
        temperature=args.temperature,
        k_B=args.boltzmann,
        n_bins=args.bins,
        tolerance=args.tol,
        max_pmf=args.max,
        min_left=args.min_from,
        min_right=args.min_before,
        set_min=args.set_min,
        max_iter=args.maxiter,
        rm_extrema=args.extrema,
        save=True,
        profile_name=args.profile,
        histo_name=args.histo,
        verbose=True,
        histo_points=args.histo_points,
        energy_units=e_units,
        position_units=p_units,
    )

    if args.show_dg:
        print(f"Calculated Delta G: {pmf.dg}")


if __name__ == '__main__':
    main()
