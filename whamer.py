# ---------------------------------------------------------
# DISCLAIMER: This code has been typed and commented by Gemini.
# ---------------------------------------------------------

"""
whamer.py
Implementation of the Weighted Histogram Analysis Method (WHAM).

Author: Alejandro H. Tanguma
Version: 0.1.0
"""

import numpy as np
from typing import List, Tuple, Optional, Union


def wham(
    trajectories: Union[np.ndarray, List[np.ndarray]],
    centers: np.ndarray,
    force_constants: np.ndarray,
    temperature: float = 300.0,
    k_B: float = 0.0083144621,
    n_bins: int = 200,
    x_range: Optional[Tuple[float, float]] = None,
    tolerance: float = 1.0e-06,
    max_iter: int = 10000,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the Potential of Mean Force (PMF) through the WHAM method.

    Parameters:
        trajectories: Array or list of 1D NumPy arrays containing sampled windows.
        centers: Position of the bias potentials (one per window).
        force_constants: Force constants of the bias potentials (one per window).
        temperature: Simulation temperature in Kelvin. Default is 300 K.
        k_B: Boltzmann constant. Default is 0.0083144621 kJ*mol^-1*K^-1.
        n_bins: Number of bins for the histogram. Default is 200.
        x_range: Tuple (min, max) for defining bins. If None, calculated from data.
        tolerance: Convergence tolerance for the iterative process.
        max_iter: Maximum number of iterations allowed.
        verbose: If True, prints the maximum change (delta) per iteration.

    Returns:
        bin_centers: 1D NumPy array with the center of each bin.
        pmf: 1D NumPy array with the PMF profile (normalized to min = 0).
    """

    # 1. Validation and Initial Setup
    n_windows: int = len(trajectories)
    if len(centers) != n_windows or len(force_constants) != n_windows:
        raise ValueError(
            "The length of 'centers' and 'force_constants' must match the number of windows."
        )

    # Number of frames per window (N_i)
    frames_per_window: np.ndarray = np.array([len(traj) for traj in trajectories])

    # 2. Define Bin Range
    if x_range is None:
        all_data: np.ndarray = np.concatenate(trajectories)
        x_min, x_max = np.nanmin(all_data), np.nanmax(all_data)
        # Margin to prevent edge effects
        margin: float = 0.05 * (x_max - x_min)
        x_min -= margin
        x_max += margin
    else:
        x_min, x_max = x_range

    # 3. Discretization and Histogram Computation
    bin_edges: np.ndarray = np.linspace(x_min, x_max, n_bins + 1)
    bin_centers: np.ndarray = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    histograms: List[np.ndarray] = []
    for traj in trajectories:
        counts, _ = np.histogram(traj, bins=bin_edges)
        histograms.append(counts)
    
    hist_matrix: np.ndarray = np.array(histograms)  # (n_windows, n_bins)
    total_counts_per_bin: np.ndarray = np.sum(hist_matrix, axis=0)  # Total counts n_j

    # 4. Bias Energy Matrix (U_ij)
    beta: float = 1.0 / (k_B * temperature)
    # Using NumPy broadcasting for efficiency
    dist_sq: np.ndarray = (bin_centers[np.newaxis, :] - centers[:, np.newaxis]) ** 2
    u_ij: np.ndarray = 0.5 * force_constants[:, np.newaxis] * dist_sq

    # 5. Iterative Self-Consistent Solver
    # Initial guess for free energies (f_i)
    f_i: np.ndarray = np.log(frames_per_window).astype(float)
    converged: bool = False

    for i_iter in range(max_iter):
        # Calculate denominator: sum_i [ N_i * exp(f_i - beta * U_ij) ]
        exp_arg: np.ndarray = f_i[:, np.newaxis] - beta * u_ij
        exp_term: np.ndarray = np.exp(exp_arg)
        denom_j: np.ndarray = np.sum(frames_per_window[:, np.newaxis] * exp_term, axis=0)

        # Unbiased probability density (P_0)
        p_0: np.ndarray = np.zeros_like(total_counts_per_bin, dtype=float)
        valid_bins: np.ndarray = denom_j > 0
        p_0[valid_bins] = total_counts_per_bin[valid_bins] / denom_j[valid_bins]

        # Update f_i: f_i = -ln( sum_j [ P_0 * exp(-beta * U_ij) ] )
        # Log-Sum-Exp strategy for numerical stability
        f_i_new: np.ndarray = np.zeros(n_windows)
        for i in range(n_windows):
            log_weights: np.ndarray = np.log(p_0 + 1e-300) - (beta * u_ij[i, :])
            max_log: float = np.max(log_weights)
            f_i_new[i] = -(max_log + np.log(np.sum(np.exp(log_weights - max_log))))

        # Convergence check
        delta: float = np.max(np.abs(f_i_new - f_i))
        if verbose:
            print(f"Iteration {i_iter + 1}: Max delta f_i = {delta:.6e}")

        f_i = f_i_new
        if delta < tolerance:
            converged = True
            break

    # 6. PMF Profile Generation
    if verbose:
        if not converged:
            print(f"WARNING: WHAM did not converge. Final delta: {delta:.6e}")
        else:
            print(f"WHAM converged. Final delta: {delta:.6e}")

    # Re-calculate final P_0 with converged f_i
    exp_term_final: np.ndarray = np.exp(f_i[:, np.newaxis] - beta * u_ij)
    denom_final: np.ndarray = np.sum(frames_per_window[:, np.newaxis] * exp_term_final, axis=0)
    
    p_0_final: np.ndarray = np.zeros_like(denom_final)
    valid_final: np.ndarray = denom_final > 0
    p_0_final[valid_final] = total_counts_per_bin[valid_final] / denom_final[valid_final]

    # PMF formula: G(x) = -1/beta * ln(P_0)
    with np.errstate(divide='ignore', invalid='ignore'):
        pmf: np.ndarray = -(1.0 / beta) * np.log(p_0_final)

    # Clean non-finite results and zero-base the minimum
    pmf[~np.isfinite(pmf)] = np.nan
    if not np.all(np.isnan(pmf)):
        pmf -= np.nanmin(pmf)

    return bin_centers, pmf
    