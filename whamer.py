# === whamer.py ===
"""
This script contains the WHAM implementation.
Use the script at your convenience.

Alejandro H. Tanguma
"""
import numpy as np

def wham(trajectories: np.ndarray, centers: list, force_constants: list, temperature: float = 300,
                    k_B: float = 0.0083144621, n_bins: int = 200, x_range: tuple = None,
                    tolerance: float = 1.0e-06, max_iter: int = 10000, verbose: bool = False) -> tuple:
    """
    Computes Potential of Mean Force through WHAM method.
    Redefine units by changing k_B and temperature values.
    Default units for the PMF are kJ/mol

    Parameters:
        trajectories : array or list of 1D NumPy arrays 
            Each element is a sampled window.
        centers : 1D NumPy array
            Position of the bias potentials. One per window.
        force_constants : 1D array
            Force constants of the bias potentials. One per window.
            Same units as k_B (kJ*mol^-1*K^-1).
        temperature : float
            Simulation temperature. Default: 300 K
        k_B : float
            Boltzmann constant. Same units as force constants.
            Default value: 0.0083144621 kJ*mol^-1*K^-1
        n_bins : int
            Number of bins for WHAM. Default: 200
        x_range : (float, float) o None
            Range (min, max) for creating bins.
            If None, the min and max is computed from all the trajectories.
            Default: None
        tolerance : float
            Convergence tolerance. Default: 1.0e-06
        max_iter : int
            Maximum number of iterations. Default: 10000
        verbose : bool
            If True, prints maximum change in each iteration.
            Default: False

    Return:
        bin_centers : numpy.ndarray
            Center of the bins. len(bin_centers) == n_bins
        pmf : numpy.ndarray
            PMF profile in same units as k_B.
    """
    # Verify length of the arrays
    n_windows = len(trajectories)
    if len(centers) != n_windows or len(force_constants) != n_windows:
        raise ValueError("Length of 'centers' and 'force_constants' must be equal to number of windows.")

    # Frames per window
    N_i = np.array([len(traj) for traj in trajectories])

    # Range of the bins
    if x_range is None:
        all_data = np.concatenate(trajectories)
        x_min, x_max = np.min(all_data), np.max(all_data)
        # Avoid the margins to be out of the pmf
        margin = 0.05 * (x_max - x_min)
        x_min -= margin
        x_max += margin
    else:
        x_min, x_max = x_range

    # Create bins and compute histograms
    bin_edges = np.linspace(x_min, x_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    histograms = []
    for traj in trajectories:
        hist, _ = np.histogram(traj, bins=bin_edges)
        histograms.append(hist)
    histograms = np.array(histograms)  # (n_windows, n_bins)

    # Bias energies
    # U_i(ξ) = 0.5 * k_i * (ξ - center_i)^2
    beta = 1.0 / (k_B * temperature)
    U_ij = np.zeros((n_windows, n_bins))
    for i in range(n_windows):
        U_ij[i, :] = 0.5 * force_constants[i] * (bin_centers - centers[i]) ** 2

    # First estimation of f_i
    f_i = np.log(N_i)
    
    # WHAM iteration
    converged = False
    for iteration in range(max_iter):
        # 1) Common denominator per bin.
        # denom_j = sum_i N_i * exp(f_i - beta * U_ij)
        # exponentials helps with numerical stability
        exp_arg = f_i[:, np.newaxis] - beta * U_ij  # shape (n_windows, n_bins)
        exp_term = np.exp(exp_arg)  # (n_windows, n_bins)
        denom_j = np.sum(N_i[:, np.newaxis] * exp_term, axis=0)  # (n_bins,)

        # 2) Compute P_0 
        # P_0(j) = (sum_i n_i(j)) / denom_j
        n_total_j = np.sum(histograms, axis=0)  # (n_bins,)
        # Avoid ZeroDivisionError
        P_0 = np.zeros_like(n_total_j, dtype=float)
        nonzero = denom_j > 0
        P_0[nonzero] = n_total_j[nonzero] / denom_j[nonzero]

        # 3) Update f_i
        # f_i_new = -log( sum_j P_0(j) * exp(-beta * U_ij) )
        f_i_new = np.zeros(n_windows)
        for i in range(n_windows):
            log_w_ij = np.log(P_0 + 1e-300) - beta * U_ij[i, :]
            max_log = np.max(log_w_ij)
            sum_exp = np.sum(np.exp(log_w_ij - max_log))
            # f_i = -log( sum ) = -( max_log + log(sum_exp) )
            f_i_new[i] = -(max_log + np.log(sum_exp))

        # 4) Maximum change
        delta = np.max(np.abs(f_i_new - f_i))
        if verbose:
            print(f"Iteration {iteration+1}, max change f_i: {delta:.6e}")
        f_i = f_i_new
        if delta < tolerance:
            converged = True
            break

    if not converged and verbose:
        print(f"WARNING: WHAM did not converge after {max_iter} iterations. Max change: {delta:.6e}, tolerance: {tolerance:.6e}")
    else:
        print(f"WHAM converged :)  change: {delta:.6e} <= tol: {tolerance:.6e}")

    # Compute P_0 with converged f_i
    exp_term = np.exp(f_i[:, np.newaxis] - beta * U_ij)
    denom_j = np.sum(N_i[:, np.newaxis] * exp_term, axis=0)
    P_0 = np.zeros_like(denom_j)
    nonzero = denom_j > 0
    P_0[nonzero] = n_total_j[nonzero] / denom_j[nonzero]

    # PMF: A(ξ) = -k_B T * log(P_0) + C
    # Set minimum to 0
    with np.errstate(divide='ignore', invalid='ignore'):
        pmf = - (1.0 / beta) * np.log(P_0)
    # divergences are presented as nan
    pmf[~np.isfinite(pmf)] = np.nan
    min_val = np.nanmin(pmf)
    pmf -= min_val

    return bin_centers, pmf



