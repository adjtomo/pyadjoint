#!/usr/bin/env python3
"""
Utility functions for calculating multitaper measurements (MTM).

:copyright:
    adjTomo Dev Team (adjtomo@gmail.com), 2022
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2016
    Martin Luessi (mluessi@nmr.mgh.harvard.edu), 2012
License : BSD 3-clause
"""
import warnings
import numpy as np
from scipy import fftpack, linalg, interpolate

from pyadjoint import logger


def process_cycle_skipping(phi_w, nfreq_max, nfreq_min, wvec, phase_step=1.5):
    """
    Check for cycle skipping by looking at the smoothness of phi

    :type phi_w: np.array
    :param phi_w: phase anomaly from transfer functions
    :type nfreq_min: int
    :param nfreq_min: minimum frequency for suitable MTM measurement
    :type nfreq_max: int
    :param nfreq_max: maximum frequency for suitable MTM measurement
    :type phase_step: float
    :param phase_step: maximum step for cycle skip correction (?)
    :type wvec: np.array
    :param wvec: angular frequency array generated from Discrete Fourier
        Transform sample frequencies
    """
    for iw in range(nfreq_min + 1, nfreq_max - 1):
        smth0 = abs(phi_w[iw + 1] + phi_w[iw - 1] - 2.0 * phi_w[iw])
        smth1 = \
            abs((phi_w[iw + 1] + 2 * np.pi) + phi_w[iw - 1] - 2.0 * phi_w[iw])
        smth2 = \
            abs((phi_w[iw + 1] - 2 * np.pi) + phi_w[iw - 1] - 2.0 * phi_w[iw])

        phase_diff = phi_w[iw] - phi_w[iw + 1]

        if abs(phase_diff) > phase_step:

            temp_period = 2.0 * np.pi / wvec[iw]

            if smth1 < smth0 and smth1 < smth2:
                logger.warning(f"2pi phase shift at {iw} T={temp_period} "
                               f"diff={phase_diff}")
                phi_w[iw + 1:nfreq_max] = phi_w[iw + 1:nfreq_max] + 2 * np.pi

            if smth2 < smth0 and smth2 < smth1:
                logger.warning(f"-2pi phase shift at {iw} T={temp_period} "
                               f"diff={phase_diff}")
                phi_w[iw + 1:nfreq_max] = phi_w[iw + 1:nfreq_max] - 2 * np.pi

    return phi_w


def tridisolve(d, e, b, overwrite_b=True):
    """
    Symmetric tridiagonal system solver, from Golub and Van Loan pg 157

    .. note::
        Copied from the mne-python package so credit goes to them.
        https://github.com/mne-tools/mne-python/blob/master/mne/time_frequency/\
        multitaper.py

    :type d: ndarray
    :param d: main diagonal stored in d[:]
    :type e: ndarray
    :param e: superdiagonal stored in e[:-1]
    :type b: ndarray
    :param b: RHS vector
    :rtype x : ndarray
    :return: Solution to Ax = b (if overwrite_b is False). Otherwise solution is
        stored in previous RHS vector b
    """
    n = len(b)
    # work vectors
    dw = d.copy()
    ew = e.copy()
    if overwrite_b:
        x = b
    else:
        x = b.copy()
    for k in range(1, n):
        # e^(k-1) = e(k-1) / d(k-1)
        # d(k) = d(k) - e^(k-1)e(k-1) / d(k-1)
        t = ew[k - 1]
        ew[k - 1] = t / dw[k - 1]
        dw[k] = dw[k] - t * ew[k - 1]
    for k in range(1, n):
        x[k] = x[k] - ew[k - 1] * x[k - 1]
    x[n - 1] = x[n - 1] / dw[n - 1]
    for k in range(n - 2, -1, -1):
        x[k] = x[k] / dw[k] - ew[k] * x[k + 1]

    if not overwrite_b:
        return x


def tridi_inverse_iteration(d, e, w, x0=None, rtol=1e-8):
    """
    Perform an inverse iteration to find the eigenvector corresponding
    to the given eigenvalue in a symmetric tridiagonal system.

    .. note::
        Copied from the mne-python package so credit goes to them.
        https://github.com/mne-tools/mne-python/blob/master/mne/time_frequency/\
        multitaper.py

    :type d: ndarray
    :param d: main diagonal stored in d[:]
    :type e: ndarray
    :param e: off diagonal stored in e[:-1]
    :type w: float
    :param w: eigenvalue of the eigenvector
    :type x0: ndarray
    :param x0: initial point to start the iteration
    :type rtol : float
    :param rtol: tolerance for the norm of the difference of iterates
    :rtype: ndarray
    :return: The converged eigenvector
    """
    eig_diag = d - w
    if x0 is None:
        x0 = np.random.randn(len(d))
    x_prev = np.zeros_like(x0)
    norm_x = np.linalg.norm(x0)
    # the eigenvector is unique up to sign change, so iterate
    # until || |x^(n)| - |x^(n-1)| ||^2 < rtol
    x0 /= norm_x
    while np.linalg.norm(np.abs(x0) - np.abs(x_prev)) > rtol:
        x_prev = x0.copy()
        tridisolve(eig_diag, e, x0)
        norm_x = np.linalg.norm(x0)
        x0 /= norm_x
    return x0


def sum_squared(x):
    """
    Compute norm of an array

    :type x: array
    :param x: Data whose norm must be found
    :rtype: float
    :return: Sum of squares of the input array X
    """
    x_flat = x.ravel(order='F' if np.isfortran(x) else 'C')
    return np.dot(x_flat, x_flat)


def dpss_windows(n, half_nbw, k_max, low_bias=True, interp_from=None,
                 interp_kind='linear'):
    """
    Returns the Discrete Prolate Spheroidal Sequences of orders [0,Kmax-1]
    for a given frequency-spacing multiple NW and sequence length N.

    .. note::
        Tridiagonal form of DPSS calculation from:

        Slepian, D. Prolate spheroidal wave functions, Fourier analysis, and
        uncertainty V: The discrete case. Bell System Technical Journal,
        Volume 57 (1978), 1371430

    .. note::
        This function was copied from NiTime

    :type n: int
    :param n: Sequence length
    :type half_nbw: float
    :param half_nbw: unitless standardized half bandwidth corresponding to
        2 * half_bw = BW*f0 = BW*N/dt but with dt taken as 1
    :type k_max: int
    :param k_max: Number of DPSS windows to return is Kmax
        (orders 0 through Kmax-1)
    :type low_bias: Bool
    :param low_bias: Keep only tapers with eigenvalues > 0.9
    :type interp_from: int (optional)
    :param interp_from: The dpss can be calculated using interpolation from a
        set of dpss with the same NW and Kmax, but shorter N. This is the
        length of this shorter set of dpss windows.
    :type interp_kind: str (optional)
    :param interp_kind: This input variable is passed to
        scipy.interpolate.interp1d and specifies the kind of interpolation as
        a string ('linear', 'nearest', 'zero', 'slinear', 'quadratic, 'cubic')
        or as an integer specifying the order of the spline interpolator to use.
    :rtype: tuple
    :return: (v, e), v is an array of DPSS windows shaped (Kmax, N),
        e are the eigenvalues
    """
    k_max = int(k_max)
    w_bin = float(half_nbw) / n
    nidx = np.arange(n, dtype='d')

    # In this case, we create the dpss windows of the smaller size
    # (interp_from) and then interpolate to the larger size (N)
    if interp_from is not None:
        if interp_from > n:
            e_s = 'In dpss_windows, interp_from is: %s ' % interp_from
            e_s += 'and N is: %s. ' % n
            e_s += 'Please enter interp_from smaller than N.'
            raise ValueError(e_s)
        dpss = []
        d, e = dpss_windows(interp_from, half_nbw, k_max, low_bias=False)
        for this_d in d:
            x = np.arange(this_d.shape[-1])
            x_interp = interpolate.interp1d(x, this_d, kind=interp_kind)
            d_temp = x_interp(np.arange(0, this_d.shape[-1] - 1,
                              float(this_d.shape[-1] - 1)/n))

            # Rescale:
            d_temp = d_temp / np.sqrt(sum_squared(d_temp))

            dpss.append(d_temp)

        dpss = np.array(dpss)

    else:
        # here we want to set up an optimization problem to find a sequence
        # whose energy is maximally concentrated within band [-W,W].
        # Thus, the measure lambda(T,W) is the ratio between the energy within
        # that band, and the total energy. This leads to the eigen-system
        # (A - (l1)I)v = 0, where the eigenvector corresponding to the largest
        # eigenvalue is the sequence with maximally concentrated energy. The
        # collection of eigenvectors of this system are called Slepian
        # sequences, or discrete prolate spheroidal sequences (DPSS). Only the
        # first K, K = 2NW/dt orders of DPSS will exhibit good spectral
        # concentration
        # [see http://en.wikipedia.org/wiki/Spectral_concentration_problem]

        # Here I set up an alternative symmetric tri-diagonal eigenvalue
        # problem such that
        # (B - (l2)I)v = 0, and v are our DPSS (but eigenvalues l2 != l1)
        # the main diagonal = ([N-1-2*t]/2)**2 cos(2PIW), t=[0,1,2,...,N-1]
        # and the first off-diagonal = t(N-t)/2, t=[1,2,...,N-1]
        # [see Percival and Walden, 1993]
        diagonal = ((n - 1 - 2*nidx)/2.)**2*np.cos(2*np.pi*w_bin)
        off_diag = np.zeros_like(nidx)
        off_diag[:-1] = nidx[1:] * (n - nidx[1:])/2.
        # put the diagonals in LAPACK "packed" storage
        ab = np.zeros((2, n), 'd')
        ab[1] = diagonal
        ab[0, 1:] = off_diag[:-1]
        # only calculate the highest Kmax eigenvalues
        w = linalg.eigvals_banded(ab, select='i',
                                  select_range=(n - k_max, n - 1))
        w = w[::-1]

        # find the corresponding eigenvectors via inverse iteration
        t = np.linspace(0, np.pi, n)
        dpss = np.zeros((k_max, n), 'd')
        for k in range(k_max):
            dpss[k] = tridi_inverse_iteration(diagonal, off_diag, w[k],
                                              x0=np.sin((k + 1) * t))

    # By convention (Percival and Walden, 1993 pg 379)
    # * symmetric tapers (k=0,2,4,...) should have a positive average.
    # * antisymmetric tapers should begin with a positive lobe
    fix_symmetric = (dpss[0::2].sum(axis=1) < 0)
    for i, f in enumerate(fix_symmetric):
        if f:
            dpss[2 * i] *= -1
    fix_skew = (dpss[1::2, 1] < 0)
    for i, f in enumerate(fix_skew):
        if f:
            dpss[2 * i + 1] *= -1

    # Now find the eigenvalues of the original spectral concentration problem
    # Use the autocorr sequence technique from Percival and Walden, 1993 pg 390

    # compute autocorr using FFT (same as nitime.utils.autocorr(dpss) * N)
    rxx_size = 2*n - 1
    n_fft = 2 ** int(np.ceil(np.log2(rxx_size)))
    dpss_fft = fftpack.fft(dpss, n_fft)
    dpss_rxx = np.real(fftpack.ifft(dpss_fft * dpss_fft.conj()))
    dpss_rxx = dpss_rxx[:, :n]

    r = 4 * w_bin * np.sinc(2 * w_bin * nidx)
    r[0] = 2 * w_bin
    eigvals = np.dot(dpss_rxx, r)

    if low_bias:
        idx = (eigvals > 0.9)
        if not idx.any():
            warnings.warn('Could not properly use low_bias, '
                          'keeping lowest-bias taper')
            idx = [np.argmax(eigvals)]
        dpss, eigvals = dpss[idx], eigvals[idx]
    assert len(dpss) > 0  # should never happen
    return dpss, eigvals
