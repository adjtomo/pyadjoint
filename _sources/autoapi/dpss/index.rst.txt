:py:mod:`dpss`
==============

.. py:module:: dpss

.. autoapi-nested-parse::

   Utility functions for calculating multitaper measurements (MTM).
   Mainly contains functions for calculating Discrete Prolate Spheroidal Sequences
   (DPSS)

   :copyright:
       adjTomo Dev Team (adjtomo@gmail.com), 2022
       Lion Krischer (krischer@geophysik.uni-muenchen.de), 2016
       Martin Luessi (mluessi@nmr.mgh.harvard.edu), 2012
   License : BSD 3-clause



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   dpss.tridisolve
   dpss.tridi_inverse_iteration
   dpss.sum_squared
   dpss.dpss_windows



.. py:function:: tridisolve(d, e, b, overwrite_b=True)

   Symmetric tridiagonal system solver, from Golub and Van Loan pg 157

   .. note::
       Copied from the mne-python package so credit goes to them.
       https://github.com/mne-tools/mne-python/blob/master/mne/time_frequency/        multitaper.py

   :type d: ndarray
   :param d: main diagonal stored in d[:]
   :type e: ndarray
   :param e: superdiagonal stored in e[:-1]
   :type b: ndarray
   :param b: RHS vector
   :rtype x : ndarray
   :return: Solution to Ax = b (if overwrite_b is False). Otherwise solution is
       stored in previous RHS vector b


.. py:function:: tridi_inverse_iteration(d, e, w, x0=None, rtol=1e-08)

   Perform an inverse iteration to find the eigenvector corresponding
   to the given eigenvalue in a symmetric tridiagonal system.

   .. note::
       Copied from the mne-python package so credit goes to them.
       https://github.com/mne-tools/mne-python/blob/master/mne/time_frequency/        multitaper.py

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


.. py:function:: sum_squared(x)

   Compute norm of an array

   :type x: array
   :param x: Data whose norm must be found
   :rtype: float
   :return: Sum of squares of the input array X


.. py:function:: dpss_windows(n, half_nbw, k_max, low_bias=True, interp_from=None, interp_kind='linear')

   Returns the Discrete Prolate Spheroidal Sequences of orders [0,Kmax-1]
   for a given frequency-spacing multiple NW and sequence length N.
   Rayleigh bin parameter typical values of half_nbw/nw are 2.5,3,3.5,4.

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


