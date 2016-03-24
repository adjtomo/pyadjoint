#!/usr/bin/env python
# -*- encoding: utf8 -*-
"""
Gallery of functions to be used in Cross correlation approach.

:copyright:
    Yanhua O. Yuan(yanhuay@princeton.edu) 2016
    Youyi Ruan (youyir@princeton.edu) 2016
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from obspy.signal.cross_correlation import xcorrPickCorrection
import numpy as np
from scipy.integrate import simps
import warnings

VERBOSE_NAME = "Cross Correlation Traveltime Misfit"

DESCRIPTION = r"""
Traveltime misfits simply measure the squared traveltime difference. The
misfit :math:`\chi(\mathbf{m})` for a given Earth model :math:`\mathbf{m}`
and a single receiver and component is given by

.. math::

    \chi (\mathbf{m}) = \frac{1}{2} \left[ T^{obs} - T(\mathbf{m}) \right] ^ 2

:math:`T^{obs}` is the observed traveltime, and :math:`T(\mathbf{m})` the
predicted traveltime in Earth model :math:`\mathbf{m}`.

In practice traveltime are measured by cross correlating observed and
predicted waveforms. This particular implementation here measures cross
correlation time shifts with subsample accuracy with a fitting procedure
explained in [Deichmann1992]_. For more details see the documentation of the
:func:`~obspy.signal.cross_correlation.xcorrPickCorrection` function and the
corresponding
`Tutorial <http://docs.obspy.org/tutorial/code_snippets/xcorr_pick_correction.html>`_.


The adjoint source for the same receiver and component is then given by

.. math::

    f^{\dagger}(t) = - \left[ T^{obs} - T(\mathbf{m}) \right] ~ \frac{1}{N} ~
    \partial_t \mathbf{s}(T - t, \mathbf{m})

For the sake of simplicity we omit the spatial Kronecker delta and define
the adjoint source as acting solely at the receiver's location. For more
details, please see [Tromp2005]_ and [Bozdag2011]_.


:math:`N` is a normalization factor given by


.. math::

    N = \int_0^T ~ \mathbf{s}(t, \mathbf{m}) ~
    \partial^2_t \mathbf{s}(t, \mathbf{m}) dt

This particular implementation here uses
`Simpson's rule <http://en.wikipedia.org/wiki/Simpson's_rule>`_
to evaluate the definite integral.
"""  # NOQA

# Optional: document any additional parameters this particular adjoint sources
# receives in addition to the ones passed to the central adjoint source
# calculation function. Make sure to indicate the default values. This is a
# bit redundant but the only way I could figure out to make it work with the
#  rest of the architecture.
ADDITIONAL_PARAMETERS = r"""
**taper_percentage** (:class:`float`)
    Decimal percentage of taper at one end (ranging from ``0.0`` (0%) to
    ``0.5`` (50%)). Defauls to ``0.15``.

**taper_type** (:class:`float`)
    The taper type, supports anything :meth:`obspy.core.trace.Trace.taper`
    can use. Defaults to ``"hann"``.
"""


def _xcorr_shift(d, s):
    cc = np.correlate(d, s, mode="full")
    time_shift = cc.argmax() - len(d) + 1
    return time_shift


def cc_correction(d, cc_shift, cc_dlna):
    """  correct d by shifting cc_shift and scaling exp(cc_dlna)
    """

    nlen_t = len(d)
    d_cc = np.zeros(nlen_t)

    for _ind in range(0, nlen_t):
        ind_dt = _ind - cc_shift

        if 0 <= ind_dt < nlen_t:
            d_cc[_ind] = d[ind_dt] * np.exp(cc_dlna)

    return d_cc


def cc_error(d1, d2, deltat, cc_shift, cc_dlna, sigma_dt_min, sigma_dlna_min):
    """
    Estimate error for dt and dlna with uncorrelation assumption
    """
    nlen_t = len(d1)

    d2_cc_dt = np.zeros(nlen_t)
    d2_cc_dtdlna = np.zeros(nlen_t)

    for index in range(0, nlen_t):
        index_shift = index - cc_shift

        if 0 <= index_shift < nlen_t:
            # corrected by c.c. shift
            d2_cc_dt[index] = d2[index_shift]

            # corrected by c.c. shift and amplitude
            d2_cc_dtdlna[index] = np.exp(cc_dlna) * d2[index_shift]

    # time derivative of d2_cc (velocity)
    d2_cc_vel = np.gradient(d2_cc_dtdlna, deltat)

    # the estimated error for dt and dlna with uncorrelation assumption
    sigma_dt_top = np.sum((d1 - d2_cc_dtdlna)**2)
    sigma_dt_bot = np.sum(d2_cc_vel**2)

    sigma_dlna_top = sigma_dt_top
    sigma_dlna_bot = np.sum(d2_cc_dt**2)

    sigma_dt = np.sqrt(sigma_dt_top / sigma_dt_bot)
    sigma_dlna = np.sqrt(sigma_dlna_top / sigma_dlna_bot)

    if sigma_dt < sigma_dt_min:
        sigma_dt = sigma_dt_min

    if sigma_dlna < sigma_dlna_min:
        sigma_dlna = sigma_dlna_min

    return sigma_dt, sigma_dlna


def subsample_xcorr_shift(d, s):
    """
    Calculate the correlation time shift around the maximum amplitude of the
    synthetic trace with subsample accuracy.
    :param s:
    :param d:
    """
    # Estimate shift and use it as a guideline for the subsample accuracy
    # shift.
    time_shift = _xcorr_shift(d.data, s.data) * d.stats.delta

    # Align on the maximum amplitude of the synthetics.
    pick_time = s.stats.starttime + s.data.argmax() * s.stats.delta

    # Will raise a warning if the trace ids don't match which we don't care
    # about here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return xcorrPickCorrection(
            pick_time, s, pick_time, d, 20.0 * time_shift,
            20.0 * time_shift, 10.0 * time_shift)[0]


def cc_adj(synt, cc_shift, cc_dlna, deltat, err_dt_cc, err_dlna_cc):
    """
    cross correlation adjoint source and misfit
    """

    misfit_p = 0.0
    misfit_q = 0.0

    dsdt = np.gradient(synt) / deltat

    nnorm = simps(y=dsdt*dsdt, dx=deltat)
    dt_adj = cc_shift * deltat / err_dt_cc**2 / nnorm * dsdt

    mnorm = simps(y=synt*synt, dx=deltat)
    am_adj = -1.0 * cc_dlna / err_dlna_cc**2 / mnorm * synt

    cc_tshift = cc_shift * deltat
    misfit_p = 0.5 * (cc_tshift/err_dt_cc)**2
    misfit_q = 0.5 * (cc_dlna/err_dlna_cc)**2

    return dt_adj, am_adj, misfit_p, misfit_q


def cc_adj_DD(synt1, synt2, shift_syn, dd_shift, deltat,
              err_dt_cc=1.0, err_dlna_cc=1.0):
    """
    double-difference cross correlation adjoint source and misfit
    Y. Yuan
    need a pair of syntheticis
    return a pair of adjoint sources
    """

    misfit = 0.0
    ds1dt = np.gradient(synt1, deltat) / deltat
    ds2dt = np.gradient(synt2, deltat) / deltat
    synt2_cc = cc_correction(synt2, shift_syn, 0.0)
    ds2_ccdt = np.gradient(synt2_cc, deltat) / deltat
    nnorm12 = simps(y=ds1dt*ds2_ccdt, dx=deltat)

    dt_adj1 = - 1.0 * ds2dt * dd_shift * deltat / nnorm12
    dt_adj2 = + 1.0 * ds1dt * dd_shift * deltat / nnorm12

    dd_tshift = dd_shift * deltat
    misfit = 0.5 * (dd_tshift/err_dt_cc)**2

    return dt_adj1, dt_adj2, misfit
