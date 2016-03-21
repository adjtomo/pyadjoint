#!/usr/bin/env python
# -*- encoding: utf8 -*-
"""
Double-Difference Cross correlation traveltime misfit.

:copyright:
    Yanhua O. Yuan(yanhuay@princeton.edu) 2016
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from obspy.signal.cross_correlation import xcorrPickCorrection
import numpy as np
from scipy.integrate import simps
import warnings

from ..utils import window_taper,  generic_adjoint_source_plot

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
details, please see [Yuan2016].


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


def cc_correction(d, cc_shift, cc_dlna):
    """  correct d by shifting cc_shift and scaling exp(cc_dlna)
    """

    nlen_t = len(d)
    d_cc = np.zeros(nlen_t)

    for _ind in range(0, nlen_t):
        ind_dt = _ind - cc_shift

        if 0 <= ind_dt < nlen_t:
            d_cc[ind_dt] = d[_ind] * np.exp(cc_dlna)

    return d_cc


def _xcorr_shift(d, s):
    cc = np.correlate(d, s, mode="full")
    time_shift = cc.argmax() - len(d) + 1
    return time_shift


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


def calculate_adjoint_source_DD(observed1, synthetic1, observed2, synthetic2,
                             config, window1, window2,
                             adjoint_src, figure):  # NOQA

    ret_val_p1 = {}
    ret_val_p2 = {}

    nlen_data = len(synthetic1.data)
    deltat = synthetic1.stats.delta

    fp1 = np.zeros(nlen_data)
    fp2 = np.zeros(nlen_data)

    misfit_sum_p = 0.0

    # ===
    # loop over time windows
    # ===
    for wins1, wins2 in zip(window1, window2):
        left_window_border_1 = wins1[0]
        right_window_border_1 = wins1[1]
        left_window_border_2 = wins2[0]
        right_window_border_2 = wins2[1]

        left_sample_1 = int(np.floor(left_window_border_1 / deltat)) + 1
        left_sample_2 = int(np.floor(left_window_border_2 / deltat)) + 1
        nlen1 = int(np.floor((right_window_border_1 -
                             left_window_border_1) / deltat)) + 1
        nlen2 = int(np.floor((right_window_border_2 -
                             left_window_border_2) / deltat)) + 1

        if(nlen1 != nlen2):
            continue
        nlen = nlen1

        right_sample_1 = left_sample_1 + nlen
        right_sample_2 = left_sample_2 + nlen

        d1 = np.zeros(nlen)
        s1 = np.zeros(nlen)
        d2 = np.zeros(nlen)
        s2 = np.zeros(nlen)

        d1[0:nlen] = observed1.data[left_sample_1:right_sample_1]
        s1[0:nlen] = synthetic1.data[left_sample_1:right_sample_1]
        d2[0:nlen] = observed2.data[left_sample_2:right_sample_2]
        s2[0:nlen] = synthetic2.data[left_sample_2:right_sample_2]

        # All adjoint sources will need some kind of windowing taper
        window_taper(d1, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        window_taper(s1, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        window_taper(d2, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        window_taper(s2, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)

        i_shift_obs = _xcorr_shift(d1, d2)
        i_shift_syn = _xcorr_shift(s1, s2)
        t_shift_DD = (i_shift_syn - i_shift_obs) * deltat

        misfit_sum_p += 0.5 * (t_shift_DD) ** 2

        ds1dt = np.gradient(s1, deltat)
        ds2dt = np.gradient(s2, deltat)
        s2_cc = cc_correction(s2, i_shift_syn, 0.0)
        ds2_ccdt = np.gradient(s2_cc, deltat)
        nnorm12 = simps(y=ds1dt*ds2_ccdt, dx=deltat)

        # taper adjoint source again
        window_taper(ds2dt, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        window_taper(ds1dt, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)

        # sum
        fp1[left_sample_1:right_sample_1] = \
            + 1.0 * ds2dt[:] * t_shift_DD / nnorm12
        fp2[left_sample_2:right_sample_2] = \
            - 1.0 * ds1dt[:] * t_shift_DD / nnorm12

    ret_val_p1["misfit"] = misfit_sum_p
    ret_val_p2["misfit"] = misfit_sum_p

    if adjoint_src is True:
        ret_val_p1["adjoint_source"] = fp1[::-1]
        ret_val_p2["adjoint_source"] = fp2[::-1]

    if config.measure_type == "dt":
        if figure:
            generic_adjoint_source_plot(observed1, synthetic1,
                                        ret_val_p1["adjoint_source"],
                                        ret_val_p1["misfit"],
                                        window1, VERBOSE_NAME)
            generic_adjoint_source_plot(observed2, synthetic2,
                                        ret_val_p2["adjoint_source"],
                                        ret_val_p2["misfit"],
                                        window2, VERBOSE_NAME)

        # ===
        # YY: only return adjoint source of
        # master trace to match adjoint_source wrapper
        # ===
        return ret_val_p1
