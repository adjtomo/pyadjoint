#!/usr/bin/env python3
"""
Cross correlation traveltime misfit.

:copyright:
    Youyi Ruan (youyir@princeton.edu) 2016
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import absolute_import, division, print_function

import warnings

import numpy as np
from obspy.signal.cross_correlation import xcorr_pick_correction
from scipy.integrate import simps

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
:func:`~obspy.signal.cross_correlation.xcorr_pick_correction` function and the
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
        return xcorr_pick_correction(
            pick_time, s, pick_time, d, 20.0 * time_shift,
            20.0 * time_shift, 10.0 * time_shift)[0]


def calculate_adjoint_source(observed, synthetic, config, window,
                             adjoint_src, figure):  # NOQA

    ret_val_p = {}
    ret_val_q = {}

    nlen_data = len(synthetic.data)
    deltat = synthetic.stats.delta

    fp = np.zeros(nlen_data)
    fq = np.zeros(nlen_data)

    misfit_sum_p = 0.0
    misfit_sum_q = 0.0

    # ===
    # loop over time windows
    # ===
    for wins in window:
        left_window_border = wins[0]
        right_window_border = wins[1]

        left_sample = int(np.floor(left_window_border / deltat)) + 1
        nlen = int(np.floor((right_window_border -
                             left_window_border) / deltat)) + 1
        right_sample = left_sample + nlen

        d = np.zeros(nlen)
        s = np.zeros(nlen)

        d[0:nlen] = observed.data[left_sample:right_sample]
        s[0:nlen] = synthetic.data[left_sample:right_sample]

        # All adjoint sources will need some kind of windowing taper
        window_taper(d, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        window_taper(s, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)

        i_shift = _xcorr_shift(d, s)
        t_shift = i_shift * deltat

        cc_dlna = 0.5 * np.log(sum(d[0:nlen]*d[0:nlen]) /
                               sum(s[0:nlen]*s[0:nlen]))

        sigma_dt, sigma_dlna = cc_error(d, s, deltat, i_shift, cc_dlna,
                                        config.dt_sigma_min,
                                        config.dlna_sigma_min)

        misfit_sum_p += 0.5 * (t_shift/sigma_dt) ** 2
        misfit_sum_q += 0.5 * (cc_dlna/sigma_dlna) ** 2

        dsdt = np.gradient(s, deltat)
        nnorm = simps(y=dsdt*dsdt, dx=deltat)
        fp[left_sample:right_sample] = dsdt[:] * t_shift / nnorm / sigma_dt**2

        mnorm = simps(y=s*s, dx=deltat)
        fq[left_sample:right_sample] =\
            -1.0 * s[:] * cc_dlna / mnorm / sigma_dlna ** 2

    ret_val_p["misfit"] = misfit_sum_p
    ret_val_q["misfit"] = misfit_sum_q

    if adjoint_src is True:
        ret_val_p["adjoint_source"] = fp[::-1]
        ret_val_q["adjoint_source"] = fq[::-1]

    if config.measure_type == "dt":
        if figure:
            generic_adjoint_source_plot(observed, synthetic,
                                        ret_val_p["adjoint_source"],
                                        ret_val_p["misfit"],
                                        window, VERBOSE_NAME)

        return ret_val_p

    if config.measure_type == "am":
        if figure:
            generic_adjoint_source_plot(observed, synthetic,
                                        ret_val_q["adjoint_source"],
                                        ret_val_q["misfit"],
                                        window, VERBOSE_NAME)

        return ret_val_q
