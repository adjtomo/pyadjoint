#!/usr/bin/env python
# -*- encoding: utf8 -*-
"""
Cross correlation traveltime misfit.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
    Youyi Ruan (youyir@princeton.edu) 2016
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from obspy.signal.cross_correlation import xcorrPickCorrection
import numpy as np
from scipy.integrate import simps
import warnings

from ..utils import generic_adjoint_source_plot, taper_window


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

def cc_error(d1, d2, deltat, cc_shift, cc_dlna):
    nlen_T = len(d1)

    # make cc-based corrections to d2
    d2_cc = np.zeros(nlen_T)
    for index in range(0, nlen_T):
        index_shift = index - cc_shift
        if 0 <= index_shift < nlen_T:
            d2_cc[index] = np.exp(cc_dlna) * d2[index_shift]

    # velocity of d2_cc
    # d2_cc_vel = np.zeros(nlen_T)
    d2_cc_vel = np.gradient(d2_cc) / deltat

    # the estimated error for dt and dlna with uncorrelation assumption
    sigma_dt_top = np.sum((d1[1:nlen_T] - d2_cc[1:nlen_T]) * 
                          (d1[1:nlen_T] - d2_cc[1:nlen_T]) )
    sigma_dt_bot = np.sum(d2_cc_vel[1:nlen_T] * d2_cc_vel[1:nlen_T])

    sigma_dlna_top = sigma_dt_top
    sigma_dlna_bot = np.sum(d2_cc[1:nlen_T] * d2_cc[1:nlen_T]) / (cc_dlna * cc_dlna)

    sigma_dt = np.sqrt(sigma_dt_top / sigma_dt_bot)
    sigma_dlna = np.sqrt(sigma_dlna_top / sigma_dlna_bot)

    return sigma_dt, sigma_dlna

def subsample_xcorr_shift(d, s):
    """
    Calculate the correlation time shift around the maximum amplitude of the
    synthetic trace with subsample accuracy.
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


#def calculate_adjoint_source(observed, synthetic, min_period, max_period,
#                             left_window_border, right_window_border,
#                             adjoint_src, figure, taper_percentage=0.15,
#                             taper_type="hann"):  # NOQA
def calculate_adjoint_source(observed, synthetic, config, window,
                             adjoint_src, figure):  # NOQA


    # All adjoint sources will need some kind of windowing taper. Thus pyadjoint has a
    # convenience function to assist with that. The next block tapers both
    # observed and synthetic data.

    #taper_window(observed, left_window_border, right_window_border,
    #             taper_percentage=taper_percentage, taper_type=taper_type)
    #taper_window(synthetic, left_window_border, right_window_border,
    #             taper_percentage=taper_percentage, taper_type=taper_type)

    ret_val_p = {}
    ret_val_q = {}

    nlen_data = len(synthetic.data)
    deltat = synthetic.stats.delta

    fp = np.zeros(nlen_data)
    fq = np.zeros(nlen_data)

    misfit_sum_p = 0.0
    misfit_sum_q = 0.0
    
    #===
    # loop over time windows
    #===
    for wins in window:
        
        left_window_border  = wins[0]
        right_window_border = wins[1]

        left_sample  = int(np.floor( left_window_border / deltat)) + 1
        nlen         = int(np.floor((right_window_border - left_window_border) / deltat)) + 1
        right_sample = left_sample + nlen

        d = np.zeros(nlen)
        s = np.zeros(nlen)

        d[0: nlen] =  observed.data[left_sample: right_sample]
        s[0: nlen] = synthetic.data[left_sample: right_sample]

        
        # All adjoint sources will need some kind of windowing taper
        # to get rid of kinks at two ends
        sac_hann_taper(d, taper_percentage=config.taper_percentage)
        sac_hann_taper(s, taper_percentage=config.taper_percentage)


        # Subsample accuracy time shift
        time_shift = subsample_xcorr_shift(observed, synthetic)
        cc_dlnA = 0.5 * np.log(sum(d[0:nlen]*d[0:nlen]) / sum(s[0:nlen]*s[0:nlen]))

        misfit_sum_p += 0.5 * time_shift ** 2
        misfit_sum_q += 0.5 * cc_dlnA ** 2

        # original code by Lion
        #s_dt = synthetic.copy().differentiate()
        #s_dt_2 = s_dt.copy().differentiate()
        #N = simps(y=synthetic.data * s_dt_2.data, dx=synthetic.stats.delta)
    
        dsdt    = np.gradient(s) / deltat

        # Reverse in time and reverse the actual values.
        nnorm   = simps(y=dsdt*dsdt, dx=deltat)
        fp = -1.0 * (time_shift / nnorm * s_dt.data)


    if adjoint_src is True:
        ret_val_p["misfit"] = misfit_sum_p
        ret_val_q["misfit"] = misfit_sum_q

        ret_val_p["adjoint_source"] = fp[::-1]
        ret_val_q["adjoint_source"] = fq[::-1]


    if figure:
        generic_adjoint_source_plot(
            observed, synthetic, ret_val["adjoint_source"], ret_val["misfit"],
            left_window_border, right_window_border,
            VERBOSE_NAME)

    if config.measure_type == "dt":
        return ret_val_p

    if config.measure_type == "am":
        return ret_val_q
