#!/usr/bin/env python
# -*- encoding: utf8 -*-
"""
Double-Difference multitaper misfit.

:copyright:
    Yanhua O. Yuan(yanhuay@princeton.edu) 2016
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from pyadjoint.dpss import dpss_windows

from pyadjoint.utils import window_taper,  generic_adjoint_source_plot
from pyadjoint.cc import _xcorr_shift, cc_error, cc_adj_DD
from pyadjoint.multitaper import frequency_limit, mt_measure,\
        mt_error, mt_measure_select

VERBOSE_NAME = "Double-difference Multitaper Misfit"

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


def calculate_adjoint_source_DD(observed1, synthetic1, observed2, synthetic2,
                                config, window1, window2,
                                adjoint_src, figure):  # NOQA

    # frequencies points for FFT
    nlen_f = 2**config.lnpt

    # constant for transfer function
    waterlevel_mtm = config.transfunc_waterlevel
    wtr = config.water_threshold

    # constant for cycle skip correction
    phase_step = config.phase_step

    # for frequency limit calculation
    ncycle_in_window = config.min_cycle_in_window

    # error estimation method
    use_cc_error = config.use_cc_error
    # use_mt_error = config.use_mt_error

    # Frequency range for adjoint src
    min_period = config.min_period
    max_period = config.max_period

    # critiaria for rejecting mtm measurements
    dt_fac = config.dt_fac
    err_fac = config.err_fac
    dt_max_scale = config.dt_max_scale

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

        # cross-correlation
        cc_shift_d = _xcorr_shift(d1, d2)
        cc_tshift_d = cc_shift_d * deltat
        cc_dlna_d = 0.5 * np.log(sum(d1**2) / sum(d2**2))
        cc_shift_s = _xcorr_shift(s1, s2)
        cc_tshift_s = cc_shift_s * deltat
        cc_dlna_s = 0.5 * np.log(sum(s1**2) / sum(s2**2))
        # double-difference cc-measurement
        dd_shift = cc_shift_s - cc_shift_d

        # uncertainty estimate based on cross-correlations
        sigma_dt_cc_d = 1.0
        sigma_dlna_cc_d = 1.0
        sigma_dt_cc_s = 1.0
        sigma_dlna_cc_s = 1.0

        if use_cc_error:
            sigma_dt_cc_d, sigma_dlna_cc_d = \
                cc_error(d1, d2, deltat, cc_shift_d,
                         cc_dlna_d, config.dt_sigma_min,
                         config.dlna_sigma_min)
            sigma_dt_cc_s, sigma_dlna_cc_s = \
                cc_error(s1, s2, deltat, cc_shift_s,
                         cc_dlna_s, config.dt_sigma_min,
                         config.dlna_sigma_min)

        # re-window observed1 to align observed1 with observed2
        # for multitaper measurement:
        # left_sample_1 = max(left_sample_1 + cc_shift_d, 0)
        # right_sample_1 = min(right_sample_1 + cc_shift_d, nlen_data)
        # nlen1 = right_sample_1 - left_sample_1

        # if nlen1 == nlen:
        #     d1[0:nlen] = observed1.data[left_sample_1:right_sample_1]
        #     d1 *= np.exp(-cc_dlna_d)
        #     window_taper(d1, taper_percentage=config.taper_percentage,
        #                  taper_type=config.taper_type)
        # else:
        #     raise Exception

        # re-window synthetic1 to align synthetic1 with synthetic2
        # for multitaper measurement:
        # left_sample_1 = max(left_sample_1 + cc_shift_s, 0)
        # right_sample_1 = min(right_sample_1 + cc_shift_s, nlen_data)
        # nlen1 = right_sample_1 - left_sample_1

        # YY: do we need to shift s1?
        # if nlen1 == nlen:
        #     s1[0:nlen] = synthetic1.data[left_sample_1:right_sample_1]
        #     s1 *= np.exp(-cc_dlna_s)
        #     window_taper(s1, taper_percentage=config.taper_percentage,
        #                  taper_type=config.taper_type)
        # else:
        #     raise Exception

        # ===
        # Make decision wihich method to use: c.c. or multi-taper
        # always starts from multi-taper, if it doesn't work then
        # switch to cross correlation misfit
        # ===
        is_mtm1 = True
        is_mtm2 = True
        is_mtm_d = True
        is_mtm_s = True

        # frequencies for FFT
        freq = np.fft.fftfreq(n=nlen_f, d=observed1.stats.delta)
        df = freq[1] - freq[0]
        wvec = freq * 2 * np.pi
        # todo: check again see if dw is not used.
        # dw = wvec[1] - wvec[0]

        # check window if okay for mtm measurements, and then find min/max
        # frequency limit for calculations.
        nfreq_min1, nfreq_max1, is_mtm1 = \
            frequency_limit(s1, nlen, nlen_f, deltat, df, wtr,
                            ncycle_in_window, min_period, max_period,
                            config.mt_nw)
        nfreq_min2, nfreq_max2, is_mtm1 = \
            frequency_limit(s2, nlen, nlen_f, deltat, df, wtr,
                            ncycle_in_window, min_period, max_period,
                            config.mt_nw)
        nfreq_min = max(nfreq_min1, nfreq_min2)
        nfreq_max = max(nfreq_max1, nfreq_max2)

        if is_mtm1 and is_mtm2:
            # Set the Rayleigh bin parameter (determin taper bandwithin
            # frequency domain): nw (typical values are 2.5,3,3.5,4).
            nw = config.mt_nw
            ntaper = config.num_taper

            # generate discrete prolate slepian sequences
            tapers = dpss_windows(nlen, nw, ntaper)[0].T

            # normalization
            tapers = tapers * np.sqrt(nlen)

            # measure frequency-dependent phase and amplitude difference
            phi_mtm_d = np.zeros(nlen_f)
            abs_mtm_d = np.zeros(nlen_f)
            phi_mtm_d, abs_mtm_d, dtau_mtm_d, dlna_mtm_d =\
                mt_measure(d1, d2, deltat, tapers, wvec, df, nlen_f,
                           waterlevel_mtm, phase_step, nfreq_min, nfreq_max,
                           cc_tshift_d, cc_dlna_d)

            phi_mtm_s = np.zeros(nlen_f)
            abs_mtm_s = np.zeros(nlen_f)
            phi_mtm_s, abs_mtm_s, dtau_mtm_s, dlna_mtm_s =\
                mt_measure(s1, s2, deltat, tapers, wvec, df, nlen_f,
                           waterlevel_mtm, phase_step, nfreq_min, nfreq_max,
                           cc_tshift_s, cc_dlna_s)

            # multi-taper error estimation
            sigma_phi_mt_d = np.zeros(nlen_f)
            sigma_abs_mt_d = np.zeros(nlen_f)
            sigma_dtau_mt_d = np.zeros(nlen_f)
            sigma_dlna_mt_d = np.zeros(nlen_f)

            sigma_phi_mt_d, sigma_abs_mt_d, \
                sigma_dtau_mt_d, sigma_dlna_mt_d =\
                mt_error(d1, d2, deltat, tapers, wvec, df, nlen_f,
                         waterlevel_mtm, phase_step, nfreq_min, nfreq_max,
                         cc_tshift_d, cc_dlna_d, phi_mtm_d, abs_mtm_d,
                         dtau_mtm_d, dlna_mtm_d)

            sigma_phi_mt_s = np.zeros(nlen_f)
            sigma_abs_mt_s = np.zeros(nlen_f)
            sigma_dtau_mt_s = np.zeros(nlen_f)
            sigma_dlna_mt_s = np.zeros(nlen_f)

            sigma_phi_mt_s, sigma_abs_mt_s, \
                sigma_dtau_mt_s, sigma_dlna_mt_s =\
                mt_error(s1, s2, deltat, tapers, wvec, df, nlen_f,
                         waterlevel_mtm, phase_step, nfreq_min, nfreq_max,
                         cc_tshift_s, cc_dlna_s, phi_mtm_s, abs_mtm_s,
                         dtau_mtm_s, dlna_mtm_s)

            # check is_mtm again if the multitaper measurement results failed
            # the selctuing criteria.  change is_mtm if it's not okay
            is_mtm_d = mt_measure_select(
                nfreq_min, nfreq_max, df, nlen, deltat,
                dtau_mtm_d, dt_fac, sigma_dtau_mt_d,
                err_fac, cc_tshift_d, dt_max_scale)
            is_mtm_s = mt_measure_select(
                nfreq_min, nfreq_max, df, nlen, deltat,
                dtau_mtm_s, dt_fac, sigma_dtau_mt_s,
                err_fac, cc_tshift_s, dt_max_scale)

        # final decision which misfit will be used for adjoint source.
        if is_mtm1 and is_mtm2 and is_mtm_d and is_mtm_s:
            # calculate multi-taper adjoint source
            dt_adj1, dt_adj2, misfit_dt = cc_adj_DD(
                s1, s2, cc_shift_s, dd_shift, deltat)
        else:
            # calculate c.c. adjoint source
            dt_adj1, dt_adj2, misfit_dt = cc_adj_DD(
                s1, s2, cc_shift_s, dd_shift, deltat)

        # taper adjoint source again
        window_taper(dt_adj1, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        window_taper(dt_adj2, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)

        misfit_sum_p += misfit_dt
        fp1[left_sample_1:right_sample_1] = dt_adj1[0:nlen]
        fp2[left_sample_2:right_sample_2] = dt_adj2[0:nlen]

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
