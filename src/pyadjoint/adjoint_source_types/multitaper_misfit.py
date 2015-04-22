#!/usr/bin/env python
# -*- encoding: utf8 -*-
"""
Multitaper based phase and amplitude misfit and adjoint source.

:copyright:
    Yanhua O. Yuan (yanhuay@princeton.edu), 2015
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from scipy.integrate import simps

from ..utils import generic_adjoint_source_plot, taper_window
from ..dpss import dpss_windows


# This is the verbose and pretty name of the adjoint source defined in this
# function.
VERBOSE_NAME = "Multitaper Misfit"

# Long and detailed description of the adjoint source defined in this file.
# Don't spare any details. This will be rendered as restructured text in the
# documentation. Be careful to escape the string with an ``r`` prefix.
# Otherwise most backslashes will have a special meaning which messes with the
# TeX like formulas.
DESCRIPTION = r"""
The misfit :math:`\chi_P(\mathbf{m})` measures
frequency-dependent phase differences
estimated with multitaper approach.
The misfit :math:`\chi_P(\mathbf{m})`
given Earth model :math:`\mathbf{m}`
and a single receiver and component is
given by

.. math::

    \chi_P (\mathbf{m}) = \frac{1}{2} \int_0^W  W_P(w) \left|
    \frac{ \tau^{\mathbf{d}}(w) - \tau^{\mathbf{s}}(w, \mathbf{m})}
    {\sigma_P(w)} \right|^ 2 dw

:math:`\tau^\mathbf{d}(w)` is the frequency-dependent
 phase measurement of the observed data and
 :math:`\tau^\mathbf{s}(w, \mathbf{m})` the frequency-dependent
phase measurement of the synthetic data.
 The function :math:`W_P(w)` denotes frequency-domain
taper corresponding to the frequency range over which
the measurements are assumed reliable.
:math:`\sigma_P(w)` is associated with the
traveltime uncertainty introduced in making measurements,
which can be estimated with cross-correlation method,
or Jackknife multitaper approach.
The adjoint source for the same receiver and component is given by

.. math::

    f_P^{\dagger}(t) = \sum_k h_k(t)P_j(t)

in which :math:`h_k(t)` is one of multitapers.
.. math::

    P_j(t) = 2\pi W_p(t) * \Delta \tau(t) * p_j(t) \\
    P_j(w) = 2\pi W_p(w) \Delta \tau(w) * p_j(w)   \\
    p_j(w) = \frac{iw s_j}{\sum_k(iw s_k)(iw s_k)^*} \\
    \Delta \tau(w) = \tau^{\mathbf{d}}(w) - \tau^{\mathbf{s}}(w, \mathbf{m})

"""
# Optional: document any additional parameters this particular adjoint sources
# receives in addition to the ones passed to the central adjoint source
# calculation function. Make sure to indicate the default values. This is a
# bit redundant but the only way I could figure out to make it work with the
# rest of the architecture of pyadjoint.
ADDITIONAL_PARAMETERS = r"""
**taper_percentage** (:class:`float`)
    Decimal percentage of taper at one end (ranging from ``0.0`` (0%) to
    ``0.5`` (50%)). Defauls to ``0.15``.

**taper_type** (:class:`str`)
    The taper type, supports anything :meth:`obspy.core.trace.Trace.taper`
    can use. Defaults to ``"hann"``.
"""
# global parameters used (save in parameter file in the future)
LNPT = 15
nlen_F = 2**LNPT
# half of frequencies
fnum = int(nlen_F/2 + 1)
# water level for tf
wtr_mtm = 1e-10
WTR = 0.02
PHASE_STEP = 1.5
# for frequency taper
ipwr_w = 10
# error estimation method
USE_CC_ERROR = False
USE_MT_ERROR = False


def _xcorr_shift(d, s):
    cc = np.correlate(d, s, mode="full")
    time_shift = cc.argmax() - len(d) + 1
    return time_shift


def cc_error(d1, d2, deltat, cc_shift, cc_dlnA):
    nlen_T = len(d1)

    # make cc-based corrections to d2
    d2_cc = np.zeros(nlen_T)
    for index in range(0, nlen_T):
        index_shift = index - cc_shift
        if index_shift >= 0 and index_shift < nlen_T:
            d2_cc[index] = np.exp(cc_dlnA) * d2[index_shift]

    # velocity of d2_cc
    # d2_cc_vel = np.zeros(nlen_T)
    d2_cc_vel = np.gradient(d2_cc) / deltat

    # the estimated error for dt and dlnA with uncorrelation assumption
    sigma_dt_top = np.sum(
        (d1[1:nlen_T] - d2_cc[1:nlen_T]) * (d1[1:nlen_T] - d2_cc[1:nlen_T]))
    sigma_dt_bot = np.sum(d2_cc_vel[1:nlen_T] * d2_cc_vel[1:nlen_T])
    sigma_dlnA_top = sigma_dt_top
    sigma_dlnA_bot = np.sum(d2_cc[1:nlen_T] * d2_cc[1:nlen_T]) / (
        cc_dlnA * cc_dlnA)
    sigma_dt = np.sqrt(sigma_dt_top / sigma_dt_bot)
    sigma_dlnA = np.sqrt(sigma_dlnA_top / sigma_dlnA_bot)

    return sigma_dt, sigma_dlnA


def frequency_limit(s, nlen, deltat, df):
        # find the maximum frequency point for measurement
        # using the spectrum of untapered synthetics
        # s_w = np.zeros(nlen_F,dtype=complex)
        s_w = np.fft.fft(s, nlen_F)
        ampmax = max(abs(s_w[0: fnum]))
        i_ampmax = np.argmax(abs(s_w[0: fnum]))
        wtr_thrd = ampmax * WTR

        # initialization
        nfreq_max = fnum - 1
        is_search = 1
        for iw in range(0, fnum):
            if iw > i_ampmax and abs(s_w[iw]) < wtr_thrd and is_search == 1:
                is_search = 0
                nfreq_max = iw
        if iw > i_ampmax and abs(s_w[iw]) > 10 * wtr_thrd and is_search == 0:
            # is_search = 1
            nfreq_max = iw
        nfreq_max = min(nfreq_max, int(1.0 / (2 * deltat) / df) - 1)

        nfreq_min = 0
        is_search = 1
        for iw in range(fnum - 1, 0, -1):
            if iw < i_ampmax and abs(s_w[iw]) < wtr_thrd and is_search == 1:
                is_search = 0
                nfreq_min = iw
        if iw < i_ampmax and abs(s_w[iw]) > 10 * wtr_thrd and is_search == 0:
            is_search = 1
            nfreq_min = iw
        # assume there are at least three cycles within the window
        nfreq_min = max(nfreq_min, int(3.0 / (nlen * deltat) / df) - 1)

        return nfreq_min, nfreq_max


def mt_measure(d1, d2, tapers, wvec, df, nfreq_min, nfreq_max, cc_tshift,
               cc_dlnA):
    nlen_T = len(d1)
    ntaper = len(tapers[0])

    # initialization
    top_tf = np.zeros(nlen_F, dtype=complex)
    bottom_tf = np.zeros(nlen_F, dtype=complex)

    # Multitaper measurements
    for itaper in range(0, ntaper):
        taper = np.zeros(nlen_T)
        taper[0:nlen_T] = tapers[0:nlen_T, itaper]
        # apply time-domain multi-tapered measurements
        d1_t = np.zeros(nlen_T)
        d2_t = np.zeros(nlen_T)
        d1_t[0:nlen_T] = d1[0:nlen_T] * taper[0:nlen_T]
        d2_t[0:nlen_T] = d2[0:nlen_T] * taper[0:nlen_T]

        # apply FFT to tapered measurements
        # d1_tw = np.zeros(nlen_F, dtype=complex)
        # d2_tw = np.zeros(nlen_F, dtype=complex)
        d1_tw = np.fft.fft(d1_t, nlen_F)
        d2_tw = np.fft.fft(d2_t, nlen_F)

        # calculate top and bottom of MT transfer function
        top_tf = top_tf + d1_tw * d2_tw.conjugate()
        bottom_tf = bottom_tf + d2_tw * d2_tw.conjugate()

    # Calculate transfer function
    # using top and bottom part of transfer function
    # water level
    wtr_use = max(abs(bottom_tf[0:fnum])) * wtr_mtm ** 2
    # transfrer function
    trans_func = np.zeros(nlen_F, dtype=complex)
    trans_func[nfreq_min:nfreq_max] = top_tf[nfreq_min:nfreq_max] / (
        bottom_tf[nfreq_min:nfreq_max] + wtr_use * (
            abs(bottom_tf[nfreq_min:nfreq_max]) < wtr_use))

    # Estimate phase and amplitude anomaly from transfer function
    phi_w = np.zeros(nlen_F)
    abs_w = np.zeros(nlen_F)
    dtau_w = np.zeros(nlen_F)
    dlnA_w = np.zeros(nlen_F)
    phi_w[nfreq_min:nfreq_max] = np.arctan2(
        trans_func[nfreq_min:nfreq_max].imag,
        trans_func[nfreq_min:nfreq_max].real)
    abs_w[nfreq_min:nfreq_max] = np.abs(trans_func[nfreq_min:nfreq_max])
    # cycle-skipping (check smoothness of phi, add cc measure, future
    # implementation)
    for iw in range(nfreq_min + 1, nfreq_max - 1):
        smth = abs(phi_w[iw + 1] + phi_w[iw - 1] - 2.0 * phi_w[iw])
        smth1 = abs(
            (phi_w[iw + 1] + 2 * np.pi) + phi_w[iw - 1] - 2.0 * phi_w[iw])
        smth2 = abs(
            (phi_w[iw + 1] - 2 * np.pi) + phi_w[iw - 1] - 2.0 * phi_w[iw])
        if smth1 < smth and smth1 < smth2 and abs(
                phi_w[iw] - phi_w[iw + 1]) > PHASE_STEP:
            print('2pi phase shift at {0} w={1} diff={2}'.format(
                iw, wvec[iw], phi_w[iw] - phi_w[iw + 1]))
            phi_w[iw + 1:nfreq_max] = phi_w[iw + 1:nfreq_max] + 2 * np.pi
        if smth2 < smth and smth2 < smth1 and abs(
                phi_w[iw] - phi_w[iw + 1]) > PHASE_STEP:
            print('-2pi phase shift at {0} w={1} diff={2}'.format(
                iw, wvec[iw], phi_w[iw] - phi_w[iw + 1]))
            phi_w[iw + 1:nfreq_max] = phi_w[iw + 1:nfreq_max] - 2 * np.pi
    # add the CC measurements to the transfer function
    dtau_w[0] = cc_tshift
    dtau_w[max(nfreq_min, 1): nfreq_max] = - 1.0 / \
        wvec[max(nfreq_min, 1): nfreq_max] * \
        phi_w[max(nfreq_min, 1): nfreq_max] + cc_tshift
    dlnA_w[nfreq_min:nfreq_max] = np.log(abs_w[nfreq_min:nfreq_max]) + cc_dlnA

    return phi_w, abs_w, dtau_w, dlnA_w


def mt_error(d1, d2, tapers, wvec, df, nfreq_min, nfreq_max, cc_tshift,
             cc_dlnA, phi_mtm, abs_mtm, dtau_mtm, dlnA_mtm):
    nlen_T = len(d1)
    ntaper = len(tapers[0])

    # Jacknife MT estimates
    # initialization
    phi_mul = np.zeros((nlen_F, ntaper))
    abs_mul = np.zeros((nlen_F, ntaper))
    dtau_mul = np.zeros((nlen_F, ntaper))
    dlnA_mul = np.zeros((nlen_F, ntaper))
    ephi_ave = np.zeros(nlen_F)
    eabs_ave = np.zeros(nlen_F)
    edtau_ave = np.zeros(nlen_F)
    edlnA_ave = np.zeros(nlen_F)
    err_phi = np.zeros(nlen_F)
    err_abs = np.zeros(nlen_F)
    err_dtau = np.zeros(nlen_F)
    err_dlnA = np.zeros(nlen_F)

    for itaper in range(0, ntaper):
        # delete one taper
        tapers_om = np.zeros((nlen_T, ntaper - 1))
        tapers_om[0:nlen_F, 0:ntaper - 1] = np.delete(tapers, itaper, 1)
        # multitaper measurements with ntaper-1 tapers
        # phi_om = np.zeros(nlen_F)
        # abs_om = np.zeros(nlen_F)
        # dtau_om = np.zeros(nlen_F)
        # dlnA_om = np.zeros(nlen_F)
        phi_om, abs_om, dtau_om, dlnA_om = \
            mt_measure(d1, d2, tapers_om, wvec, df, nfreq_min, nfreq_max,
                       cc_tshift, cc_dlnA)
        phi_mul[0:nlen_F, itaper] = phi_om[0:nlen_F]
        abs_mul[0:nlen_F, itaper] = abs_om[0:nlen_F]
        dtau_mul[0:nlen_F, itaper] = dtau_om[0:nlen_F]
        dlnA_mul[0:nlen_F, itaper] = dlnA_om[0:nlen_F]
        # error estimation
        ephi_ave[nfreq_min: nfreq_max] = ephi_ave[nfreq_min: nfreq_max] + \
            ntaper * phi_mtm[nfreq_min: nfreq_max] - (ntaper - 1) * \
            phi_mul[nfreq_min: nfreq_max, itaper]
        eabs_ave[nfreq_min:nfreq_max] = eabs_ave[nfreq_min: nfreq_max] + \
            ntaper * abs_mtm[nfreq_min: nfreq_max] - (ntaper - 1) * \
            abs_mul[nfreq_min: nfreq_max, itaper]
        edtau_ave[nfreq_min: nfreq_max] = edtau_ave[nfreq_min: nfreq_max] + \
            ntaper * dtau_mtm[nfreq_min: nfreq_max] - (ntaper - 1) * \
            dtau_mul[nfreq_min: nfreq_max, itaper]
        edlnA_ave[nfreq_min: nfreq_max] = edlnA_ave[nfreq_min: nfreq_max] + \
            ntaper * dlnA_mtm[nfreq_min: nfreq_max] - (ntaper - 1) * \
            dlnA_mul[nfreq_min: nfreq_max, itaper]
    # take average
    ephi_ave = ephi_ave / ntaper
    eabs_ave = eabs_ave / ntaper
    edtau_ave = edtau_ave / ntaper
    edlnA_ave = edlnA_ave / ntaper

    # deviation
    for itaper in range(0, ntaper):
        err_phi[nfreq_min: nfreq_max] = err_phi[nfreq_min: nfreq_max] + \
            (phi_mul[nfreq_min: nfreq_max, itaper] -
             ephi_ave[nfreq_min: nfreq_max]) ** 2
        err_abs[nfreq_min: nfreq_max] = err_abs[nfreq_min: nfreq_max] + \
            (abs_mul[nfreq_min: nfreq_max, itaper] -
             eabs_ave[nfreq_min: nfreq_max]) ** 2
        err_dtau[nfreq_min: nfreq_max] = err_dtau[nfreq_min: nfreq_max] + \
            (dtau_mul[nfreq_min: nfreq_max, itaper] -
             edtau_ave[nfreq_min: nfreq_max]) ** 2
        err_dlnA[nfreq_min: nfreq_max] = err_dlnA[nfreq_min: nfreq_max] + \
            (dlnA_mul[nfreq_min: nfreq_max, itaper] -
             edlnA_ave[nfreq_min: nfreq_max]) ** 2

    # standard deviation (msre)
    err_phi[nfreq_min: nfreq_max] = np.sqrt(
        err_phi[nfreq_min:nfreq_max] / (ntaper * (ntaper - 1)))
    err_abs[nfreq_min: nfreq_max] = np.sqrt(
        err_abs[nfreq_min: nfreq_max] / (ntaper * (ntaper - 1)))
    err_dtau[nfreq_min: nfreq_max] = np.sqrt(
        err_dtau[nfreq_min: nfreq_max] / (ntaper * (ntaper - 1)))
    # err_dtau[0] = LARGE_VAL
    err_dlnA[nfreq_min: nfreq_max] = np.sqrt(
        err_dlnA[nfreq_min: nfreq_max] / (ntaper * (ntaper - 1)))

    return err_phi, err_abs, err_dtau, err_dlnA


def mt_adj(d1, d2, deltat, tapers, dtau_mtm, dlnA_mtm, df, nfreq_min,
           nfreq_max, err_dt_cc, err_dlnA_cc, err_dtau_mt, err_dlnA_mt):
    nlen_T = len(d1)
    ntaper = len(tapers[0])

    # prepare frequency-domain taper based on reliable frequency band and
    # error estimation (future development)
    W_taper = np.zeros(nlen_F)
    # Wp_w = np.zeros(nlen_F)
    # Wq_w = np.zeros(nlen_F)
    iw = np.arange(nfreq_min, nfreq_max, 1)
    W_taper[nfreq_min: nfreq_max] = 1.0 - np.cos(
        np.pi * (iw[0:len(iw)] - nfreq_min) / (
            nfreq_max - nfreq_min)) ** ipwr_w
    # normalized factor
    ffac = 2.0 * df * np.sum(W_taper[nfreq_min: nfreq_max])
    Wp_w = W_taper / ffac
    Wq_w = W_taper / ffac

    # add error estimate
    # cc error
    if USE_CC_ERROR:
        Wp_w = Wp_w / (err_dt_cc * err_dt_cc)
        Wq_w = Wq_w / (err_dlnA_cc * err_dlnA_cc)
    # mt error
    if USE_MT_ERROR:
        dtau_wtr = WTR * \
            np.sum(np.abs(dtau_mtm[nfreq_min: nfreq_max])) / \
            (nfreq_max - nfreq_min)
        dlnA_wtr = WTR * \
            np.sum(np.abs(dlnA_mtm[nfreq_min: nfreq_max])) / \
            (nfreq_max - nfreq_min)
        err_dtau_mt[nfreq_min: nfreq_max] = \
            err_dtau_mt[nfreq_min: nfreq_max] + dtau_wtr * \
            (err_dtau_mt[nfreq_min: nfreq_max] < dtau_wtr)
        err_dlnA_mt[nfreq_min: nfreq_max] = \
            err_dlnA_mt[nfreq_min: nfreq_max] + dlnA_wtr * \
            (err_dlnA_mt[nfreq_min: nfreq_max] < dlnA_wtr)
        Wp_w[nfreq_min: nfreq_max] = Wp_w[nfreq_min: nfreq_max] / \
            ((err_dtau_mt[nfreq_min: nfreq_max]) ** 2)
        Wq_w[nfreq_min: nfreq_max] = Wq_w[nfreq_min: nfreq_max] / \
            ((err_dlnA_mt[nfreq_min: nfreq_max]) ** 2)

    # initialization
    bottom_p = np.zeros(nlen_F, dtype=complex)
    bottom_q = np.zeros(nlen_F, dtype=complex)
    d2_tw = np.zeros((nlen_F, ntaper), dtype=complex)
    d2_tvw = np.zeros((nlen_F, ntaper), dtype=complex)
    # Multitaper measurements
    for itaper in range(0, ntaper):
        taper = np.zeros(nlen_F)
        taper[0:nlen_T] = tapers[0:nlen_T, itaper]
        # multi-tapered measurements
        d2_t = np.zeros(nlen_T)
        # d2_tv = np.zeros(nlen_T)
        d2_t[0:nlen_T] = d2[0:nlen_T] * taper[0:nlen_T]
        d2_tv = np.gradient(d2_t) / deltat

        # apply FFT to tapered measurements
        d2_tw[:, itaper] = np.fft.fft(d2_t, nlen_F)[:]
        d2_tvw[:, itaper] = np.fft.fft(d2_tv, nlen_F)[:]
        # calculate bottom of adjoint term pj(w) qj(w)
        bottom_p[:] = bottom_p[:] + d2_tvw[:, itaper] * \
            d2_tvw[:, itaper].conjugate()
        bottom_q[:] = bottom_q[:] + d2_tw[:, itaper] * \
            d2_tw[:, itaper].conjugate()

    # Calculate adjoint source
    # initialization
    fp_t = np.zeros(nlen_F)
    fq_t = np.zeros(nlen_F)

    for itaper in range(0, ntaper):
        taper = np.zeros(nlen_F)
        taper[0: nlen_T] = tapers[0:nlen_T, itaper]

        # calculate pj(w) qj(w)
        p_w = np.zeros(nlen_F, dtype=complex)
        q_w = np.zeros(nlen_F, dtype=complex)
        p_w[nfreq_min:nfreq_max] = d2_tvw[nfreq_min:nfreq_max, itaper] / \
            (bottom_p[nfreq_min:nfreq_max])
        q_w[nfreq_min:nfreq_max] = - d2_tw[nfreq_min:nfreq_max, itaper] / \
            (bottom_q[nfreq_min:nfreq_max])

        # calculate weighted adjoint Pj(w) Qj(w) adding measurement dtau dlnA
        # P_w = np.zeros(nlen_F, dtype=complex)
        # P_w = np.zeros(nlen_F, dtype=complex)
        P_w = p_w * dtau_mtm * Wp_w
        Q_w = q_w * dlnA_mtm * Wq_w

        # inverse FFT to weighted adjoint (take real part)
        # P_wt = np.zeros(nlen_F)
        # P_wt = np.zeros(nlen_F)
        P_wt = np.fft.ifft(P_w, nlen_F).real * 2
        Q_wt = np.fft.ifft(Q_w, nlen_F).real * 2

        # apply tapering to adjoint
        fp_t = fp_t + P_wt * taper
        fq_t = fq_t + Q_wt * taper

    return fp_t, fq_t


# Each adjoint source file must contain a calculate_adjoint_source()
# function. It must take observed, synthetic, min_period, max_period,
# left_window_border, right_window_border, adjoint_src, and figure as
# parameters. Other optional keywork arguments are possible.
def calculate_adjoint_source(observed, synthetic, min_period, max_period,
                             left_window_border, right_window_border,
                             adjoint_src, figure, taper_percentage=0.15,
                             taper_type="hann"):  # NOQA
    # There is no need to perform any sanity checks on the passed trace
    # object. At this point they will be guaranteed to have the same
    # sampling rate, be sampled at the same points in time and a couple
    # other things.

    # All adjoint sources will need some kind of taper. Thus pyadjoint has a
    # convenience function to assist with that. The next block tapers both
    # observed and synthetic data.
    taper_window(observed, left_window_border, right_window_border,
                 taper_percentage=taper_percentage, taper_type=taper_type)
    taper_window(synthetic, left_window_border, right_window_border,
                 taper_percentage=taper_percentage, taper_type=taper_type)

    ret_val_p = {}
    ret_val_q = {}

    # initialization
    nlen_data = len(observed.data)
    deltat = observed.stats.delta
    fp = np.zeros(nlen_data)
    fq = np.zeros(nlen_data)

    # pre-processing of the observed and sythetic to get windowed obs and syn
    left_sample = int(left_window_border / deltat)
    right_sample = int(right_window_border / deltat)
    nlen = right_sample - left_sample
    d = np.zeros(nlen)
    s = np.zeros(nlen)
    d[0: nlen] = observed.data[left_sample: right_sample]
    s[0: nlen] = synthetic.data[left_sample: right_sample]

    # cross-correlation
    cc_shift = _xcorr_shift(d, s)
    cc_tshift = cc_shift * deltat

    # uncertainty estimate based on cross-correlations
    sigma_dt_cc = 1
    sigma_dlnA_cc = 1
    if USE_CC_ERROR:
        cc_dlnA = 0.5 * np.log(sum(d[0:nlen] * d[0:nlen]) /
                               sum(s[0:nlen] * s[0:nlen]))
        sigma_dt_cc, sigma_dlnA_cc = cc_error(d, s, deltat, cc_shift, cc_dlnA)

    # window for obs
    left_sample_d = max(left_sample + cc_shift, 0)
    right_sample_d = min(right_sample + cc_shift, nlen_data)
    nlen_d = right_sample_d - left_sample_d
    if nlen_d == nlen:
        cc_dlnA = 0
        d[0:nlen] = np.exp(-cc_dlnA) * \
            observed.data[left_sample_d: right_sample_d]
    else:
        raise Exception

    # multi-taper
    is_mtm = True
    if is_mtm:
        # discrete prolate slepian sequences
        # The time half bandwidth parameter (typical values are 2.5,3,3.5,4).
        NW = 4
        # number of tapers
        # ntaper = int(2 * NW)
        ntaper = 5
        tapers = dpss_windows(nlen, NW, ntaper)[0].T
        # normalized
        tapers = tapers * np.sqrt(nlen)

        # frequencies for FFT
        freq = np.fft.fftfreq(n=nlen_F, d=observed.stats.delta)
        df = freq[1] - freq[0]
        wvec = freq * 2 * np.pi
        dw = wvec[1] - wvec[0]

    # find min/max frequency limit for calculations
    nfreq_min, nfreq_max = frequency_limit(s, nlen, deltat, df)

    # calculate frequency-dependent phase and amplitude anomaly using
    # multi-taper approach
    phi_mtm = np.zeros(nlen_F)
    abs_mtm = np.zeros(nlen_F)
    # dtau_mtm = np.zeros(nlen_F)
    # dlnA_mtm = np.zeros(nlen_F)
    phi_mt, abs_mtm, dtau_mtm, dlnA_mtm = mt_measure(
        d, s, tapers, wvec, df, nfreq_min, nfreq_max, cc_tshift, cc_dlnA)

    # multi-taper error estimation
    # sigma_phi_mt = np.zeros(nlen_F)
    # sigma_abs_mt = np.zeros(nlen_F)
    sigma_dtau_mt = np.zeros(nlen_F)
    sigma_dlnA_mt = np.zeros(nlen_F)
    if USE_MT_ERROR:
        sigma_phi_mt, sigma_abs_mt, sigma_dtau_mt, sigma_dlnA_mt = mt_error(
            d, s, tapers, wvec, df, nfreq_min, nfreq_max, cc_tshift, cc_dlnA,
            phi_mtm, abs_mtm, dtau_mtm, dlnA_mtm)

    # calculate multi-taper adjoint source
    # fp_t = np.zeros(nlen_F)
    # fq_t = np.zeros(nlen_F)
    fp_t, fq_t = mt_adj(d, s, deltat, tapers, dtau_mtm, dlnA_mtm, df,
                        nfreq_min, nfreq_max, sigma_dt_cc, sigma_dlnA_cc,
                        sigma_dtau_mt, sigma_dlnA_mt)

    # post-processing
    # and return to original location before windowing
    # initialization
    fp_wind = np.zeros(len(synthetic.data))
    fq_wind = np.zeros(len(synthetic.data))
    fp_wind[left_sample: right_sample] = fp_t[0:nlen]
    fq_wind[left_sample: right_sample] = fq_t[0:nlen]
    fp = fp + fp_wind
    fq = fq + fq_wind

    # return misfit and adjoint source
    # Integrate with the composite Simpson's rule.
    ret_val_p["misfit"] = 0.5 * simps(y=dtau_mtm ** 2, dx=dw)
    ret_val_q["misfit"] = 0.5 * simps(y=dlnA_mtm ** 2, dx=dw)

    if adjoint_src is True:
        # Reverse in time and reverse the actual values.
        ret_val_p["adjoint_source"] = fp[::-1]
        ret_val_q["adjoint_source"] = fq[::-1]

    # outputs (amplitude misfit and adjoint is optional)
    if figure:
        generic_adjoint_source_plot(
            observed, synthetic, ret_val_p["adjoint_source"],
            ret_val_p["misfit"], left_window_border, right_window_border,
            VERBOSE_NAME)

    return ret_val_p
