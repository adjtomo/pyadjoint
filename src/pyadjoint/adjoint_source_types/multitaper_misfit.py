#!/usr/bin/env python
# -*- encoding: utf8 -*-
"""
Multitaper based phase and amplitude misfit and adjoint source.

:copyright:
    Youyi Ruan (youyir@princeton.edu), 2016
    Matthieu Lefebvre (ml15@princeton.edu), 2016
    Yanhua O. Yuan (yanhuay@princeton.edu), 2015
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from scipy.integrate import simps

from ..utils import generic_adjoint_source_plot
from ..utils import window_taper
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
and a single receiver is
given by

.. math::

    \chi_P (\mathbf{m}) = \frac{1}{2} \int_0^W  W_P(w) \left|
    \frac{ \tau^{\mathbf{d}}(w) - \tau^{\mathbf{s}}(w, \mathbf{m})}
    {\sigma_P(w)} \right|^ 2 dw

:math:`\tau^\mathbf{d}(w)` is the frequency-dependent
phase measurement of the observed data;
:math:`\tau^\mathbf{s}(w, \mathbf{m})` the frequency-dependent
phase measurement of the synthetic data.
The function :math:`W_P(w)` denotes frequency-domain
taper corresponding to the frequency range over which
the measurements are assumed reliable.
:math:`\sigma_P(w)` is associated with the
traveltime uncertainty introduced in making measurements,
which can be estimated with cross-correlation method,
or Jackknife multitaper approach.

The adjoint source for the same receiver is given by

.. math::

    f_P^{\dagger}(t) = \sum_k h_k(t)P_j(t)

in which :math:`h_k(t)` is one (the :math:`k`th) of multi-tapers.

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
# ADDITIONAL_PARAMETERS = r"""
# **taper_percentage** (:class:`float`)
#    Decimal percentage of taper at one end (ranging from ``0.0`` (0%) to
#    ``0.5`` (50%)). Defauls to ``0.15``.
#
# **taper_type** (:class:`str`)
#    The taper type, supports anything :meth:`obspy.core.trace.Trace.taper`
#    can use. Defaults to ``"hann"``.
#
# """


def _xcorr_shift(d, s):
    """
    Calculate cross correlation shift in points
    """
    cc = np.correlate(d, s, mode="full")
    time_shift = cc.argmax() - len(d) + 1
    return time_shift


def cc_error(d1, d2, deltat, cc_shift, cc_dlna, sigma_dt_min, sigma_dlna_min):
    """
    Estimate error for dt and dlna with uncorrelation assumption
    copied from c.c. misfit. should not duplicate the code but keep it for
    now and may need to move to utils later
    """
    nlen_t = len(d1)

    # make cc-based corrections to d2
    d2_cc_dt = np.zeros(nlen_t)
    d2_cc_dtdlna = np.zeros(nlen_t)

    for index in range(0, nlen_t):
        index_shift = index - cc_shift

        if 0 <= index_shift < nlen_t:
            d2_cc_dt[index] = d2[index_shift]
            d2_cc_dtdlna[index] = np.exp(cc_dlna) * d2[index_shift]

    # time derivative of d2_cc (velocity)
    d2_cc_vel = np.gradient(d2_cc_dtdlna, deltat)

    # Note: Beware of the first and last value in gradient calculation,
    #       ignored for safety reason, will be added in future
    sigma_dt_top = np.sum((d1 - d2_cc_dtdlna) ** 2)
    sigma_dt_bot = np.sum(d2_cc_vel** 2)

    sigma_dlna_top = sigma_dt_top
    sigma_dlna_bot = np.sum(d2_cc_dt**2)

    sigma_dt = np.sqrt(sigma_dt_top / sigma_dt_bot)
    sigma_dlna = np.sqrt(sigma_dlna_top / sigma_dlna_bot)

    if sigma_dt < sigma_dt_min:
            sigma_dt = sigma_dt_min

    if sigma_dlna < sigma_dlna_min:
                sigma_dlna = sigma_dlna_min

    return sigma_dt, sigma_dlna


def frequency_limit(s, nlen, nlen_f, deltat, df, wtr, ncycle_in_window,
                    min_period, max_period, nw):
    """
    First check if the window is suitable for mtm measurements, then
    find the maximum frequency point for measurement using the spectrum of
    tapered synthetics.

    :param is_mtm:
    :param nw:
    :param max_period:
    :param min_period:
    :param ncycle_in_window:
    :param wtr:
    :param df:
    :param nlen_f:
    :param deltat:
    :param nlen:
    :param s: synthetics
    :type s: float ndarray
    """

    ifreq_min = int(1.0 / (max_period * df))
    ifreq_max = int(1.0 / (min_period * df))

    # reject mtm if wave of min_period experience cycles less than ncycle
    # _in_window in the selected window, and switch to c.c. method.
    # In this case frequency limits is not needed.
    if ncycle_in_window * min_period > nlen * deltat:
        print ("min_period: %6.0f  window length: %6.0f" %
               (min_period, nlen*deltat))
        print ("MTM: rejecting for too few cycles within time window:")
        return (ifreq_min, ifreq_max, False)

    fnum = int(nlen_f/2 + 1)
    s_spectra = np.fft.fft(s, nlen_f) * deltat

    ampmax = max(abs(s_spectra[0: fnum]))
    i_ampmax = np.argmax(abs(s_spectra[0: fnum]))

    water_threshold = ampmax * wtr

    nfreq_max = get_max_frequency_limit(deltat, df, fnum, i_ampmax, ifreq_max,
                        s_spectra, water_threshold)

    nfreq_min = get_min_frequency_limit(deltat, df, fnum, i_ampmax, ifreq_min,
                        ncycle_in_window, nlen, s_spectra, water_threshold)

    # Assume the frequency range is larger than the bandwidth of multi-tapers
    # Too strict to implement, more experiments are needed. 
    # if (nfreq_max - nfreq_min) * df < nw / (nlen * deltat):
    #    is_mtm = False
    #    print ("(nfreq_max - nfreq_min) * df: %f" %
    #           ((nfreq_max - nfreq_min) * df))
    #    print ("nw*2.0 / (nlen * deltat): %f" % (nw*2.0 / (nlen * deltat)))
    #    print ("MTM: rejecting for frequency "
    #           "range narrower than taper bandwith:")
    #    return int(1.0 / (max_period * df)), int(1.0 / (min_period * df))

    return nfreq_min, nfreq_max, True


def get_min_frequency_limit(deltat, df, fnum, i_ampmax, ifreq_min,
                            ncycle_in_window, nlen, s_spectra, water_threshold):
    nfreq_min = 0
    is_search = True

    for iw in range(fnum - 1, 0, -1):
        if iw < i_ampmax:
            nfreq_min = search_frequency_limit(is_search, iw, nfreq_min, s_spectra,
                                               water_threshold)

    # assume there are at least N cycles within the window
    nfreq_min = max(nfreq_min, int(ncycle_in_window/(nlen*deltat)/df) - 1)
    nfreq_min = max(nfreq_min, ifreq_min)

    return nfreq_min


def get_max_frequency_limit(deltat, df, fnum, i_ampmax, ifreq_max, s_spectra,
                            water_threshold):
    nfreq_max = fnum - 1
    is_search = True

    for iw in range(0, fnum):
        if iw > i_ampmax:
            nfreq_max = search_frequency_limit(is_search, iw, nfreq_max, s_spectra,
                                               water_threshold)
    # Don't go beyond the Nyquist frequency
    nfreq_max = min(nfreq_max, int(1.0/(2*deltat)/df) - 1)
    nfreq_max = min(nfreq_max, ifreq_max)

    return nfreq_max


def search_frequency_limit(is_search, index, nfreq_limit, spectra,
                           water_threshold):
    """
    Search valid frequency range of spectra

    :param is_search: Logic switch
    :param spectra: spectra of signal
    :param index: index of spectra
    :water_threshold: the triggering value to stop the search
                    
    If the spectra larger than 10*water_threshold will trigger the 
    search again, works like the heating thermostat.

    The constant 10 may need to move outside to allow user choose  
    different values.
    """

    if abs(spectra[index]) < water_threshold and is_search:
        is_search = False
        nfreq_limit = index

    if abs(spectra[index]) > 10 * water_threshold and not is_search:
        is_search = True
        nfreq_limit = index

    return nfreq_limit


def mt_measure_select(nfreq_min, nfreq_max, df, nlen, deltat, dtau_w, dt_fac,
                      err_dt, err_fac, cc_tshift, dt_max_scale):
    """
    check mtm measurement see if the measurements are good to keep,
    otherwise use c.c. measurement instead

    :param is_mtm: logic switch of c.c. or mtm 
    :param dt_max_scale: 
    :param cc_tshift:
    :param err_fac:
    :param err_dt:
    :param dt_fac:
    :param dtau_w:
    :param deltat:
    :param nlen:
    :param df:
    :param nfreq_max:
    :param nfreq_min:
    """

    # If any mtm measurements is out of the resonable range,
    # switch from mtm to c.c.
    for j in range(nfreq_min, nfreq_max):

        # dt larger than 1/dt_fac of the wave period
        if np.abs(dtau_w[j]) > 1./(dt_fac*j*df):
            return False

        # error larger than 1/err_fac of wave period
        if err_dt[j] > 1./(err_fac*j*df):
            return False

        # dt larger than the maximum time shift allowed
        if np.abs(dtau_w[j]) > dt_max_scale*abs(cc_tshift):
            return False

    return True


def mt_measure(d1, d2, dt, tapers, wvec, df, nlen_f, waterlevel_mtm, phase_step,
               nfreq_min, nfreq_max, cc_tshift, cc_dlna):

    nlen_t = len(d1)
    ntaper = len(tapers[0])

    fnum = int(nlen_f/2 + 1)

    # initialization
    top_tf = np.zeros(nlen_f, dtype=complex)
    bot_tf = np.zeros(nlen_f, dtype=complex)

    # Multitaper measurements
    for itaper in range(0, ntaper):

        taper = np.zeros(nlen_t)
        taper[0:nlen_t] = tapers[0:nlen_t, itaper]

        # apply time-domain multi-tapered measurements
        # Youyi Ruan 10/29/2015 (no cc shift) change to complex
        d1_t = np.zeros(nlen_t, dtype=complex)
        d2_t = np.zeros(nlen_t, dtype=complex)

        d1_t[0:nlen_t] = d1[0:nlen_t] * taper[0:nlen_t]
        d2_t[0:nlen_t] = d2[0:nlen_t] * taper[0:nlen_t]

        d1_tw = np.fft.fft(d1_t, nlen_f) * dt
        d2_tw = np.fft.fft(d2_t, nlen_f) * dt

        # calculate top and bottom of MT transfer function
        top_tf[:] = top_tf[:] + d1_tw[:] * d2_tw[:].conjugate()
        bot_tf[:] = bot_tf[:] + d2_tw[:] * d2_tw[:].conjugate()

    # ===
    # Calculate transfer function 
    # ===

    # water level
    wtr_use = max(abs(bot_tf[0:fnum])) * waterlevel_mtm ** 2

    # transfrer function
    trans_func = np.zeros(nlen_f, dtype=complex)
    for i in range(nfreq_min,nfreq_max):
        if abs(bot_tf[i]) < wtr_use:
            trans_func[i] = top_tf[i] / bot_tf[i]
        else:
            trans_func[i] = top_tf[i] / (bot_tf[i] + wtr_use)
    #trans_func[nfreq_min:nfreq_max] = \
    #    top_tf[nfreq_min:nfreq_max] / \
    #    (bot_tf[nfreq_min:nfreq_max] + wtr_use *
    #     (abs(bot_tf[nfreq_min:nfreq_max]) < wtr_use))

    # Estimate phase and amplitude anomaly from transfer function
    phi_w = np.zeros(nlen_f)
    abs_w = np.zeros(nlen_f)
    dtau_w = np.zeros(nlen_f)
    dlna_w = np.zeros(nlen_f)

    phi_w[nfreq_min:nfreq_max] = np.arctan2(
        trans_func[nfreq_min:nfreq_max].imag,
        trans_func[nfreq_min:nfreq_max].real)

    abs_w[nfreq_min:nfreq_max] = np.abs(trans_func[nfreq_min:nfreq_max])

    # cycle-skipping (check smoothness of phi, add cc measure, future
    # implementation)
    for iw in range(nfreq_min + 1, nfreq_max - 1):
        smth = abs(phi_w[iw + 1] + phi_w[iw - 1] - 2.0 * phi_w[iw])
        smth1 = abs((phi_w[iw + 1] + 2*np.pi) + phi_w[iw - 1] - 2.0*phi_w[iw])
        smth2 = abs((phi_w[iw + 1] - 2*np.pi) + phi_w[iw - 1] - 2.0*phi_w[iw])

        if smth1 < smth and smth1 < smth2 and \
                abs(phi_w[iw] - phi_w[iw + 1]) > phase_step:
            print('2pi phase shift at {0} w={1} diff={2}'.format(
                iw, wvec[iw], phi_w[iw] - phi_w[iw + 1]))
            phi_w[iw + 1:nfreq_max] = phi_w[iw + 1:nfreq_max] + 2 * np.pi

        if smth2 < smth and smth2 < smth1 and \
                abs(phi_w[iw] - phi_w[iw + 1]) > phase_step:
            print('-2pi phase shift at {0} w={1} diff={2}'.format(
                iw, wvec[iw], phi_w[iw] - phi_w[iw + 1]))
            phi_w[iw + 1:nfreq_max] = phi_w[iw + 1:nfreq_max] - 2 * np.pi

    # add the CC measurements to the transfer function
    dtau_w[0] = cc_tshift
    dtau_w[max(nfreq_min, 1): nfreq_max] =\
        - 1.0 / wvec[max(nfreq_min, 1): nfreq_max] * \
        phi_w[max(nfreq_min, 1): nfreq_max] + cc_tshift

    dlna_w[nfreq_min:nfreq_max] = np.log(abs_w[nfreq_min:nfreq_max]) + cc_dlna
    # Warning: needs to be fixed. check if cc_dlna is not applied before the transf cal
    #dlna_w[nfreq_min:nfreq_max] = np.log(abs_w[nfreq_min:nfreq_max])

    return phi_w, abs_w, dtau_w, dlna_w


def mt_error(d1, d2, deltat, tapers, wvec, df, nlen_f, waterlevel_mtm, phase_step,
             nfreq_min, nfreq_max, cc_tshift, cc_dlna, phi_mtm, abs_mtm,
             dtau_mtm, dlna_mtm):

    nlen_t = len(d1)
    ntaper = len(tapers[0])

    # Jacknife MT estimates
    # initialization
    phi_mul = np.zeros((nlen_f, ntaper))
    abs_mul = np.zeros((nlen_f, ntaper))
    dtau_mul = np.zeros((nlen_f, ntaper))
    dlna_mul = np.zeros((nlen_f, ntaper))
    ephi_ave = np.zeros(nlen_f)
    eabs_ave = np.zeros(nlen_f)
    edtau_ave = np.zeros(nlen_f)
    edlna_ave = np.zeros(nlen_f)
    err_phi = np.zeros(nlen_f)
    err_abs = np.zeros(nlen_f)
    err_dtau = np.zeros(nlen_f)
    err_dlna = np.zeros(nlen_f)

    for itaper in range(0, ntaper):
        # delete one taper
        tapers_om = np.zeros((nlen_t, ntaper - 1))
        tapers_om[0:nlen_f, 0:ntaper - 1] = np.delete(tapers, itaper, 1)

        phi_om, abs_om, dtau_om, dlna_om =\
            mt_measure(d1, d2, deltat, tapers_om,
                       wvec, df, nlen_f, waterlevel_mtm, phase_step,
                       nfreq_min, nfreq_max, cc_tshift, cc_dlna)

        phi_mul[0:nlen_f, itaper] = phi_om[0:nlen_f]
        abs_mul[0:nlen_f, itaper] = abs_om[0:nlen_f]
        dtau_mul[0:nlen_f, itaper] = dtau_om[0:nlen_f]
        dlna_mul[0:nlen_f, itaper] = dlna_om[0:nlen_f]

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
        edlna_ave[nfreq_min: nfreq_max] = edlna_ave[nfreq_min: nfreq_max] + \
            ntaper * dlna_mtm[nfreq_min: nfreq_max] - (ntaper - 1) * \
            dlna_mul[nfreq_min: nfreq_max, itaper]

    # take average
    ephi_ave /= ntaper
    eabs_ave /= ntaper
    edtau_ave /= ntaper
    edlna_ave /= ntaper

    # deviation
    for itaper in range(0, ntaper):
        err_phi[nfreq_min:nfreq_max] += \
            (phi_mul[nfreq_min: nfreq_max, itaper] -
             ephi_ave[nfreq_min: nfreq_max])**2
        err_abs[nfreq_min:nfreq_max] += \
            (abs_mul[nfreq_min: nfreq_max, itaper] -
             eabs_ave[nfreq_min: nfreq_max])**2
        err_dtau[nfreq_min:nfreq_max] += \
            (dtau_mul[nfreq_min: nfreq_max, itaper] -
             edtau_ave[nfreq_min: nfreq_max])**2
        err_dlna[nfreq_min:nfreq_max] += \
            (dlna_mul[nfreq_min: nfreq_max, itaper] -
             edlna_ave[nfreq_min: nfreq_max])**2

    # standard deviation
    err_phi[nfreq_min: nfreq_max] = np.sqrt(
        err_phi[nfreq_min:  nfreq_max] / (ntaper * (ntaper - 1)))
    err_abs[nfreq_min: nfreq_max] = np.sqrt(
        err_abs[nfreq_min:  nfreq_max] / (ntaper * (ntaper - 1)))
    err_dtau[nfreq_min: nfreq_max] = np.sqrt(
        err_dtau[nfreq_min: nfreq_max] / (ntaper * (ntaper - 1)))
    err_dlna[nfreq_min: nfreq_max] = np.sqrt(
        err_dlna[nfreq_min: nfreq_max] / (ntaper * (ntaper - 1)))

    return err_phi, err_abs, err_dtau, err_dlna


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


def mt_adj(d1, d2, deltat, tapers, dtau_mtm, dlna_mtm, df, nlen_f,
           use_cc_error, use_mt_error, nfreq_min, nfreq_max, err_dt_cc,
           err_dlna_cc, err_dtau_mt, err_dlna_mt, wtr):

    nlen_t = len(d1)
    ntaper = len(tapers[0])

    misfit_p = 0.0
    misfit_q = 0.0


    # Y. Ruan, 11/05/2015
    # frequency-domain taper based on adjusted frequency band and
    # error estimation. It's not one of the filtering processes that
    # needed to applied to adjoint source but an frequency domain
    # weighting function for adjoint source and misfit function.

    wp_w = np.zeros(nlen_f)
    wq_w = np.zeros(nlen_f)

    iw = np.arange(nfreq_min, nfreq_max, 1)

    w_taper = np.zeros(nlen_f)
    #w_taper[nfreq_min: nfreq_max] = 1.0

    # Y. Ruan, 11/09/2015
    # Original higher order cosine taper used in measure_adj
    # this cosine weighting function tapering too much information
    # will be replaced by a less aggressive taper
    ipwr_w = 10
    w_taper[nfreq_min: nfreq_max] = 1.0 - \
        np.cos(np.pi * (iw - nfreq_min) / (nfreq_max - nfreq_min)) ** ipwr_w

    # normalization factor, factor 2 is needed for the integration from 
    # -inf to inf
    ffac = 2.0 * df * np.sum(w_taper[nfreq_min: nfreq_max])

    wp_w = w_taper / ffac
    wq_w = w_taper / ffac

    # cc error
    if use_cc_error:
        wp_w /= err_dt_cc**2
        wq_w /= err_dlna_cc**2

    # mt error
    if use_mt_error:
        dtau_wtr = wtr * \
            np.sum(np.abs(dtau_mtm[nfreq_min: nfreq_max])) / \
            (nfreq_max - nfreq_min)
        dlna_wtr = wtr * \
            np.sum(np.abs(dlna_mtm[nfreq_min: nfreq_max])) / \
            (nfreq_max - nfreq_min)

        err_dtau_mt[nfreq_min: nfreq_max] = \
            err_dtau_mt[nfreq_min: nfreq_max] + dtau_wtr * \
            (err_dtau_mt[nfreq_min: nfreq_max] < dtau_wtr)
        err_dlna_mt[nfreq_min: nfreq_max] = \
            err_dlna_mt[nfreq_min: nfreq_max] + dlna_wtr * \
            (err_dlna_mt[nfreq_min: nfreq_max] < dlna_wtr)

        wp_w[nfreq_min: nfreq_max] = wp_w[nfreq_min: nfreq_max] / \
            ((err_dtau_mt[nfreq_min: nfreq_max]) ** 2)
        wq_w[nfreq_min: nfreq_max] = wq_w[nfreq_min: nfreq_max] / \
            ((err_dlna_mt[nfreq_min: nfreq_max]) ** 2)

    # initialization
    bottom_p = np.zeros(nlen_f, dtype=complex)
    bottom_q = np.zeros(nlen_f, dtype=complex)

    d2_tw = np.zeros((nlen_f, ntaper), dtype=complex)
    d2_tvw = np.zeros((nlen_f, ntaper), dtype=complex)

    # Multitaper measurements
    for itaper in range(0, ntaper):
        taper = np.zeros(nlen_f)
        taper[0:nlen_t] = tapers[0:nlen_t, itaper]

        # multi-tapered measurements
        d2_t = np.zeros(nlen_t)
        d2_tv = np.zeros(nlen_t)
        #d2_t[0:nlen_t] = d2[0:nlen_t] * taper[0:nlen_t]
        d2_t = d2 * taper[0:nlen_t]
        d2_tv = np.gradient(d2_t, deltat)

        # apply FFT to tapered measurements
        d2_tw[:, itaper] = np.fft.fft(d2_t, nlen_f)[:] * deltat
        d2_tvw[:, itaper] = np.fft.fft(d2_tv, nlen_f)[:] * deltat

        # calculate bottom of adjoint term pj(w) qj(w)
        bottom_p[:] = bottom_p[:] + \
            d2_tvw[:, itaper] * d2_tvw[:, itaper].conjugate()
        bottom_q[:] = bottom_q[:] + \
            d2_tw[:, itaper] * d2_tw[:, itaper].conjugate()

    fp_t = np.zeros(nlen_f)
    fq_t = np.zeros(nlen_f)

    for itaper in range(0, ntaper):
        taper = np.zeros(nlen_f)
        taper[0: nlen_t] = tapers[0:nlen_t, itaper]

        # calculate pj(w), qj(w)
        p_w = np.zeros(nlen_f, dtype=complex)
        q_w = np.zeros(nlen_f, dtype=complex)

        p_w[nfreq_min:nfreq_max] = d2_tvw[nfreq_min:nfreq_max, itaper] / \
            (bottom_p[nfreq_min:nfreq_max])
        q_w[nfreq_min:nfreq_max] = -d2_tw[nfreq_min:nfreq_max, itaper] / \
            (bottom_q[nfreq_min:nfreq_max])

        # calculate weighted adjoint Pj(w), Qj(w) adding measurement dtau dlna
        p_w *= dtau_mtm * wp_w
        q_w *= dlna_mtm * wq_w

        # inverse FFT to weighted adjoint (take real part)
        p_wt = np.fft.ifft(p_w, nlen_f).real * 2. / deltat
        q_wt = np.fft.ifft(q_w, nlen_f).real * 2. / deltat

        # apply tapering to adjoint source
        fp_t += p_wt * taper
        fq_t += q_wt * taper

    # calculate misfit
    dtau_mtm_weigh_sqr = dtau_mtm**2 * wp_w
    dlna_mtm_weigh_sqr = dlna_mtm**2 * wq_w

    # Integrate with the composite Simpson's rule.
    misfit_p = 0.5 * 2.0 * simps(y=dtau_mtm_weigh_sqr, dx=df)
    misfit_q = 0.5 * 2.0 * simps(y=dlna_mtm_weigh_sqr, dx=df)

    return fp_t, fq_t, misfit_p, misfit_q


def calculate_adjoint_source(observed, synthetic, config, window,
                             adjoint_src, figure):  # NOQA
    # There is no need to perform any sanity checks on the passed trace
    # object. At this point they will be guaranteed to have the same
    # sampling rate, be sampled at the same points in time and a couple
    # other things.

    # All adjoint sources will need some kind of windowing taper.
    # Thus pyadjoint has a convenience function to assist with that.
    # The next block tapers both observed and synthetic data.

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
    use_mt_error = config.use_mt_error

    # Frequency range for adjoint src
    min_period = config.min_period
    max_period = config.max_period

    # critiaria for rejecting mtm measurements
    dt_fac = config.dt_fac
    err_fac = config.err_fac
    dt_max_scale = config.dt_max_scale

    # initialized the adjoint source
    ret_val_p = {}
    ret_val_q = {}

    nlen_data = len(synthetic.data)
    deltat = synthetic.stats.delta

    fp = np.zeros(nlen_data)
    fq = np.zeros(nlen_data)

    misfit_sum_p = 0.0
    misfit_sum_q = 0.0

    # ===
    # Loop over time windows
    # ===
    for wins in window:

        left_window_border = wins[0]
        right_window_border = wins[1]

        # ===
        # pre-processing of the observed and sythetic
        # to get windowed obsd and synt
        # ===
        left_sample = int(np.floor(left_window_border / deltat))
        nlen = int(np.floor((right_window_border - left_window_border) /
                   deltat)) + 1
        right_sample = left_sample + nlen

        d = np.zeros(nlen)
        s = np.zeros(nlen)

        d[0: nlen] = observed.data[left_sample: right_sample]
        s[0: nlen] = synthetic.data[left_sample: right_sample]

        # Taper signals following the SAC taper command
        window_taper(d, taper_percentage=config.taper_percentage,
                        taper_type=config.taper_type)
        window_taper(s, taper_percentage=config.taper_percentage,
                        taper_type=config.taper_type)

        # cross-correlation
        cc_shift = _xcorr_shift(d, s)
        cc_tshift = cc_shift * deltat

        # uncertainty estimate based on cross-correlations
        sigma_dt_cc = 1.0
        sigma_dlna_cc = 1.0

        if use_cc_error:
            cc_dlna = 0.5 * np.log(sum(d**2) / sum(s**2))
            sigma_dt_cc, sigma_dlna_cc = cc_error(d, s, deltat, cc_shift, cc_dlna,
                                        config.dt_sigma_min, config.dlna_sigma_min)

            # print("cc_dt  : %f +/- %f" % (cc_tshift, sigma_dt_cc))
            # print("cc_dlna: %f +/- %f" % (cc_dlna, sigma_dlna_cc))

        # re-window observed to align observed with synthetic for multitaper
        # measurement:
        left_sample_d = max(left_sample + cc_shift, 0)
        right_sample_d = min(right_sample + cc_shift, nlen_data)

        nlen_d = right_sample_d - left_sample_d

        if nlen_d == nlen:
            # Y. Ruan: No need to correct cc_dlna in multitaper measurements
            d[0:nlen] = observed.data[left_sample_d:right_sample_d]
            window_taper(d, taper_percentage=config.taper_percentage,
                            taper_type=config.taper_type)
        else:
            raise Exception

        # ===
        # Make decision wihich method to use: c.c. or multi-taper
        # always starts from multi-taper, if it doesn't work then
        # switch to cross correlation misfit
        # ===
        is_mtm = True

        # frequencies for FFT
        freq = np.fft.fftfreq(n=nlen_f, d=observed.stats.delta)
        df = freq[1] - freq[0]
        wvec = freq * 2 * np.pi
        dw = wvec[1] - wvec[0]

        # check window if okay for mtm measurements, and then find min/max
        # frequency limit for calculations.
        nfreq_min, nfreq_max, is_mtm = \
            frequency_limit(s, nlen, nlen_f, deltat, df, wtr, ncycle_in_window,
                            min_period, max_period, config.mt_nw)

        if is_mtm:
            # Set the Rayleigh bin parameter (determin taper bandwithin
            # frequency domain): nw (typical values are 2.5,3,3.5,4).
            nw = config.mt_nw
            ntaper = config.num_taper

            # generate discrete prolate slepian sequences
            tapers = dpss_windows(nlen, nw, ntaper)[0].T

            # normalization
            tapers = tapers * np.sqrt(nlen)

            # measure frequency-dependent phase and amplitude difference
            phi_mtm = np.zeros(nlen_f)
            abs_mtm = np.zeros(nlen_f)

            phi_mtm, abs_mtm, dtau_mtm, dlna_mtm =\
                mt_measure(d, s, deltat, tapers, wvec, df, nlen_f,
                           waterlevel_mtm, phase_step, nfreq_min, nfreq_max,
                           cc_tshift, cc_dlna)

            # multi-taper error estimation
            sigma_phi_mt = np.zeros(nlen_f)
            sigma_abs_mt = np.zeros(nlen_f)
            sigma_dtau_mt = np.zeros(nlen_f)
            sigma_dlna_mt = np.zeros(nlen_f)

            sigma_phi_mt, sigma_abs_mt, sigma_dtau_mt, sigma_dlna_mt =\
                mt_error(d, s, deltat, tapers, wvec, df, nlen_f, waterlevel_mtm,
                         phase_step, nfreq_min, nfreq_max, cc_tshift, cc_dlna,
                         phi_mtm, abs_mtm, dtau_mtm, dlna_mtm)

            # check is_mtm again if the multitaper measurement results failed
            # the selctuing criteria.  change is_mtm if it's not okay
            is_mtm = mt_measure_select(nfreq_min, nfreq_max, df, nlen, deltat,
                                       dtau_mtm, dt_fac, sigma_dtau_mt,
                                       err_fac, cc_tshift, dt_max_scale)

        # final decision which misfit will be used for adjoint source.
        if is_mtm:
            # calculate multi-taper adjoint source
            fp_t, fq_t, misfit_p, misfit_q = mt_adj(d, s, deltat, tapers, 
                                dtau_mtm, dlna_mtm, df, nlen_f, use_cc_error,
                                use_mt_error, nfreq_min, nfreq_max,
                                sigma_dt_cc, sigma_dlna_cc, sigma_dtau_mt,
                                sigma_dlna_mt, wtr)


        else:
            # calculate c.c. adjoint source
            fp_t, fq_t, misfit_p, misfit_q = cc_adj(s, cc_shift, cc_dlna, 
                                deltat, sigma_dt_cc, sigma_dlna_cc)


        # return to original location before windowing
        # initialization
        fp_wind = np.zeros(len(synthetic.data))
        fq_wind = np.zeros(len(synthetic.data))

        fp_wind[left_sample: right_sample] = fp_t[0:nlen]
        fq_wind[left_sample: right_sample] = fq_t[0:nlen]

        fp += fp_wind
        fq += fq_wind

        misfit_sum_p += misfit_p
        misfit_sum_q += misfit_q

    ret_val_p["misfit"] = misfit_sum_p
    ret_val_q["misfit"] = misfit_sum_q

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

    if config.measure_type == "dt":
        return ret_val_p

    if config.measure_type == "am":
        return ret_val_q
