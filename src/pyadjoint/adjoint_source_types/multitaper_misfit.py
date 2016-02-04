#!/usr/bin/env python
# -*- encoding: utf8 -*-
"""
Multitaper based phase and amplitude misfit and adjoint source.

:copyright:
    Yanhua O. Yuan (yanhuay@princeton.edu), 2015
    Youyi Ruan (youyir@princeton.edu), 2016
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from scipy.integrate import simps

from ..utils import generic_adjoint_source_plot
from ..utils import taper_window, sac_hann_taper
from ..dpss  import dpss_windows


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
#ADDITIONAL_PARAMETERS = r"""
#**taper_percentage** (:class:`float`)
#    Decimal percentage of taper at one end (ranging from ``0.0`` (0%) to
#    ``0.5`` (50%)). Defauls to ``0.15``.
#
#**taper_type** (:class:`str`)
#    The taper type, supports anything :meth:`obspy.core.trace.Trace.taper`
#    can use. Defaults to ``"hann"``.
#"""


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



def frequency_limit(s, nlen, nlen_F, deltat, df, wtr, ncycle_in_window, 
                    min_period, max_period, nw, is_mtm):
    """
    First check if the window is suitable for mtm measurements, then
    find the maximum frequency point for measurement using the spectrum of 
    tapered synthetics.
    """

    # reject mtm if wave of min_period experience cycles less than ncycle_in_window
    # in the selected window, and switch to c.c. method. In this case frequency limits
    # is not needed. 
    if ncycle_in_window * min_period > nlen * deltat:
        is_mtm = False 
        print ("ncycle_in_window * min_period: %f" % (ncycle_in_window*min_period))
        print ("nlen * deltat: %f" % (nlen * deltat))
        print ("%s" % "MTM: rejecting for too few cycles within time window:")        
        return int(1.0 / (max_period * df)), int(1.0 / (min_period * df))

    # s_w = np.zeros(nlen_F,dtype=complex)
    fnum = int(nlen_F/2 + 1)
    s_w  = np.fft.fft(s, nlen_F) * deltat 
    ampmax = max(abs(s_w[0: fnum]))
    i_ampmax = np.argmax(abs(s_w[0: fnum]))
    wtr_thrd = ampmax * wtr

    #=== Youyi Ruan, 11/05/2015. Add for the lower limit of spectra
    #winlen = delta * len(s)
    ifreq_min = int(1.0 / (max_period * df)) 
    ifreq_max = int(1.0 / (min_period * df))   
 
    #=== Youyi Ruan test 11/05/2015
    #print('i_ampmax, freq_ampmax, T_ampmax:',i_ampmax,i_ampmax*df,1.0/(i_ampmax*df))
    #=== 

    # initialization
    nfreq_max = fnum - 1
    is_search = 1
    for iw in range(0, fnum):
        if iw > i_ampmax and abs(s_w[iw]) < wtr_thrd and is_search == 1:
            is_search = 0
            nfreq_max = iw
        if iw > i_ampmax and abs(s_w[iw]) > 10 * wtr_thrd and is_search == 0:
            is_search = 1
            nfreq_max = iw

    nfreq_max = min(nfreq_max, int(1.0 / (2 * deltat) / df) - 1)
    nfreq_max = min(nfreq_max, ifreq_max)

    nfreq_min = 0
    is_search = 1
    for iw in range(fnum - 1, 0, -1):
        if iw < i_ampmax and abs(s_w[iw]) < wtr_thrd and is_search == 1:
            is_search = 0
            nfreq_min = iw
        if iw < i_ampmax and abs(s_w[iw]) > 10 * wtr_thrd and is_search == 0:
            is_search = 1
            nfreq_min = iw

    # assume there are at least N cycles within the window
    nfreq_min = max(nfreq_min, int(ncycle_in_window / (nlen * deltat) / df) - 1)
    nfreq_min = max(nfreq_min, ifreq_min)

    # assume the frequency range is larger than the bandwidth of multi-tapers
    if (nfreq_max - nfreq_min) * df < nw / (nlen * deltat):
        is_mtm = False 
        print ("(nfreq_max - nfreq_min) * df: %f" % ((nfreq_max - nfreq_min) * df))
        print ("nw*2.0 / (nlen * deltat): %f" % (nw*2.0 / (nlen * deltat)))
        print ("%s" % "MTM: rejecting for frequency range narrower than taper bandwith:")        
        return int(1.0 / (max_period * df)), int(1.0 / (min_period * df))


    #=== Youyi Ruan test 11/05/2015
    #print('nfreq_min nfreq_max: %9.4f %9.4f' % (1.0/(nfreq_min*df), 1.0/(nfreq_max*df)))

    return nfreq_min, nfreq_max



def mt_measure_select(nfreq_min, nfreq_max, df, nlen, deltat, dtau_w, dt_fac, err_dt, 
        err_fac, cc_tshift,  dt_max_scale, is_mtm):
    """
    check mtm measurement see if the measurements are good to keep
    otherwise use c.c. measurement instead 
    """

    # If any mtm measurements is out of the resonable range, switch from mtm to c.c.
    for j in range(nfreq_min, nfreq_max):

        # dt larger than 1/dt_fac of the wave period
        if np.abs(dtau_w[j]) > 1./(dt_fac*j*df):
            is_mtm = False
            return 

        # error larger than 1/err_fac of wave period
        if err_dt[j] > 1./(err_fac*j*df):
            is_mtm = False
            return

        # dt larger than the maximum time shift allowed
        if np.abs(dtau_w[j]) > dt_max_scale*abs(cc_tshift):
            is_mtm = False
            return
      


def mt_measure(d1, d2, dt, tapers, wvec, df, nlen_F, wtr_mtm, phase_step, 
               nfreq_min, nfreq_max, cc_tshift, cc_dlna):

    nlen_T = len(d1)
    ntaper = len(tapers[0])

    fnum = int(nlen_F/2 + 1)

    # initialization
    top_tf = np.zeros(nlen_F, dtype=complex)
    bot_tf = np.zeros(nlen_F, dtype=complex)

    # Multitaper measurements
    for itaper in range(0, ntaper):
        taper = np.zeros(nlen_T)
        taper[0:nlen_T] = tapers[0:nlen_T, itaper]
        # apply time-domain multi-tapered measurements 
        # Youyi Ruan 10/29/2015 (no cc shift) change to complex
        d1_t = np.zeros(nlen_T,dtype=complex)
        d2_t = np.zeros(nlen_T,dtype=complex)

        d1_t[0:nlen_T] = d1[0:nlen_T] * taper[0:nlen_T]
        d2_t[0:nlen_T] = d2[0:nlen_T] * taper[0:nlen_T]

        # apply FFT to tapered measurements
        # d1_tw = np.zeros(nlen_F, dtype=complex)
        # d2_tw = np.zeros(nlen_F, dtype=complex)
        # Youyi Ruan 11/04/2015 fft
        d1_tw = np.fft.fft(d1_t, nlen_F) * dt
        d2_tw = np.fft.fft(d2_t, nlen_F) * dt
        
        # calculate top and bottom of MT transfer function
        # YY RUAN 19/01/16 for test
        #top_tf = top_tf + d1_tw * d2_tw.conjugate()
        #bot_tf = bot_tf + d2_tw * d2_tw.conjugate()
        top_tf[:] = top_tf[:] + d1_tw[:] * d2_tw[:].conjugate()
        bot_tf[:] = bot_tf[:] + d2_tw[:] * d2_tw[:].conjugate()

        #==== YY Ruan for test 
        #print('ictaper:',itaper,'nlen_T:',nlen_T,'nlen_F:',nlen_F)
        #nlen_freq = len(top_tf)
        #print('transf nlen_freq:',nlen_freq)
        freq = np.fft.fftfreq(n=nlen_F, d=dt)

        #f8 = open('bottf.abs.py.transf','w')
        #nlen_half = int(len(bot_tf) / 2)
        #for idx, ele in enumerate(bot_tf[0:nlen_half]):
        #    f8.write("%f %e\n" %(freq[idx], np.abs(ele)))
        #f8.close()
   
        #f11 = open('toptf.abs.py.transf','w')
        #nlen_half = int(len(top_tf) / 2)
        #for idx, ele in enumerate(top_tf[0:nlen_half]):
        #    f11.write("%f %e\n" %(freq[idx], np.abs(ele)))
        #f11.close()

        #nlen_half = int((len(d2_tw) / 2)) 
        #f9 = open('syn.py.am','w')
        #for idx, ele in enumerate(d2_tw[0:nlen_half]):
        #    f9.write("%f %e\n" %(freq[idx], np.abs(ele)))
        #f9.close()
         
        #nlen_freq = len(d2)
        #f10 = open('py.syn','w')
        #for idx, ele in enumerate(d2_t[0:nlen_freq]):
        #    f10.write("%f %f\n" %(idx, ele.real))
        #f10.close()

        #nlen_half = int((len(d1_tw) / 2))
        #f12 = open('dat.py.am','w')
        #for idx, ele in enumerate(d1_tw[0:nlen_half]):
        #    f12.write("%f %e\n" %(freq[idx], np.abs(ele)))
        #f12.close()
         
        #nlen_freq = len(d1)
        #f13 = open('py.dat','w')
        #for idx, ele in enumerate(d1_t[0:nlen_freq]):
        #    f13.write("%f %f\n" %(idx, ele.real))
        #f13.close()

        
    #===
    # Calculate transfer function using top and bottom part of transfer function
    #===
    # water level 
    wtr_use = max(abs(bot_tf[0:fnum])) * wtr_mtm ** 2
    # transfrer function
    trans_func = np.zeros(nlen_F, dtype=complex)
    trans_func[nfreq_min:nfreq_max] = top_tf[nfreq_min:nfreq_max] / \
                                      ( bot_tf[nfreq_min:nfreq_max] + wtr_use * \
                                      (abs(bot_tf[nfreq_min:nfreq_max]) < wtr_use) )

    # Estimate phase and amplitude anomaly from transfer function
    phi_w  = np.zeros(nlen_F)
    abs_w  = np.zeros(nlen_F)
    dtau_w = np.zeros(nlen_F)
    dlna_w = np.zeros(nlen_F)

    phi_w[nfreq_min:nfreq_max] = np.arctan2(
        trans_func[nfreq_min:nfreq_max].imag,
        trans_func[nfreq_min:nfreq_max].real)

    abs_w[nfreq_min:nfreq_max] = np.abs(trans_func[nfreq_min:nfreq_max])

    # cycle-skipping (check smoothness of phi, add cc measure, future
    # implementation)
    for iw in range(nfreq_min + 1, nfreq_max - 1):
        smth  = abs(phi_w[iw + 1] + phi_w[iw - 1] - 2.0 * phi_w[iw])
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
    dtau_w[max(nfreq_min, 1): nfreq_max] = - 1.0 / \
                     wvec[max(nfreq_min, 1): nfreq_max] * \
                     phi_w[max(nfreq_min, 1): nfreq_max] + cc_tshift

    dlna_w[nfreq_min:nfreq_max] = np.log(abs_w[nfreq_min:nfreq_max]) + cc_dlna

    return phi_w, abs_w, dtau_w, dlna_w


def mt_error(d1, d2, deltat, tapers, wvec, df, nlen_F, wtr_mtm, phase_step, 
             nfreq_min, nfreq_max, cc_tshift, cc_dlna, phi_mtm, abs_mtm, 
             dtau_mtm, dlna_mtm):

    nlen_T = len(d1)
    ntaper = len(tapers[0])

    # Jacknife MT estimates
    # initialization
    phi_mul   = np.zeros((nlen_F, ntaper))
    abs_mul   = np.zeros((nlen_F, ntaper))
    dtau_mul  = np.zeros((nlen_F, ntaper))
    dlna_mul  = np.zeros((nlen_F, ntaper))
    ephi_ave  = np.zeros(nlen_F)
    eabs_ave  = np.zeros(nlen_F)
    edtau_ave = np.zeros(nlen_F)
    edlna_ave = np.zeros(nlen_F)
    err_phi   = np.zeros(nlen_F)
    err_abs   = np.zeros(nlen_F)
    err_dtau  = np.zeros(nlen_F)
    err_dlna  = np.zeros(nlen_F)

    for itaper in range(0, ntaper):
        # delete one taper
        tapers_om = np.zeros((nlen_T, ntaper - 1))
        tapers_om[0:nlen_F, 0:ntaper - 1] = np.delete(tapers, itaper, 1)
        # multitaper measurements with ntaper-1 tapers
        # phi_om  = np.zeros(nlen_F)
        # abs_om  = np.zeros(nlen_F)
        # dtau_om = np.zeros(nlen_F)
        # dlna_om = np.zeros(nlen_F)
        phi_om, abs_om, dtau_om, dlna_om = mt_measure(d1, d2, deltat, tapers_om, 
                                           wvec, df, nlen_F, wtr_mtm, phase_step,
                                           nfreq_min, nfreq_max,cc_tshift, cc_dlna)

        phi_mul[0:nlen_F, itaper]  =  phi_om[0:nlen_F]
        abs_mul[0:nlen_F, itaper]  =  abs_om[0:nlen_F]
        dtau_mul[0:nlen_F, itaper] = dtau_om[0:nlen_F]
        dlna_mul[0:nlen_F, itaper] = dlna_om[0:nlen_F]

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
    ephi_ave  =  ephi_ave / ntaper
    eabs_ave  =  eabs_ave / ntaper
    edtau_ave = edtau_ave / ntaper
    edlna_ave = edlna_ave / ntaper

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
        err_dlna[nfreq_min: nfreq_max] = err_dlna[nfreq_min: nfreq_max] + \
            (dlna_mul[nfreq_min: nfreq_max, itaper] -
            edlna_ave[nfreq_min: nfreq_max]) ** 2

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


def cc_adj(synt, cc_shift, deltat, err_dt_cc, err_dlna_cc):

    #ret_val_p["misfit"] = 0.5 * time_shift ** 2

    dsdt   = np.gradient(synt) / deltat

    nnorm = simps(y=dsdt*dsdt, dx=deltat)
    dt_adj = cc_shift / err_dt_cc**2 / nnorm * dsdt

    nnorm = simps(y=synt*synt, dx=deltat)
    am_adj = -1.0 * cc_dlna / err_dlna_cc**2 / nnorm * synt

    return dt_adj, am_adj


def mt_adj(d1, d2, deltat, tapers, dtau_mtm, dlna_mtm, df, nlen_F, 
           use_cc_error, use_mt_error, nfreq_min, nfreq_max, err_dt_cc, 
           err_dlna_cc, err_dtau_mt, err_dlna_mt):

    nlen_T = len(d1)
    ntaper = len(tapers[0])

    # YY Ruan, 11/05/2015
    # frequency-domain taper based on adjust frequency band and
    # error estimation. It's not one of the tapering processes but
    # an frequency domain weighting function for adjoint source and 
    # misfit function.

    W_taper = np.zeros(nlen_F)
    Wp_w = np.zeros(nlen_F)
    Wq_w = np.zeros(nlen_F)

    iw = np.arange(nfreq_min, nfreq_max, 1)
    W_taper[nfreq_min: nfreq_max] = 1.0

    # Original higher order cosine taper 
    #ipwr_w = 10
    #W_taper[nfreq_min: nfreq_max] = 1.0 - np.cos(np.pi * (iw[0:len(iw)] 
    #                  - nfreq_min) / (nfreq_max - nfreq_min)) ** ipwr_w

    # normalized factor
    ffac = 2.0 * df * np.sum(W_taper[nfreq_min: nfreq_max])
    Wp_w = W_taper / ffac
    Wq_w = W_taper / ffac

    # add error estimate
    # cc error
    if use_cc_error:
        Wp_w = Wp_w / (err_dt_cc * err_dt_cc)
        Wq_w = Wq_w / (err_dlna_cc * err_dlna_cc)

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
        Wp_w[nfreq_min: nfreq_max] = Wp_w[nfreq_min: nfreq_max] / \
            ((err_dtau_mt[nfreq_min: nfreq_max]) ** 2)
        Wq_w[nfreq_min: nfreq_max] = Wq_w[nfreq_min: nfreq_max] / \
            ((err_dlna_mt[nfreq_min: nfreq_max]) ** 2)

    # YY Ruan for test 10/28/2015
    # don't weight with err
    Wp_w = W_taper / ffac
    Wq_w = W_taper / ffac
    

    # initialization
    bottom_p = np.zeros(nlen_F, dtype=complex)
    bottom_q = np.zeros(nlen_F, dtype=complex)
    d2_tw  = np.zeros((nlen_F, ntaper), dtype=complex)
    d2_tvw = np.zeros((nlen_F, ntaper), dtype=complex)

    # Multitaper measurements
    for itaper in range(0, ntaper):
        taper = np.zeros(nlen_F)
        taper[0:nlen_T] = tapers[0:nlen_T, itaper]

        # multi-tapered measurements
        d2_t = np.zeros(nlen_T)
        d2_tv = np.zeros(nlen_T)
        d2_t[0:nlen_T] = d2[0:nlen_T] * taper[0:nlen_T]
        d2_tv = np.gradient(d2_t) / deltat

        # apply FFT to tapered measurements
        d2_tw[:,  itaper] = np.fft.fft(d2_t,  nlen_F)[:] * deltat
        d2_tvw[:, itaper] = np.fft.fft(d2_tv, nlen_F)[:] * deltat

        # calculate bottom of adjoint term pj(w) qj(w)
        bottom_p[:] = bottom_p[:] + d2_tvw[:, itaper] * d2_tvw[:, itaper].conjugate()
        bottom_q[:] = bottom_q[:] +  d2_tw[:, itaper] *  d2_tw[:, itaper].conjugate()

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
        q_w[nfreq_min:nfreq_max] = -d2_tw[nfreq_min:nfreq_max, itaper] / \
            (bottom_q[nfreq_min:nfreq_max])

        # calculate weighted adjoint Pj(w) Qj(w) adding measurement dtau dlna
        # P_w = np.zeros(nlen_F, dtype=complex)
        # P_w = np.zeros(nlen_F, dtype=complex)
        P_w = p_w * dtau_mtm * Wp_w
        Q_w = q_w * dlna_mtm * Wq_w

        # inverse FFT to weighted adjoint (take real part)
        # P_wt = np.zeros(nlen_F)
        # P_wt = np.zeros(nlen_F)
        P_wt = np.fft.ifft(P_w, nlen_F).real * 2 / deltat
        Q_wt = np.fft.ifft(Q_w, nlen_F).real * 2 / deltat

        # apply tapering to adjoint
        fp_t = fp_t + P_wt * taper
        fq_t = fq_t + Q_wt * taper

    return fp_t, fq_t


# Each adjoint source file must contain a calculate_adjoint_source()
# function. It must take observed, synthetic, min_period, max_period,
# left_window_border, right_window_border, adjoint_src, and figure as
# parameters. Other optional keywork arguments are possible.
def calculate_adjoint_source(observed, synthetic, config, window,
                             adjoint_src, figure):  # NOQA
    # There is no need to perform any sanity checks on the passed trace
    # object. At this point they will be guaranteed to have the same
    # sampling rate, be sampled at the same points in time and a couple
    # other things.

    # All adjoint sources will need some kind of taper. Thus pyadjoint has a
    # convenience function to assist with that. The next block tapers both
    # observed and synthetic data.
    
    # frequencies points for FFT
    nlen_F = 2**config.lnpt
    fnum = int(nlen_F/2 + 1)

    # constant for transfer function 
    wtr_mtm = config.transfunc_waterlevel
    wtr     = config.water_threshold

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
    dt_fac  = config.dt_fac
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

    #===
    # Loop over time windows 
    #===
    for wins in window:

        left_window_border  = wins[0]
        right_window_border = wins[1]

        # Youyi Ruan: for Test
        #print("window: %f %f" % (left_window_border,right_window_border))

        #taper_window(observed, left_window_border, right_window_border,
        #         taper_percentage=config.taper_percentage, taper_type=config.taper_type)
        #taper_window(synthetic, left_window_border, right_window_border,
        #         taper_percentage=config.taper_percentage, taper_type=config.taper_type)

        #===
        # pre-processing of the observed and sythetic to get windowed obsd and synt
        #===
        left_sample  = int(np.floor( left_window_border / deltat)) + 1
        nlen         = int(np.floor((right_window_border - left_window_border) / deltat)) + 1
        right_sample = left_sample + nlen

        d = np.zeros(nlen)
        s = np.zeros(nlen)

        d[0: nlen] =  observed.data[left_sample: right_sample]
        s[0: nlen] = synthetic.data[left_sample: right_sample]

        # Taper signals following the SAC taper command 
        sac_hann_taper(d, taper_percentage=config.taper_percentage)
        sac_hann_taper(s, taper_percentage=config.taper_percentage)

        # cross-correlation
        cc_shift  = _xcorr_shift(d, s)
        cc_tshift = cc_shift * deltat

        # uncertainty estimate based on cross-correlations
        sigma_dt_cc   = 1.0
        sigma_dlna_cc = 1.0

        # Y. Ruan: 01/27/2016 
        if use_cc_error:
            cc_dlna = 0.5 * np.log(sum(d[0:nlen]*d[0:nlen]) / sum(s[0:nlen]*s[0:nlen]))
            sigma_dt_cc, sigma_dlna_cc = cc_error(d, s, deltat, cc_shift, cc_dlna)

            print("cc_dt  : %f +/- %f" % (cc_tshift,sigma_dt_cc))
            print("cc_dlna: %f +/- %f" % (cc_dlna,sigma_dlna_cc))
        # re-window for obsd
        left_sample_d  = max(left_sample  + cc_shift, 0)
        right_sample_d = min(right_sample + cc_shift, nlen_data)
  
        nlen_d = right_sample_d - left_sample_d

        if nlen_d == nlen:
            # YY Ruan: No need to add cc_dlna in multitaper measurements
            cc_dlna = 0
            d[0:nlen] = np.exp(-cc_dlna) * observed.data[left_sample_d: right_sample_d]
            sac_hann_taper(d, taper_percentage=config.taper_percentage)
        else:
            raise Exception

        #===
        # Make decision wihich method to use: c.c. or multi-taper
        #===
        is_mtm = True

        # frequencies for FFT
        freq = np.fft.fftfreq(n=nlen_F, d=observed.stats.delta)
        df = freq[1] - freq[0]
        wvec = freq * 2 * np.pi
        dw = wvec[1] - wvec[0]

        # check window if okay for mtm measurements, and then find min/max 
        # frequency limit for calculations.
        nfreq_min, nfreq_max = frequency_limit(s, nlen, nlen_F, deltat, df, wtr, 
                ncycle_in_window, min_period, max_period, config.mt_nw, is_mtm)


        if is_mtm:
            # discrete prolate slepian sequences
            # The time half bandwidth parameter: nw (typical values are 2.5,3,3.5,4).
            nw = config.mt_nw
            ntaper = config.num_taper
            tapers = dpss_windows(nlen, nw, ntaper)[0].T

            # normalized
            tapers = tapers * np.sqrt(nlen)

            # calculate frequency-dependent phase and amplitude anomaly using
            # multi-taper approach
            phi_mtm = np.zeros(nlen_F)
            abs_mtm = np.zeros(nlen_F)
            # dtau_mtm = np.zeros(nlen_F)
            # dlna_mtm = np.zeros(nlen_F)
            # YY Ruan, phi_mt --> phi_mtm ? 10/29/2015
            phi_mtm, abs_mtm, dtau_mtm, dlna_mtm = mt_measure(d, s, deltat, tapers, 
                        wvec, df, nlen_F, wtr_mtm, phase_step,nfreq_min, 
                        nfreq_max, cc_tshift, cc_dlna)

            # multi-taper error estimation
            sigma_phi_mt  = np.zeros(nlen_F)
            sigma_abs_mt  = np.zeros(nlen_F)
            sigma_dtau_mt = np.zeros(nlen_F)
            sigma_dlna_mt = np.zeros(nlen_F)

            # Y. Ruan 01/27/2016
            #if use_mt_error:
            sigma_phi_mt, sigma_abs_mt, sigma_dtau_mt, sigma_dlna_mt = mt_error(
                        d, s, deltat, tapers, wvec, df, nlen_F, wtr_mtm, phase_step, 
                        nfreq_min, nfreq_max, cc_tshift, cc_dlna, phi_mtm, abs_mtm, 
                        dtau_mtm, dlna_mtm)

            # check if the multi-taper measurements okay.
            mt_measure_select(nfreq_min, nfreq_max, df, nlen, deltat, dtau_mtm, dt_fac, 
                        sigma_dtau_mt, err_fac, cc_tshift, dt_max_scale, is_mtm)

        if is_mtm:
            # calculate multi-taper adjoint source
            fp_t, fq_t = mt_adj(d, s, deltat, tapers, dtau_mtm, dlna_mtm, df, nlen_F,
                        use_cc_error, use_mt_error, nfreq_min, nfreq_max, 
                        sigma_dt_cc, sigma_dlna_cc, sigma_dtau_mt, sigma_dlna_mt)

        else:
            # calculate c.c. adjoint source
            fp_t, fq_t = cc_adj(synt, cc_shift, deltat, sigma_dt_cc, sigma_dlna_cc)

        #===
        # post-processing
        #===

        # return to original location before windowing
        # initialization
        fp_wind = np.zeros(len(synthetic.data))
        fq_wind = np.zeros(len(synthetic.data))

        fp_wind[left_sample: right_sample] = fp_t[0:nlen]
        fq_wind[left_sample: right_sample] = fq_t[0:nlen]

        fp = fp + fp_wind
        fq = fq + fq_wind

        # return misfit and adjoint source
        # Integrate with the composite Simpson's rule.
        #ret_val_p["misfit"] += 0.5 * simps(y=dtau_mtm ** 2, dx=dw)
        #ret_val_q["misfit"] += 0.5 * simps(y=dlna_mtm ** 2, dx=dw)
        misfit_sum_p += 0.5 * simps(y=dtau_mtm ** 2, dx=dw)
        misfit_sum_q += 0.5 * simps(y=dlna_mtm ** 2, dx=dw)

        #=== Youyi Ruan test ===
        #print(len(fp_wind),len(fp),nlen_data)
        #print(left_sample,right_sample)


    if adjoint_src is True:

        ret_val_p["misfit"] = misfit_sum_p
        ret_val_q["misfit"] = misfit_sum_q

        # Reverse in time and reverse the actual values.
        ret_val_p["adjoint_source"] = fp[::-1]
        ret_val_q["adjoint_source"] = fq[::-1]

        #=== added by YY Ruan for test
        #nlen_T = len(tapers[:,0])
        #print(nlen_T)
        #f0 = open('mtm.tpr.0','w')
        #f1 = open('mtm.tpr.1','w')
        #f2 = open('mtm.tpr.2','w')
        #f3 = open('mtm.tpr.3','w')
        #f4 = open('mtm.tpr.4','w')
       
        #for idx, ele in enumerate(tapers[0:nlen_T,0]):
        #  f0.write("%f %f\n" %(idx, ele))
        #for idx, ele in enumerate(tapers[0:nlen_T,1]):
        #  f1.write("%f %f\n" %(idx, ele))
        #for idx, ele in enumerate(tapers[0:nlen_T,2]):
        #  f2.write("%f %f\n" %(idx, ele))
        #for idx, ele in enumerate(tapers[0:nlen_T,3]):
        #  f3.write("%f %f\n" %(idx, ele))
        #for idx, ele in enumerate(tapers[0:nlen_T,4]):
        #  f4.write("%f %f\n" %(idx, ele))

        #f0.close()
        #f1.close()
        #f2.close()
        #f3.close()
        #f4.close()

        #f5 = open('fp.py.adj','w')
        #for idx, ele in enumerate(fp[left_sample:right_sample]):
          #f5.write("%f %f\n" %(idx*deltat, ele))
        #  f5.write("%f %f\n" %(idx, ele))
        #f5.close()
        #f15 = open('fq.py.adj','w')
        #for idx, ele in enumerate(fq[0:nlen_data]):
        #  f15.write("%f %f\n" %(idx*deltat, ele))
        #f15.close()

        #nlen_freq = len(dtau_mtm)

        #f6 = open('mtm.py.dtau','w')
        #f7 = open('mtm.py.dlna','w')
        #for idx, ele in enumerate(dtau_mtm[nfreq_min:nfreq_max+1]):
        #  f6.write("%f %f\n" %(idx*df, ele))
        #for idx, ele in enumerate(dlna_mtm[nfreq_min:nfreq_max+1]):
        #  f7.write("%f %f\n" %(idx*df, ele))
        #f6.close()
        #f7.close()

        #f6 = open('mtm.py.abs','w')
        #f7 = open('mtm.py.phi','w')
        #for idx, ele in enumerate(abs_mtm[0:nfreq_max]):
        #  f6.write("%f %f\n" %(idx*df, ele))
        #for idx, ele in enumerate(phi_mtm[0:nfreq_max]):
        #  f7.write("%f %f\n" %(idx*df, ele))
        #f6.close()
        #f7.close()

        #print("nfreq_min=",nfreq_min*df,"nfreq_max=",nfreq_max*df)

        #===


    # outputs (amplitude misfit and adjoint is optional)
    if figure:
        generic_adjoint_source_plot(
            observed, synthetic, ret_val_p["adjoint_source"],
            ret_val_p["misfit"], left_window_border, right_window_border,
            VERBOSE_NAME)

    if(config.measure_type == "dt"):
        return ret_val_p
    if(config.measure_type == "am"):
        return ret_val_q
