#!/usr/bin/env python
# -*- encoding: utf8 -*-
"""
Multitaper based phase and amplitude misfit and adjoint source.

This file will also serve as an explanation of how to add new adjoint
sources to Pyadjoint.

:copyright:
    Yanhua Yuan (yanhuay@princeton.edu), 2015
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np

from scipy.integrate import simps

from ..utils import generic_adjoint_source_plot, taper_window

from spectrum import dpss

import matplotlib.pylab as plt


# This is the verbose and pretty name of the adjoint source defined in this
# function.
VERBOSE_NAME = "Multitaper Misfit"

# Long and detailed description of the adjoint source defined in this file.
# Don't spare any details. This will be rendered as restructured text in the
# documentation. Be careful to escape the string with an ``r`` prefix.
# Otherwise most backslashes will have a special meaning which messes with the
# TeX like formulas.
DESCRIPTION = r"""
This is the simplest of all misfits and is defined as the squared difference
between observed and synthetic data. The misfit :math:`\chi(\mathbf{m})` for a
given Earth model :math:`\mathbf{m}` and a single receiver and component is
given by

.. math::

    \chi (\mathbf{m}) = \frac{1}{2} \int_0^T \left| \mathbf{d}(t) -
    \mathbf{s}(t, \mathbf{m}) \right| ^ 2 dt

:math:`\mathbf{d}(t)` is the observed data and
:math:`\mathbf{s}(t, \mathbf{m})` the synthetic data.

The adjoint source for the same receiver and component is given by

.. math::

    f^{\dagger}(t) = - \left[ \mathbf{d}(T - t) -
    \mathbf{s}(T - t, \mathbf{m}) \right]

For the sake of simplicity we omit the spatial Kronecker delta and define
the adjoint source as acting solely at the receiver's location. For more
details, please see [Tromp2005]_ and [Bozdag2011]_.

This particular implementation here uses
`Simpson's rule <http://en.wikipedia.org/wiki/Simpson's_rule>`_
to evaluate the definite integral.
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
display = False

def _xcorr_shift(d, s):
    cc = np.correlate(d, s, mode="full")
    time_shift = cc.argmax() - len(d) + 1
    return time_shift

def mt_measure(d1, d2, tapers, wvec, df, nfreq_max,cc_tshift,cc_dlnA):
    nlen_T = len(d1)
    ntaper = len(tapers[0])

    # initialization
    top_tf = np.zeros(nlen_F,dtype=complex)
    bottom_tf = np.zeros(nlen_F,dtype=complex)

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
        d1_tw = np.zeros(nlen_F,dtype=complex)
        d2_tw = np.zeros(nlen_F,dtype=complex)
        d1_tw = np.fft.fft(d1_t,nlen_F)
        d2_tw = np.fft.fft(d2_t,nlen_F)

        # calculate top and bottom of MT transfer function
        top_tf = top_tf + d1_tw * d2_tw.conjugate()
        bottom_tf = bottom_tf + d2_tw * d2_tw.conjugate()

    # Calculate transfer function
    # using top and bottom part of transfer function
    # water level
    wtr_use = max(abs(bottom_tf[0:fnum])) * wtr_mtm **2
    # transfrer function
    trans_func = np.zeros(nlen_F,dtype=complex)
    trans_func[0:fnum] = top_tf[0:fnum] / (bottom_tf[0:fnum] + wtr_use * (abs(trans_func[0:fnum]) < wtr_use))

    # Estimate phase and amplitude anomaly from transfer function
    phi_w = np.zeros(nlen_F)
    abs_w = np.zeros(nlen_F)
    dtau_w = np.zeros(nlen_F)
    dlnA_w = np.zeros(nlen_F)
    phi_w[0:nfreq_max] = np.arctan2(trans_func[0:nfreq_max].imag, trans_func[0:nfreq_max].real)
    abs_w[0:nfreq_max] = np.abs(trans_func[0:nfreq_max])
    # cycle-skipping (check smoothness of phi, add cc measure, future implementation)
    for iw in range(1, nfreq_max-1) :
        smth =  abs(phi_w[iw+1] + phi_w[iw-1] - 2.0 * phi_w[iw])
        smth1 = abs((phi_w[iw+1]+2*np.pi) + phi_w[iw-1] - 2.0 * phi_w[iw])
        smth2 = abs((phi_w[iw+1]-2*np.pi) + phi_w[iw-1] - 2.0 * phi_w[iw])
        if ( smth1 > smth and smth1 > smth2 and abs(phi_w[iw]-phi_w[iw+1]) > PHASE_STEP) :
            phi_w[iw+1:nfreq_max] = phi_w[iw+1:nfreq_max] + 2*np.pi
        if ( smth2 > smth and smth2 > smth1 and abs(phi_w[iw]-phi_w[iw+1]) > PHASE_STEP) :
            phi_w[iw+1:nfreq_max] = phi_w[iw+1:nfreq_max] - 2*np.pi    
    # add the CC measurements to the transfer function
    dtau_w[1:nfreq_max] = - 1.0 / wvec[1:nfreq_max] * phi_w[1:nfreq_max] + cc_tshift 
    dlnA_w[0:nfreq_max] = np.log(abs_w[0:nfreq_max]) + cc_dlnA 

    if(display == True) :
            freq = np.arange(0, nlen_F, 1) * df 
            d1_w = np.fft.fft(d1,nlen_F)
	    d2_w = np.fft.fft(d2,nlen_F)
	    plt.plot(freq, abs(d1_w))
	    plt.hold(True)
	    plt.plot(freq,abs(d2_w))
	    plt.plot(freq, abs(trans_func * d2_w ))
	    plt.xlim(0, (nfreq_max-1)*df)
	    plt.legend(['d(w)','s(w)','T(w)*s(w)'])
	    plt.title('spectral amplitude')
	    plt.xlabel('frequency (Hz)')
	    plt.show()

	    plt.plot(freq, dtau_w)
	    plt.hold(True)
	    plt.plot(freq, dlnA_w)
	    plt.xlim(0, (nfreq_max-1)*df)
	    plt.legend(['dtau','dlnA'])
	    plt.title('MT measurement')
	    plt.xlabel('frequency (Hz)')
	    plt.show()

    # multitaper error estimation (future implementation) 

    return phi_w, abs_w, dtau_w, dlnA_w

def mt_error(d1, d2, tapers, wvec,df, nfreq_max, cc_tshift, cc_dlnA, phi_mtm, abs_mtm, dtau_mtm, dlnA_mtm):
    nlen_T = len(d1)
    ntaper = len(tapers[0])

    # Jacknife MT estimates
    # initialization
    phi_mul = np.zeros((nlen_F,ntaper))
    abs_mul = np.zeros((nlen_F,ntaper))
    dtau_mul = np.zeros((nlen_F,ntaper))
    dlnA_mul = np.zeros((nlen_F,ntaper))
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
        tapers_om = np.zeros((nlen_T,ntaper-1))
        tapers_om[0:nlen_F,0:ntaper-1] = np.delete(tapers,itaper,1)
        # multitaper measurements with ntaper-1 tapers
        phi_om = np.zeros(nlen_F)
        abs_om = np.zeros(nlen_F)
        dtau_om = np.zeros(nlen_F)
        dlnA_om = np.zeros(nlen_F)
        phi_om, abs_om, dtau_om, dlnA_om = mt_measure(d1, d2, tapers_om, wvec, df, nfreq_max, cc_tshift, cc_dlnA)
        phi_mul[0:nlen_F,itaper] = phi_om[0:nlen_F]
        abs_mul[0:nlen_F,itaper] = abs_om[0:nlen_F]
        dtau_mul[0:nlen_F,itaper] = dtau_om[0:nlen_F]
        dlnA_mul[0:nlen_F,itaper] = dlnA_om[0:nlen_F]
        # error estimation 
        ephi_ave[0:nfreq_max] = ephi_ave[0:nfreq_max] + ntaper * phi_mtm[0:nfreq_max] -(ntaper-1) * phi_mul[0:nfreq_max,itaper]
        eabs_ave[0:nfreq_max] = eabs_ave[0:nfreq_max] + ntaper * abs_mtm[0:nfreq_max] -(ntaper-1) * abs_mul[0:nfreq_max,itaper]
        edtau_ave[0:nfreq_max] = edtau_ave[0:nfreq_max] + ntaper * dtau_mtm[0:nfreq_max] -(ntaper-1) * dtau_mul[0:nfreq_max,itaper]
        edlnA_ave[0:nfreq_max] = edlnA_ave[0:nfreq_max] + ntaper * dlnA_mtm[0:nfreq_max] -(ntaper-1) * dlnA_mul[0:nfreq_max,itaper]
    # take average 
    ephi_ave = ephi_ave /ntaper
    eabs_ave = eabs_ave /ntaper
    edtau_ave = edtau_ave /ntaper
    edlnA_ave = edlnA_ave /ntaper

    # deviation
    for itaper in range(0, ntaper):
        err_phi[0:nfreq_max] = err_phi[0:nfreq_max] + (phi_mul[0:nfreq_max,itaper] - ephi_ave[0:nfreq_max])**2
        err_abs[0:nfreq_max] = err_abs[0:nfreq_max] + (abs_mul[0:nfreq_max,itaper] - eabs_ave[0:nfreq_max])**2
        err_dtau[0:nfreq_max] = err_dtau[0:nfreq_max] + (dtau_mul[0:nfreq_max,itaper] - edtau_ave[0:nfreq_max])**2
        err_dlnA[0:nfreq_max] = err_dlnA[0:nfreq_max] + (dlnA_mul[0:nfreq_max,itaper] - edlnA_ave[0:nfreq_max])**2 
    # standard deviation (msre) 
    err_phi[0:nfreq_max] = np.sqrt(err_phi[0:nfreq_max]/(ntaper*(ntaper-1)))
    err_abs[0:nfreq_max] = np.sqrt(err_abs[0:nfreq_max]/(ntaper*(ntaper-1)))
    err_dtau[0:nfreq_max] = np.sqrt(err_dtau[0:nfreq_max]/(ntaper*(ntaper-1)))
   # err_dtau[0] = LARGE_VAL
    err_dlnA[0:nfreq_max] = np.sqrt(err_dlnA[0:nfreq_max]/(ntaper*(ntaper-1)))

    if(display == True) :
            freq = np.arange(0, nlen_F, 1) * df
            plt.subplot(2,1,1)
            plt.plot(freq,err_dtau)
            plt.hold(True)
            plt.xlim(0, (nfreq_max-1)*df)            
            plt.title('MT error estimation : err_dtau')
            plt.xlabel('frequency (Hz)')
            plt.subplot(2,1,2)
            plt.plot(freq,err_dlnA)
            plt.hold(True)
            plt.xlim(0, (nfreq_max-1)*df)
            plt.title('MT error estimation : err_dlnA')
            plt.xlabel('frequency (Hz)') 
            plt.show() 

    return err_phi, err_abs, err_dtau, err_dlnA 

def mt_adj(d1, d2, tapers, dtau_mtm, dlnA_mtm, df, nfreq_max,err_dtau,err_dlnA):
    nlen_T = len(d1)
    ntaper = len(tapers[0])

    # prepare frequency-domain taper based on reliable frequency band and error estimation (future development)
    W_taper = np.zeros(nlen_F)
    Wp_w = np.zeros(nlen_F)
    Wq_w = np.zeros(nlen_F)
    nfreq_min = 0
    iw = np.arange(nfreq_min, nfreq_max, 1) 
    W_taper[nfreq_min: nfreq_max] = 1.0 - np.cos(np.pi*(iw[0:len(iw)] - nfreq_min )/ (nfreq_max - nfreq_min)) ** ipwr_w
    # normalized factor 
    ffac = 2.0 * df * np.sum(W_taper[nfreq_min: nfreq_max])
    # add error estimate 
    dtau_wtr = WTR * np.sum(np.abs(dtau_mtm[nfreq_min: nfreq_max]))/(nfreq_max - nfreq_min)
    dlnA_wtr = WTR * np.sum(np.abs(dlnA_mtm[nfreq_min: nfreq_max]))/(nfreq_max - nfreq_min)
    wtr_use = max(abs(dtau_mtm[nfreq_min: nfreq_max])) * 0.01
    wtr_use = 0.035
    err_dtau[nfreq_min: nfreq_max] = err_dtau[nfreq_min: nfreq_max] + wtr_use
    wtr_use = max(abs(dlnA_mtm[nfreq_min: nfreq_max])) * 0.01
    err_dlnA[nfreq_min: nfreq_max] = err_dlnA[nfreq_min: nfreq_max] + wtr_use
    #err_dtau[nfreq_min: nfreq_max] = err_dtau[nfreq_min: nfreq_max] + dtau_wtr * (err_dtau[nfreq_min: nfreq_max] < dtau_wtr)
    #err_dlnA[nfreq_min: nfreq_max] = err_dlnA[nfreq_min: nfreq_max] + dlnA_wtr * (err_dlnA[nfreq_min: nfreq_max] < dlnA_wtr)
    
    min_err_dtau = np.min(err_dtau[nfreq_min: nfreq_max])
    min_err_dlnA = np.min(err_dlnA[nfreq_min: nfreq_max])
    Wp_w[nfreq_min: nfreq_max] = W_taper[nfreq_min: nfreq_max]  / ((err_dtau[nfreq_min: nfreq_max])**2)
    Wq_w[nfreq_min: nfreq_max] = W_taper[nfreq_min: nfreq_max] / ((err_dlnA[nfreq_min: nfreq_max])**2)
    Wp_w = Wp_w / ffac
    Wq_w = Wq_w / ffac
    if(display == True) :
            freq = np.arange(0, nlen_F, 1) * df
            plt.subplot(3,1,1)
            plt.plot(freq,W_taper)
            plt.hold(True)
            plt.xlim(0, (nfreq_max-1)*df)
            plt.title('frequency filter')
            plt.xlabel('frequency (Hz)')
            plt.subplot(3,1,2)
            plt.plot(freq,Wp_w)
            plt.hold(True)
            plt.xlim(0, (nfreq_max-1)*df)
            plt.title('frequency filter with err_dtau')
            plt.xlabel('frequency (Hz)')
            plt.subplot(3,1,3)
            plt.plot(freq,Wq_w)
            plt.hold(True)
            plt.xlim(0, (nfreq_max-1)*df)
            plt.title('frequency filter with err_dlnA')
            plt.xlabel('frequency (Hz)')
            plt.show()


    # initialization
    bottom_p = np.zeros(nlen_F,dtype=complex)
    bottom_q = np.zeros(nlen_F,dtype=complex)
    d2_tw = np.zeros((nlen_F, ntaper), dtype=complex)
    d2_tvw = np.zeros((nlen_F, ntaper), dtype=complex)
    # Multitaper measurements
    for itaper in range(0, ntaper):
        taper = np.zeros(nlen_F)
        taper[0:nlen_T] = tapers[0:nlen_T, itaper]
        # multi-tapered measurements
        d2_t = np.zeros(nlen_T)
        d2_tv = np.zeros(nlen_T)
        d2_t[0:nlen_T] = d2[0:nlen_T] * taper[0:nlen_T]
        d2_tv = np.gradient(d2_t)

        # apply FFT to tapered measurements
        d2_tw[:,itaper] = np.fft.fft(d2_t,nlen_F)[:]
        d2_tvw[:,itaper] = np.fft.fft(d2_tv,nlen_F)[:]
        # calculate bottom of adjoint term pj(w) qj(w)
        bottom_p[:] = bottom_p[:] + d2_tvw[:,itaper] * d2_tvw[:,itaper].conjugate()
        bottom_q[:] = bottom_p[:] + d2_tw[:,itaper] * d2_tw[:,itaper].conjugate()

    # Calculate adjoint source
    # initialization
    fp_t = np.zeros(nlen_F)
    fq_t = np.zeros(nlen_F)
  
    for itaper in range(0, ntaper):
        taper = np.zeros(nlen_F)
        taper[0:nlen_T] = tapers[0:nlen_T, itaper]

        # calculate pj(w) qj(w)
        p_w = np.zeros(nlen_F,dtype=complex)
        q_w = np.zeros(nlen_F,dtype=complex)
        p_w[0:nfreq_max] = d2_tvw[0:nfreq_max,itaper] /(bottom_p[0:nfreq_max])
        q_w[0:nfreq_max] = - d2_tw[0:nfreq_max,itaper] / (bottom_q[0:nfreq_max])

        # calculate weighted adjoint Pj(w) Qj(w) adding measurement dtau dlnA
        P_w = np.zeros(nlen_F,dtype=complex)
        P_w = np.zeros(nlen_F,dtype=complex)
        P_w = p_w * dtau_mtm * Wp_w 
        Q_w = q_w * dlnA_mtm * Wq_w

        # inverse FFT to weighted adjoint (take real part)
        P_wt = np.zeros(nlen_F)
        P_wt = np.zeros(nlen_F)
        P_wt = np.fft.ifft(P_w,nlen_F).real
        Q_wt = np.fft.ifft(Q_w,nlen_F).real

        # apply tapering to adjoint
        fp_t = fp_t + P_wt * taper
        fq_t = fq_t + Q_wt * taper

        if(display == False) :
	        plt.subplot(4,2,1)
	        plt.plot(P_wt)
	        plt.hold(True)
	        plt.xlim(0,nlen_T)
	        plt.title('Pj(t)')
                plt.subplot(4,2,2)
                plt.plot(Q_wt)
                plt.hold(True)
                plt.xlim(0,nlen_T)
                plt.title('Qj(t)')
	        plt.subplot(4,2,3)
	        plt.plot(P_wt*taper)
	        plt.hold(True)
	        plt.xlim(0,nlen_T)
	        plt.title('Pj(t)*hj(t)')
	        plt.subplot(4,2,4)
	        plt.plot(Q_wt*taper)
	        plt.hold(True)
	        plt.xlim(0,nlen_T)
	        plt.title('Qj(t)*hj(t)')
	        plt.subplot(4,2,5)
	        plt.plot(fp_t)
	        plt.hold(False)
	        plt.xlim(0,nlen_T)
	        plt.title('fp_t(t)')
                plt.subplot(4,2,6)
                plt.plot(fq_t)
                plt.hold(False)
                plt.xlim(0,nlen_T)
                plt.title('fq_t(t)')

    #plt.show()
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

    ret_val_phi = {}
    ret_val_abs = {}

    # initialization
    nlen_data = len(observed.data)
    fp = np.zeros(nlen_data)
    fq = np.zeros(nlen_data)

   # 
    left_window_border = 800 
    right_window_border = 900

# pre-processing of the observed and sythetic to get windowed obs and syn
    left_sample = int(left_window_border / observed.stats.delta)
    right_sample = int(right_window_border / observed.stats.delta)
    nlen = right_sample - left_sample
    time_window = np.hanning(nlen)
    d = np.zeros(nlen)
    s = np.zeros(nlen)
    d[0:nlen] = observed.data[left_sample: right_sample] * time_window[0:nlen]
    s[0:nlen] = synthetic.data[left_sample: right_sample] * time_window[0:nlen]
   
    if(display == True) : 
	    plt.subplot(2,1,1)
	    plt.plot(observed.data)
	    plt.hold(True)
	    plt.plot(synthetic.data)
	    plt.xlim(0, nlen_data)
	    plt.legend(['input data','input syn'])
	    plt.title('input data and synthetics')
 
	    plt.subplot(2,1,2)
	    plt.plot(d)
	    plt.hold(True)
	    plt.plot(s)
	    plt.xlim(0,nlen)
	    plt.title('windowed data and synthetics between %d and %d '%(left_sample,right_sample))
 
    # cross-correlation correction
    cc_shift=_xcorr_shift(d, s)
    cc_tshift = cc_shift * observed.stats.delta
    cc_dlnA = 0 
    #cc_dlnA = 0.5 * np.log( sum(d[0:nlen] * d[0:nlen]) / sum(s[0:nlen] * s[0:nlen] ) )
    print('cc shift={0},cc_tshift={1},cc_dlnA={2}'.format(cc_shift,cc_tshift,cc_dlnA))
    # window for obs
    left_sample_d = max(left_sample+cc_shift,0)
    right_sample_d = min(right_sample+cc_shift,nlen_data)
    nlen_d = right_sample_d - left_sample_d
    if( nlen_d == nlen) :
       d[0:nlen] = np.exp(-cc_dlnA) * observed.data[left_sample_d: right_sample_d] * time_window[0:nlen]
    else :
       exit()

    if(display == True) :
        plt.subplot(2,1,2)
        plt.plot(d)
        plt.legend(['windowed data','windowed syn','cc corrected data'])
        plt.show()
    
    # multi-taper
    is_mtm = True
    if is_mtm is True:
        # discrete prolate slepian sequences
        # The time half bandwidth parameter (typical values are 2.5,3,3.5,4).
        NW = 2.5
        # number of tapers
        ntaper = int(2 * NW)
        [tapers, eigens] = dpss(nlen, NW, ntaper)
        # normalized
        tapers = tapers/ np.sqrt(nlen)
        if(display == True) :
	        plt.plot(tapers)
	        plt.xlim(0, nlen)
	        plt.legend(['1st','2nd','3rd','4th','5th'])
	        plt.title('slepian sequences with LW=%1.1f'%NW)
	        plt.show()
        
        # frequencies for FFT
        freq = np.fft.fftfreq(n = nlen_F, d = observed.stats.delta )
        df = freq[1] -freq[0]
        wvec = freq * 2 * np.pi 
        dw = wvec[1] - wvec[0]
        
        # find the maximum frequency point for measurement
        # using the spectrum of untapered synthetics 
        s_w = np.zeros(nlen_F,dtype=complex)
        s_w = np.fft.fft(s,nlen_F)
        ampmax = max(abs(s_w[0:fnum]))
        i_ampmax = np.argmax(abs(s_w[0:fnum]))
        wtr_thrd = ampmax * WTR
        # initialization 
        nfreq_max = fnum
        is_search = 1
        for iw in range (0,fnum):
           if (iw > i_ampmax and abs(s_w[iw]) < wtr_thrd and is_search == 1 ):
               is_search = 0
               nfreq_max = iw
           if (iw > i_ampmax and abs(s_w[iw]) > 10*wtr_thrd and is_search == 0 ):
               is_search = 1
               nfreq_max = iw
        print('max reliable frequency sample={0}, freq_max={1}, df={2}, dw={3}'.format(nfreq_max,nfreq_max*dw,df,dw))

    # calculate frequency-dependent phase and amplitude anomaly using
    # multi-taper approach
    phi_mtm = np.zeros(nlen_F)
    abs_mtm = np.zeros(nlen_F)
    dtau_mtm = np.zeros(nlen_F)
    dlnA_mtm = np.zeros(nlen_F)
    phi_mt, abs_mtm, dtau_mtm, dlnA_mtm = mt_measure(d, s, tapers, wvec, df, nfreq_max, cc_tshift, cc_dlnA)
  
    # multi-taper error estimation 
    err_phi = np.zeros(nlen_F)
    err_abs = np.zeros(nlen_F)
    err_dtau = np.zeros(nlen_F)
    err_dlnA = np.zeros(nlen_F)
    err_phi, err_abs, err_dtau, err_dlnA = mt_error(d, s, tapers, wvec, df, nfreq_max, cc_tshift, cc_dlnA, phi_mtm, abs_mtm, dtau_mtm, dlnA_mtm)

    # calculate multi-taper adjoint source
    fp_t = np.zeros(nlen_F)
    fq_t = np.zeros(nlen_F)
    fp_t, fq_t = mt_adj(d, s, tapers, dtau_mtm, dlnA_mtm, df, nfreq_max, err_dtau, err_dlnA)

    # post-processing of adjoint source time-domain taper
    # and return to original location before windowing
    # initialization
    fp_wind = np.zeros(len(synthetic.data))
    fq_wind = np.zeros(len(synthetic.data))
    fp_wind[left_sample: right_sample] = fp_t[0:nlen] * time_window[0:nlen]
    fq_wind[left_sample: right_sample] = fq_t[0:nlen] * time_window[0:nlen]
    fp = fp + fp_wind
    fq = fq + fq_wind

    if(display == False) :
                plt.subplot(4,2,7)
                plt.plot(fp)
                plt.hold(False)
                plt.title('fp(t)')
                plt.xlim(left_sample, right_sample)
                plt.subplot(4,2,8)
                plt.plot(fq)
                plt.hold(False)
                plt.title('fq(t)')
                plt.xlim(left_sample, right_sample)
 
	#    plt.subplot(1,2,1)
	#    plt.plot(fp)
	#    plt.xlim(left_sample, right_sample)
	#    plt.title('multitaper adjoint fp(t)')
	#    plt.subplot(1,2,2)
	#    plt.plot(fq)
	#    plt.title('multitaper adjoint fq(t)')
 	#    plt.xlim(left_sample, right_sample)
	        plt.show()

    ### return misfit and adjoint source
    # Integrate with the composite Simpson's rule.
    ret_val_phi["misfit"] = 0.5 * simps(y=dtau_mtm ** 2, dx=dw)
    ret_val_abs["misfit"] = 0.5 * simps(y=dlnA_mtm ** 2, dx=dw)

    if adjoint_src is True:
        # Reverse in time and reverse the actual values.
        ret_val_phi["adjoint_source"] = fp[::-1]
        ret_val_abs["adjoint_source"] = fq[::-1]

    if figure:
        generic_adjoint_source_plot(
            observed, synthetic, ret_val_phi["adjoint_source"], ret_val_phi["misfit"],
            left_window_border, right_window_border,
            VERBOSE_NAME)
        plt.title('multitaper phase adjoint source')
        plt.show()
 
#        generic_adjoint_source_plot(
#            observed, synthetic, ret_val_abs["adjoint_source"], ret_val_abs["misfit"],
#            left_window_border, right_window_border,
#            VERBOSE_NAME)
#        plt.title('multitaper amplitude adjoint source')
#        plt.show()

    return ret_val_phi
