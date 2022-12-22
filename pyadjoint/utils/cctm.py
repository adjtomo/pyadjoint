"""
General utility functions used for cross correlation traveltime misfit function
and other misfit functions which might require CC measurements
"""
import numpy as np
import warnings
from obspy.signal.cross_correlation import xcorr_pick_correction


def xcorr_shift(d, s):
    """
    Determine the required time shift for peak cross-correlation of two arrays

    :type d: np.array
    :param d: observed time series array
    :type s:  np.array
    :param s: synthetic time series array
    """
    cc = np.correlate(d, s, mode="full")
    time_shift = cc.argmax() - len(d) + 1
    return time_shift


def subsample_xcorr_shift(d, s):
    """
    Calculate the correlation time shift around the maximum amplitude of the
    synthetic trace `s` with subsample accuracy.

    :type d: obspy.core.trace.Trace
    :param d: observed waveform to calculate adjoint source
    :type s:  obspy.core.trace.Trace
    :param s: synthetic waveform to calculate adjoint source
    """
    # Estimate shift and use it as a guideline for the subsample accuracy shift.
    time_shift = xcorr_shift(d.data, s.data) * d.stats.delta

    # Align on the maximum amplitude of the synthetics.
    pick_time = s.stats.starttime + s.data.argmax() * s.stats.delta

    # Will raise a warning if the trace ids don't match which we don't care
    # about here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return xcorr_pick_correction(
            pick_time, s, pick_time, d, 20.0 * time_shift,
            20.0 * time_shift, 10.0 * time_shift)[0]


def cc_error(d1, d2, deltat, cc_shift, cc_dlna, dt_sigma_min, dlna_sigma_min):
    """
    Estimate error for `dt` and `dlna` with uncorrelation assumption. Used for
    normalization of the traveltime measurement

    :type d1: np.array
    :param d1: time series array to calculate error for
    :type d2: np.array
    :param d2: time series array to calculate error for
    :type cc_shift: int
    :param cc_shift: total amount of cross correlation time shift
    :type cc_dlna: float
    :param cc_dlna: amplitude anomaly calculated for cross-correlation
        measurement
    :type dt_sigma_min: float
    :param dt_sigma_min: minimum travel time error allowed
    :type dlna_sigma_min: float
    :param dlna_sigma_min: minimum amplitude error allowed
    """
    # Correct d by shifting cc_shift and scaling dlna
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

    if sigma_dt < dt_sigma_min or np.isnan(sigma_dt):
        sigma_dt = dt_sigma_min

    if sigma_dlna < dlna_sigma_min or np.isnan(sigma_dlna):
        sigma_dlna = dlna_sigma_min

    return sigma_dt, sigma_dlna
