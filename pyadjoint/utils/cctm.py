"""
General utility functions used to calculate misfit and adjoint sources for the
cross correlation traveltime misfit function
"""
import numpy as np
import warnings
from obspy.signal.cross_correlation import xcorr_pick_correction
from scipy.integrate import simps
from pyadjoint import logger
from pyadjoint.utils.signal import window_taper, get_window


def calculate_cc_shift(observed, synthetic, window=None, taper_percentage=0.3,
                       taper_type="hann", use_cc_error=True, dt_sigma_min=1.0,
                       dlna_sigma_min=0.5, **kwargs):
    """
    Calculate cross-correlation traveltime misfit (time shift, amplitude
    anomaly) and associated errors, for a given window.
    This is accessed by both the CC and MTM measurement methods.

    .. note::
        Kwargs not used but allows Config class to pass relevant parameters
        without explicitely naming them in the function call

    :type observed: obspy.core.trace.Trace
    :param observed: observed waveform to calculate adjoint source
    :type synthetic:  obspy.core.trace.Trace
    :param synthetic: synthetic waveform to calculate adjoint source
    :type window: tuple or list
    :param window: (left, right) representing left and right window borders
    :type taper_percentage: float
    :param taper_percentage: Percentage of a time window needs to be
    tapered at two ends, to remove the non-zero values for adjoint
    source and for fft.
    :type taper_type: str
    :param taper_type: Taper type, see `pyaadjoint.utils.TAPER_COLLECTION`
        for a list of available taper types
    :type use_cc_error: bool
    :param use_cc_error: use cross correlation errors for normalization
    :type dt_sigma_min: float
    :param dt_sigma_min: minimum travel time error allowed
    :type dlna_sigma_min: float
    :param dlna_sigma_min: minimum amplitude error allowed
    :rtype: tuple (np.array, np.array, float, float, float, float)
    :return: (windowed data, windowed synthetics, time shift [s], amplitude
        anomaly, time shift error [s], amplitude anomaly error)
    """
    # Convenience variables for quick access to information about time series
    dt = synthetic.stats.delta

    if not window:
        window = [0, len(synthetic.data) * dt]

    left_sample, right_sample, nlen_w = get_window(window, dt)

    # Pre-allocate arrays for memory efficiency
    d = np.zeros(nlen_w)
    s = np.zeros(nlen_w)

    # d and s represent the windowd data and synthetic arrays, respectively
    d[0: nlen_w] = observed.data[left_sample: right_sample]
    s[0: nlen_w] = synthetic.data[left_sample: right_sample]

    # Taper windowed signals in place
    window_taper(d, taper_percentage=taper_percentage, taper_type=taper_type)
    window_taper(s, taper_percentage=taper_percentage, taper_type=taper_type)

    # Note that CC values may dramatically change with/without the tapering
    ishift = xcorr_shift(d, s)  # timeshift in unit samples
    tshift = ishift * dt  # timeshift in unit seconds
    dlna = 0.5 * np.log(sum(d[0:nlen_w] * d[0:nlen_w]) /
                        sum(s[0:nlen_w] * s[0:nlen_w]))  # amplitude anomaly

    # Uncertainty estimate based on cross-correlations to be used for norm.
    if use_cc_error:
        sigma_dt, sigma_dlna = calculate_cc_error(d=d, s=s, dt=dt,
                                                  cc_shift=tshift, dlna=dlna,
                                                  dt_sigma_min=dt_sigma_min,
                                                  dlna_sigma_min=dlna_sigma_min
                                                  )
        logger.debug("calculated CC error: "
                     f"dt = {tshift} +/- {sigma_dt} s; "
                     f"dlna = {dlna} +/- {sigma_dlna}"
                     )
    else:
        sigma_dt = 1.0
        sigma_dlna = 1.0

    return d, s, tshift, dlna, sigma_dt, sigma_dlna


def calculate_cc_adjsrc(s, tshift, dlna, dt, sigma_dt=1., sigma_dlna=0.5,
                        **kwargs):
    """
    Calculate adjoint source and misfit of the cross correlation traveltime
    misfit function. This is accessed by both the CC and MTM measurement
    methods.

    .. note::
        Kwargs not used but allows Config class to pass relevant parameters
        without explicitely naming them in the function call

    :type s: np.array
    :param s: synthetic data array
    :type tshift: float
    :param tshift: measured time shift from `calculate_cc_shift`
    :type dlna: float
    :param dlna: measured amplitude anomaly from `calculate_cc_shift`
    :type dt: float
    :param dt: delta t, time sampling rate of `s`
    :type sigma_dt: float
    :param sigma_dt: traveltime error from `calculate_cc_shift`
    :type sigma_dlna: float
    :param sigma_dlna: amplitude anomaly error from `calculate_cc_shift`
    :rtype: (float, float, np.array, np.array)
    :return: (tshift misfit, dlna misfit, tshift adjsrc, dlna adjsrc)
    """
    n = len(s)

    # Initialize empty arrays for memory efficiency
    fp = np.zeros(n)
    fq = np.zeros(n)

    # Calculate the misfit for both time shift and amplitude anomaly
    misfit_p = 0.5 * (tshift / sigma_dt) ** 2
    misfit_q = 0.5 * (dlna / sigma_dlna) ** 2

    # Calculate adjoint sources for both time shift and amplitude anomaly
    dsdt = np.gradient(s, dt)
    nnorm = simps(y=dsdt * dsdt, dx=dt)
    fp[0:n] = -1.0 * dsdt[0:n] * tshift / nnorm / sigma_dt ** 2

    mnorm = simps(y=s * s, dx=dt)
    fq[0:n] = -1.0 * s[0:n] * dlna / mnorm / sigma_dlna ** 2

    return misfit_p, misfit_q, fp, fq


def calculate_cc_error(d, s, dt, cc_shift, dlna, dt_sigma_min=1.0,
                       dlna_sigma_min=0.5):
    """
    Estimate error for `dt` and `dlna` with uncorrelation assumption. Used for
    normalization of the traveltime measurement

    :type d: np.array
    :param d: observed time series array to calculate error for
    :type s: np.array
    :param s: synthetic time series array to calculate error for
    :type dt: float
    :param dt: delta t, time sampling rate
    :type cc_shift: int
    :param cc_shift: total amount of cross correlation time shift
    :type dlna: float
    :param dlna: amplitude anomaly calculated for cross-correlation measurement
    :type dt_sigma_min: float
    :param dt_sigma_min: minimum travel time error allowed
    :type dlna_sigma_min: float
    :param dlna_sigma_min: minimum amplitude error allowed
    """
    # Correct d by shifting cc_shift and scaling dlna
    nlen_t = len(d)
    s_cc_dt = np.zeros(nlen_t)
    s_cc_dtdlna = np.zeros(nlen_t)

    for index in range(0, nlen_t):
        index_shift = index - cc_shift

        if 0 <= index_shift < nlen_t:
            # corrected by c.c. shift
            s_cc_dt[index] = s[index_shift]

            # corrected by c.c. shift and amplitude
            s_cc_dtdlna[index] = np.exp(dlna) * s[index_shift]

    # time derivative of s_cc (velocity)
    s_cc_vel = np.gradient(s_cc_dtdlna, dt)

    # The estimated error for dt and dlna with uncorrelation assumption
    sigma_dt_top = np.sum((d - s_cc_dtdlna)**2)
    sigma_dt_bot = np.sum(s_cc_vel**2)

    sigma_dlna_top = sigma_dt_top
    sigma_dlna_bot = np.sum(s_cc_dt**2)

    sigma_dt = np.sqrt(sigma_dt_top / sigma_dt_bot)
    sigma_dlna = np.sqrt(sigma_dlna_top / sigma_dlna_bot)

    # Check that errors do not go below the pre-defined threshold value
    if sigma_dt < dt_sigma_min or np.isnan(sigma_dt):
        sigma_dt = dt_sigma_min

    if sigma_dlna < dlna_sigma_min or np.isnan(sigma_dlna):
        sigma_dlna = dlna_sigma_min

    return sigma_dt, sigma_dlna


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
