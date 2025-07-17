"""
General utility functions used to calculate misfit and adjoint sources for the
cross correlation traveltime misfit function
"""
import numpy as np
import warnings
from obspy.signal.cross_correlation import xcorr_pick_correction
from scipy.integrate import simpson
from pyadjoint import logger


def calculate_cc_shift(d, s, dt, use_cc_error=True, dt_sigma_min=1.0,
                       dlna_sigma_min=0.5, **kwargs):
    """
    Calculate cross-correlation traveltime misfit (time shift, amplitude
    anomaly) and associated errors, for a given window.
    This is accessed by both the CC and MTM measurement methods.

    .. note::
        Kwargs not used but allows Config class to pass relevant parameters
        without explicitely naming them in the function call

    :type d: np.array
    :param d: observed data to calculate cc shift and dlna
    :type s: np.array
    :param s: synthetic data to calculate cc shift and dlna
    :type dt: float
    :param dt: time sampling rate delta t units seconds
    :type use_cc_error: bool
    :param use_cc_error: use cross correlation errors for normalization
    :type dt_sigma_min: float
    :param dt_sigma_min: minimum travel time error allowed
    :type dlna_sigma_min: float
    :param dlna_sigma_min: minimum amplitude error allowed
    :rtype: tuple (float, float, float, float)
    :return: (time shift [s], amplitude anomaly, time shift error [s],
        amplitude anomaly error)
    """
    # Note that CC values may dramatically change with/without the tapering
    ishift = xcorr_shift(d, s)  # timeshift in unit samples
    tshift = ishift * dt  # timeshift in unit seconds
    dlna = 0.5 * np.log(sum(d[:] * d[:]) /
                        sum(s[:] * s[:]))  # amplitude anomaly

    # Uncertainty estimate based on cross-correlations to be used for norm.
    if use_cc_error:
        sigma_dt, sigma_dlna = calculate_cc_error(d=d, s=s, dt=dt,
                                                  cc_shift=tshift, dlna=dlna,
                                                  dt_sigma_min=dt_sigma_min,
                                                  dlna_sigma_min=dlna_sigma_min
                                                  )
        logger.debug("CC error: "
                     f"dt={tshift:.2f}+/-{sigma_dt:.2f}s; "
                     f"dlna = {dlna:.3f}+/-{sigma_dlna:.3f}"
                     )
    else:
        sigma_dt = 1.0
        sigma_dlna = 1.0

    return tshift, dlna, sigma_dt, sigma_dlna


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
    nnorm = simpson(y=dsdt * dsdt, dx=dt)
    # note: Princeton ver. of code has a -1 on `fp`  because they have a '-' on 
    #   `nnorm`. Current format follows original Krischer code implementation
    fp[0:n] = dsdt[0:n] * tshift / nnorm / sigma_dt ** 2

    mnorm = simpson(y=s * s, dx=dt)
    fq[0:n] = -1.0 * s[0:n] * dlna / mnorm / sigma_dlna ** 2

    return misfit_p, misfit_q, fp, fq


def calculate_dd_cc_shift(d, s, d_2, s_2, dt, use_cc_error=True,
                          dt_sigma_min=1.0, dlna_sigma_min=0.5, **kwargs):
    """
    Calculate double difference cross-correlation traveltime misfit
    (time shift, amplitude anomaly) and associated errors, for a given window.
    Slight variation on normal CC shift calculation

    TODO
     - DD dlna measurement was not properly calculated in the RDNO version

    Assumes d, s, d_2 and s_2 all have the same sampling rate

    .. note::
        Kwargs not used but allows Config class to pass relevant parameters
        without explicitely naming them in the function call

    :type d: np.array
    :param d: observed data to calculate cc shift and dlna
    :type s: np.array
    :param s: synthetic data to calculate cc shift and dlna
    :type dt: float
    :param dt: time sampling rate delta t units seconds
    :type d_2: np.array
    :param d_2: 2nd pair observed data to calculate cc shift and dlna
    :type s_2: np.array
    :param s_2: 2nd pair synthetic data to calculate cc shift and dlna
    :type use_cc_error: bool
    :param use_cc_error: use cross correlation errors for normalization
    :type dt_sigma_min: float
    :param dt_sigma_min: minimum travel time error allowed
    :type dlna_sigma_min: float
    :param dlna_sigma_min: minimum amplitude error allowed
    :rtype: tuple (float, float, float, float)
    :return: (time shift [s], amplitude anomaly, time shift error [s],
        amplitude anomaly error)
    """
    # Calculate time shift between 'observed' or 'data' waveforms
    ishift_obs = xcorr_shift(d, d_2)  # timeshift in unit samples
    tshift_obs = ishift_obs * dt  # timeshift in unit seconds

    # Calculate time shift between 'synthetic' waveforms
    ishift_syn = xcorr_shift(s, s_2)  # timeshift in unit samples
    tshift_syn = ishift_obs * dt  # timeshift in unit seconds

    # Overall shift is difference between differential measurements
    ishift_dd = ishift_syn - ishift_obs
    tshift = ishift_dd * dt

    # FIXME: !!! This is not properly calculated as a differential !!!
    dlna_obs = 0.5 * np.log(sum(d[:] * d[:]) /
                            sum(d_2[:] * d_2[:]))  # amplitude anomaly
    dlna_syn = 0.5 * np.log(sum(s[:] * s[:]) /
                            sum(s_2[:] * s_2[:]))  # amplitude anomaly

    # Uncertainty is estimated based on DATA cross correlation
    if use_cc_error:
        sigma_dt, sigma_dlna = calculate_cc_error(
            d=d, s=d_2, dt=dt, cc_shift=ishift_obs, dlna=dlna_obs,
            dt_sigma_min=dt_sigma_min, dlna_sigma_min=dlna_sigma_min
        )
        logger.debug("CC error: "
                     f"dt={tshift_obs:.2f}+/-{sigma_dt:.2f}s; "
                     f"dlna = {dlna_obs:.3f}+/-{sigma_dlna:.3f}"
                     )
    else:
        sigma_dt = 1.0
        sigma_dlna = 1.0

    return tshift, tshift_obs, tshift_syn, dlna_obs, dlna_syn, sigma_dt, \
        sigma_dlna


def calculate_dd_cc_adjsrc(s, s_2, tshift, dlna, dt, sigma_dt=1.,
                           sigma_dlna=0.5, **kwargs):
    """
    Calculate double difference cross corrrelation adjoint sources.

    TODO
        - Add dlna capability to this function

    .. note::
        Kwargs not used but allows Config class to pass relevant parameters
        without explicitely naming them in the function call

    :type s: np.array
    :param s: synthetic data array
    :type s_2: np.array
    :param s_2: second synthetic data array
    :type tshift: float
    :param tshift: measured dd time shift from `calculate_dd_cc_shift`
    :type dlna: float
    :param dlna: measured dd amplitude anomaly from `calculate_dd_cc_shift`
    :type dt: float
    :param dt: delta t, time sampling rate of `s`
    :type sigma_dt: float
    :param sigma_dt: traveltime error from `calculate_cc_shift`
    :type sigma_dlna: float
    :param sigma_dlna: amplitude anomaly error from `calculate_cc_shift`
    :rtype: (float, float, np.array, np.array, np.array, np.array)
    :return: (tshift misfit, dlna misfit, tshift adjsrc, dlna adjsrc,
        tshift adjsrc 2, dlna adjsrc 2)
    """
    # So that we don't have to pass this in as an argument
    ishift_dd = int(tshift / dt)  # time shift in samples

    n = len(s)

    # Initialize empty arrays for memory efficiency
    fp = np.zeros(n)  # time shift
    fp_2 = np.zeros(n)

    fq = np.zeros(n)  # amplitude anomaly
    fq_2 = np.zeros(n)

    # Calculate the misfit for both time shift and amplitude anomaly
    misfit_p = 0.5 * (tshift / sigma_dt) ** 2
    misfit_q = 0.5 * (dlna / sigma_dlna) ** 2

    # Calculate adjoint sources for both time shift and amplitude anomaly
    dsdt = np.gradient(s, dt)

    # Time shift and gradient the first set of synthetics in reverse time
    s_cc_dt, _ = cc_correction(s, -1 * ishift_dd, 0.)
    dsdt_cc = np.gradient(s_cc_dt, dt)

    # Time shift and gradient the second of synthetics
    s_2_cc_dt, _ = cc_correction(s_2, ishift_dd, 0.)
    dsdt_cc_2 = np.gradient(s_2_cc_dt, dt)

    # Integrate the product of gradients
    # FIXME: Is `dsdt` supposed to be `dsdt_cc`? Need to check equations
    nnorm = simpson(y=dsdt * dsdt_cc_2, dx=dt)

    # note: Princeton ver. of code has a -1 on `fp`  because they have a '-' on
    #   `nnorm`. Current format follows original Krischer code implementation
    fp[0:n] = -1 * dsdt_cc_2[0:n] * tshift / nnorm / sigma_dt ** 2  # -1
    fp_2[0:n] = +1 * dsdt_cc[0:n] * tshift / nnorm / sigma_dt ** 2  # +1

    return misfit_p, misfit_q, fp, fp_2, fq, fq_2


def cc_correction(s, cc_shift, dlna):
    """
    Apply a correction to synthetics by shifting in time by `cc_shift` samples
    and scaling amplitude by `dlna`. Provides the 'best fitting' synthetic
    array w.r.t data as realized by the cross correlation misfit function

    :type s: np.array
    :param s: synthetic data array
    :type cc_shift: int
    :param cc_shift: time shift (in samples) as calculated using cross a
        cross correlation
    :type dlna: float
    :param dlna: amplitude anomaly as calculated by amplitude anomaly eq.
    :rtype: (np.array, np.array)
    :return: (time shifted synthetic array, amplitude scaled synthetic array)
    """
    nlen_t = int(len(s))
    s_cc_dt = np.zeros(nlen_t)
    s_cc_dtdlna = np.zeros(nlen_t)

    for index in range(0, nlen_t):
        index_shift = index - int(cc_shift)

        if 0 <= index_shift < nlen_t:
            # corrected by c.c. shift
            s_cc_dt[index] = s[index_shift]

            # corrected by c.c. shift and amplitude
            s_cc_dtdlna[index] = np.exp(dlna) * s[index_shift]

    return s_cc_dt, s_cc_dtdlna


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
    :param cc_shift: total amount of cross correlation time shift in samples
    :type dlna: float
    :param dlna: amplitude anomaly calculated for cross-correlation measurement
    :type dt_sigma_min: float
    :param dt_sigma_min: minimum travel time error allowed
    :type dlna_sigma_min: float
    :param dlna_sigma_min: minimum amplitude error allowed
    """
    # Apply a scaling and time shift to the synthetic data
    s_cc_dt, s_cc_dtdlna = cc_correction(s, cc_shift, dlna)

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
