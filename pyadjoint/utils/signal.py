#!/usr/bin/env python3
"""
Utility functions for Pyadjoint.

:copyright:
    adjTomo Dev Team (adjtomo@gmail.com), 2022
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
import numpy as np
import obspy
import warnings
from pyadjoint import PyadjointError, PyadjointWarning, logger


EXAMPLE_DATA_PDIFF = (800, 900)
EXAMPLE_DATA_SDIFF = (1500, 1600)
TAPER_COLLECTION = ('cos', 'cos_p10', 'hann', "hamming")


def get_window_info(window, dt):
    """
    Convenience function to get window start and end times, and start and end
    samples. Repeated a lot throughout package so useful to keep it defined
    in one place.

    :type window: tuple, list
    :param window: (left sample, right sample) borders of window in sample
    :type dt: float
    :param dt: delta T, time step of time series
    :rtype: tuple (float, float, int)
    :return: (left border in sample, right border in sample, length of window
        in sample)
    """
    assert(window[1] >= window[0]), f"`window` is reversed in time"

    nlen = int(np.floor((window[1] - window[0]) / dt)) + 1  # unit: sample
    left_sample = int(np.floor(window[0] / dt))
    right_sample = left_sample + nlen

    return left_sample, right_sample, nlen


def sanity_check_waveforms(observed, synthetic):
    """
    Perform a number of basic sanity checks to assure the data is valid
    in a certain sense.

    It checks the types of both, the start time, sampling rate, number of
    samples, etc.

    :param observed: The observed data.
    :type observed: :class:`obspy.core.trace.Trace`
    :param synthetic: The synthetic data.
    :type synthetic: :class:`obspy.core.trace.Trace`

    :raises: :class:`~pyadjoint.PyadjointError`
    """
    if not isinstance(observed, obspy.Trace):
        # Also accept Stream objects.
        if isinstance(observed, obspy.Stream) and \
                len(observed) == 1:
            observed = observed[0]
        else:
            raise PyadjointError(
                "Observed data must be an ObsPy Trace object.")
    if not isinstance(synthetic, obspy.Trace):
        if isinstance(synthetic, obspy.Stream) and \
                len(synthetic) == 1:
            synthetic = synthetic[0]
        else:
            raise PyadjointError(
                "Synthetic data must be an ObsPy Trace object.")

    if observed.stats.npts != synthetic.stats.npts:
        raise PyadjointError("Observed and synthetic data must have the same "
                             "number of samples.")

    sr1 = observed.stats.sampling_rate
    sr2 = synthetic.stats.sampling_rate

    if abs(sr1 - sr2) / sr1 >= 1E-5:
        raise PyadjointError("Observed and synthetic data must have the same "
                             "sampling rate.")

    # Make sure data and synthetics start within half a sample interval.
    if abs(observed.stats.starttime - synthetic.stats.starttime) > \
            observed.stats.delta * 0.5:
        raise PyadjointError("Observed and synthetic data must have the same "
                             "starttime.")

    ptp = sorted([observed.data.ptp(), synthetic.data.ptp()])
    if ptp[1] / ptp[0] >= 5:
        warnings.warn("The amplitude difference between data and "
                      "synthetic is fairly large.", PyadjointWarning)

    # Also check the components of the data to avoid silly mistakes of
    # users.
    if len(set([observed.stats.channel[-1].upper(),
                synthetic.stats.channel[-1].upper()])) != 1:
        warnings.warn("The orientation code of synthetic and observed "
                      "data is not equal.")

    observed = observed.copy()
    synthetic = synthetic.copy()
    observed.data = np.require(observed.data, dtype=np.float64,
                               requirements=["C"])
    synthetic.data = np.require(synthetic.data, dtype=np.float64,
                                requirements=["C"])

    return observed, synthetic


def taper_window(trace, left_border_in_seconds, right_border_in_seconds,
                 taper_percentage, taper_type, **kwargs):
    """
    Helper function to taper a window within a data trace.
    This function modifies the passed trace object in-place.

    :param trace: The trace to be tapered.
    :type trace: :class:`obspy.core.trace.Trace`
    :param left_border_in_seconds: The left window border in seconds since
        the first sample.
    :type left_border_in_seconds: float
    :param right_border_in_seconds: The right window border in seconds since
        the first sample.
    :type right_border_in_seconds: float
    :param taper_percentage: Decimal percentage of taper at one end (ranging
        from ``0.0`` (0%) to ``0.5`` (50%)).
    :type taper_percentage: float
    :param taper_type: The taper type, supports anything
        :meth:`obspy.core.trace.Trace.taper` can use.
    :type taper_type: str

    Any additional keyword arguments are passed to the
    :meth:`obspy.core.trace.Trace.taper` method.

    .. rubric:: Example

    >>> import obspy
    >>> tr = obspy.read()[0]
    >>> tr.plot()

    .. plot::

        import obspy
        tr = obspy.read()[0]
        tr.plot()

    >>> from pyadjoint.utils.signal import taper_window
    >>> taper_window(tr, 4, 11, taper_percentage=0.10, taper_type="hann")
    >>> tr.plot()

    .. plot::

        import obspy
        from pyadjoint.utils import taper_window
        tr = obspy.read()[0]
        taper_window(tr, 4, 11, taper_percentage=0.10, taper_type="hann")
        tr.plot()

    """
    s, e = trace.stats.starttime, trace.stats.endtime
    trace.trim(s + left_border_in_seconds, s + right_border_in_seconds)
    trace.taper(max_percentage=taper_percentage, type=taper_type, **kwargs)
    trace.trim(s, e, pad=True, fill_value=0.0)
    # Enable method chaining.
    return trace


def window_taper(signal, taper_percentage, taper_type):
    """
    Window taper function to taper a time series with various taper functions.
    Affect arrays in place but also returns the array. Both will edit the array.

    :param signal: time series
    :type signal: ndarray(float)
    :param taper_percentage: total percentage of taper in decimal
    :type taper_percentage: float
    :param taper_type: select available taper type, options are:
        cos, cos_p10, hann, hamming
    :type taper_type: str
    :return: tapered `signal` array
    :rtype: ndarray(float)
    """
    # Check user inputs
    if taper_type not in TAPER_COLLECTION:
        raise ValueError(f"Window taper not supported, must be in "
                         f"{TAPER_COLLECTION}")
    if taper_percentage < 0 or taper_percentage > 1:
        raise ValueError("taper percentage must be 0 < % < 1")

    npts = len(signal)
    if taper_percentage == 0.0 or taper_percentage == 1.0:
        frac = int(npts*taper_percentage / 2.0)
    else:
        frac = int(npts*taper_percentage / 2.0 + 0.5)

    idx1 = frac
    idx2 = npts - frac
    if taper_type == "hann":
        signal[:idx1] *=\
            (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(0, frac) /
                                (2 * frac - 1)))
        signal[idx2:] *=\
            (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(frac, 2 * frac) /
                                (2 * frac - 1)))
    elif taper_type == "hamming":
        signal[:idx1] *=\
            (0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(0, frac) /
                                  (2 * frac - 1)))
        signal[idx2:] *=\
            (0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(frac, 2 * frac) /
                                  (2 * frac - 1)))
    elif taper_type == "cos":
        power = 1.
        signal[:idx1] *= np.cos(np.pi * np.arange(0, frac) /
                                (2 * frac - 1) - np.pi / 2.0) ** power
        signal[idx2:] *= np.cos(np.pi * np.arange(frac, 2 * frac) /
                                (2 * frac - 1) - np.pi / 2.0) ** power
    elif taper_type == "cos_p10":
        power = 10.
        signal[:idx1] *= 1. - np.cos(np.pi * np.arange(0, frac) /
                                     (2 * frac - 1)) ** power
        signal[idx2:] *= 1. - np.cos(np.pi * np.arange(frac, 2 * frac) /
                                     (2 * frac - 1)) ** power

    return signal


def process_cycle_skipping(phi_w, nfreq_max, nfreq_min, wvec, phase_step=1.5):
    """
    Check for cycle skipping by looking at the smoothness of phi

    :type phi_w: np.array
    :param phi_w: phase anomaly from transfer functions
    :type nfreq_min: int
    :param nfreq_min: minimum frequency for suitable MTM measurement
    :type nfreq_max: int
    :param nfreq_max: maximum frequency for suitable MTM measurement
    :type phase_step: float
    :param phase_step: maximum step for cycle skip correction (?)
    :type wvec: np.array
    :param wvec: angular frequency array generated from Discrete Fourier
        Transform sample frequencies
    """
    for iw in range(nfreq_min + 1, nfreq_max - 1):
        smth0 = abs(phi_w[iw + 1] + phi_w[iw - 1] - 2.0 * phi_w[iw])
        smth1 = \
            abs((phi_w[iw + 1] + 2 * np.pi) + phi_w[iw - 1] - 2.0 * phi_w[iw])
        smth2 = \
            abs((phi_w[iw + 1] - 2 * np.pi) + phi_w[iw - 1] - 2.0 * phi_w[iw])

        phase_diff = phi_w[iw] - phi_w[iw + 1]

        if abs(phase_diff) > phase_step:

            temp_period = 2.0 * np.pi / wvec[iw]

            if smth1 < smth0 and smth1 < smth2:
                logger.warning(f"2pi phase shift at {iw} T={temp_period} "
                               f"diff={phase_diff}")
                phi_w[iw + 1:nfreq_max] = phi_w[iw + 1:nfreq_max] + 2 * np.pi

            if smth2 < smth0 and smth2 < smth1:
                logger.warning(f"-2pi phase shift at {iw} T={temp_period} "
                               f"diff={phase_diff}")
                phi_w[iw + 1:nfreq_max] = phi_w[iw + 1:nfreq_max] - 2 * np.pi

    return phi_w

def matlab_range(start, stop, step):
    """
    Simple function emulating the behaviour of Matlab's colon notation.

    This is very similar to np.arange(), except that the endpoint is included
    if it would be the logical next sample. Useful for translating Matlab code
    to Python.
    """
    # Some tolerance
    if (abs(stop - start) / step) % 1 < 1e-7:
        return np.linspace(
            start, stop, int(round((stop - start) / step)) + 1, endpoint=True
        )
    return np.arange(start, stop, step)

def window_trace(trace, window, taper, taper_ratio, taper_type):
    """
    Helper function to taper a window within a data trace.

    This function modifies the passed trace object in-place.

    :param trace: The trace to be tapered.
    :type trace: :class:`obspy.core.trace.Trace`
    :param window: Tuples with UCTDateTime objects for start and end time
        and potentially a weight as well
    :type window: Tuple with UCTDateTime objects
    :param taper: True if you want to apply tapering
    :type taper: binary
    :param taper_percentage: Decimal percentage of taper at one end (ranging
        from ``0.0`` (0%) to ``0.5`` (50%)).
    :type taper_percentage: float
    :param taper_type: The taper type, supports anything
        :meth:`obspy.core.trace.Trace.taper` can use.
    :type taper_type: str

    Any additional keyword arguments are passed to the
    :meth:`obspy.core.trace.Trace.taper` method.
    """
    s, e = trace.stats.starttime, trace.stats.endtime
   # print(window)
   # print(trace)
    # print(s+window[0], s+window[1])
    # print(taper)
    trace.trim(s + window[0], s + window[1])
    # print(trace)
    # print(len(trace))
    # print(taper_ratio,taper_type)
    if taper:
        trace.taper(max_percentage=taper_ratio, type=taper_type, side='both')
   # print(len(trace))
    #
    trace.trim(s, e, pad=True, fill_value=0.0)
    # Enable method chaining.
    return trace
