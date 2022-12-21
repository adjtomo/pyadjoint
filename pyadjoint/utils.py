#!/usr/bin/env python3
"""
Utility functions for Pyadjoint.

:copyright:
    adjTomo Dev Team (adjtomo@gmail.com), 2022
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
import inspect
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import obspy
import pkgutil
import warnings
from pyadjoint import PyadjointError, PyadjointWarning

EXAMPLE_DATA_PDIFF = (800, 900)
EXAMPLE_DATA_SDIFF = (1500, 1600)
TAPER_COLLECTION = ('cos', 'cos_p10', 'hann', "hamming")


def discover_adjoint_sources():
    """
    Discovers the available adjoint sources in the package. This should work
    regardless of whether Pyadjoint is checked out from git, packaged as .egg
    etc.
    """
    from pyadjoint import adjoint_source_types

    adjoint_sources = {}

    fct_name = "calculate_adjoint_source"
    name_attr = "VERBOSE_NAME"
    desc_attr = "DESCRIPTION"
    add_attr = "ADDITIONAL_PARAMETERS"

    path = os.path.join(
        os.path.dirname(inspect.getfile(inspect.currentframe())),
        "adjoint_source_types")
    for importer, modname, _ in pkgutil.iter_modules(
            [path], prefix=adjoint_source_types.__name__ + "."):
        m = importer.find_module(modname).load_module(modname)
        if not hasattr(m, fct_name):
            continue
        fct = getattr(m, fct_name)
        if not callable(fct):
            continue

        name = modname.split('.')[-1]

        if not hasattr(m, name_attr):
            raise PyadjointError(
                "Adjoint source '%s' does not have a variable named %s." %
                (name, name_attr))

        if not hasattr(m, desc_attr):
            raise PyadjointError(
                "Adjoint source '%s' does not have a variable named %s." %
                (name, desc_attr))

        # Add tuple of name, verbose name, and description.
        adjoint_sources[name] = (
            fct, getattr(m, name_attr), getattr(m, desc_attr),
            getattr(m, add_attr) if hasattr(m, add_attr) else None
        )

    return adjoint_sources


def sanity_check_waveforms(observed, synthetic):
    """
    Perform a number of basic sanity checks to assure the data is valid
    in a certain sense.

    It checks the types of both, the start time, sampling rate, number of
    samples, ...

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

    >>> from pyadjoint.utils import taper_window
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
    Window taper function to taper a time series with various taper functions

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


def get_example_data():
    """
    Helper function returning example data for Pyadjoint.

    The returned data is fully preprocessed and ready to be used with Pyadjoint.

    :returns: Tuple of observed and synthetic streams
    :rtype: tuple of :class:`obspy.core.stream.Stream` objects

    .. rubric:: Example

    >>> from pyadjoint.utils import get_example_data
    >>> observed, synthetic = get_example_data()
    >>> print(observed)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    3 Trace(s) in Stream:
    SY.DBO.S3.MXR | 2014-11-15T02:31:50.259999Z - ... | 1.0 Hz, 3600 samples
    SY.DBO.S3.MXT | 2014-11-15T02:31:50.259999Z - ... | 1.0 Hz, 3600 samples
    SY.DBO.S3.MXZ | 2014-11-15T02:31:50.259999Z - ... | 1.0 Hz, 3600 samples
    >>> print(synthetic)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    3 Trace(s) in Stream:
    SY.DBO..LXR   | 2014-11-15T02:31:50.259999Z - ... | 1.0 Hz, 3600 samples
    SY.DBO..LXT   | 2014-11-15T02:31:50.259999Z - ... | 1.0 Hz, 3600 samples
    SY.DBO..LXZ   | 2014-11-15T02:31:50.259999Z - ... | 1.0 Hz, 3600 samples
    """
    path = os.path.join(
        os.path.dirname(inspect.getfile(inspect.currentframe())),
        "example_data")
    observed = obspy.read(os.path.join(path, "observed_processed.mseed"))
    observed.sort()
    synthetic = obspy.read(os.path.join(path, "synthetic_processed.mseed"))
    synthetic.sort()

    return observed, synthetic


def generic_adjoint_source_plot(observed, synthetic, adjoint_source, misfit,
                                window, adjoint_source_name):
    """
    Generic plotting function for adjoint sources and data.

    Many types of adjoint sources can be represented in the same manner.
    This is a convenience function that can be called by different
    the implementations for different adjoint sources.

    :param observed: The observed data.
    :type observed: :class:`obspy.core.trace.Trace`
    :param synthetic: The synthetic data.
    :type synthetic: :class:`obspy.core.trace.Trace`
    :param adjoint_source: The adjoint source.
    :type adjoint_source: `numpy.ndarray`
    :param misfit: The associated misfit value.
    :float misfit: misfit value
    :type window: list of tuples
    :param window: [(left, right),...] representing left and right window
        borders to be tapered in units of seconds since first sample in data
        array
    :param adjoint_source_name: The name of the adjoint source.
    :type adjoint_source_name: str
    """
    x_range = observed.stats.endtime - observed.stats.starttime
    left_window_border = 60000.
    right_window_border = 0.

    for wins in window:
        left_window_border = min(left_window_border, wins[0])
        right_window_border = max(right_window_border, wins[1])

    buf = (right_window_border - left_window_border) * 0.3
    left_window_border -= buf
    right_window_border += buf
    left_window_border = max(0, left_window_border)
    right_window_border = min(x_range, right_window_border)

    plt.subplot(211)
    plt.plot(observed.times(), observed.data, color="0.2", label="Observed",
             lw=2)
    plt.plot(synthetic.times(), synthetic.data, color="#bb474f",
             label="Synthetic", lw=2)
    for wins in window:
        re = patches.Rectangle((wins[0], plt.ylim()[0]),
                               wins[1] - wins[0],
                               plt.ylim()[1] - plt.ylim()[0],
                               color="blue", alpha=0.1)
        plt.gca().add_patch(re)

    plt.grid()
    plt.legend(fancybox=True, framealpha=0.5)
    plt.xlim(left_window_border, right_window_border)
    ylim = max(map(abs, plt.ylim()))
    plt.ylim(-ylim, ylim)

    plt.subplot(212)
    plt.plot(observed.times(), adjoint_source[::-1], color="#2f8d5b", lw=2,
             label="Adjoint Source")
    plt.grid()
    plt.legend(fancybox=True, framealpha=0.5)
    # No time reversal for comparison with data
    # plt.xlim(x_range - right_window_border, x_range - left_window_border)
    # plt.xlabel("Time in seconds since first sample")
    plt.xlim(left_window_border, right_window_border)
    plt.xlabel("Time (seconds)")
    ylim = max(map(abs, plt.ylim()))
    plt.ylim(-ylim, ylim)

    plt.suptitle("%s Adjoint Source with a Misfit of %.3g" % (
        adjoint_source_name, misfit))
