#!/usr/bin/env python
# -*- encoding: utf8 -*-
"""
Utility functions for Pyadjoint.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
import inspect
import matplotlib.pyplot as plt
import os

import obspy
import numpy as np

EXAMPLE_DATA_PDIFF = (800, 900)
EXAMPLE_DATA_SDIFF = (1500, 1600)


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
    window taper function.

    :param signal: time series
    :type signal: ndarray(float)

    :param taper_percentage: total percentage of taper in decimal
    :type taper_percentage: float
    
    return : tapered input ndarray

    taper_type:
    1, cos
    2, cos_p10
    3, hann


    To do: 
    with options of more tapers
    """
    taper_collection = ('cos', 'cos_p10', 'hann')

    if taper_type not in taper_collection:
        raise ValueError("Window taper not supported")

    if taper_percentage < 0 or taper_percentage > 1:
        raise ValueError("Wrong taper percentage")

    npts = len(signal)

    if taper_percentage == 0.0 or taper_percentage == 1.0:
        frac = int(npts*taper_percentage / 2.0)
    else:
        frac = int(npts*taper_percentage / 2.0 + 0.5)

    idx1 = frac
    idx2 = npts - frac

    if taper_type == 'hann':
        signal[:idx1] *= (0.5 - 0.5 * np.cos(2.0 * np.pi *
                        np.arange(0, frac) / (2*frac-1)))
        signal[idx2:] *= (0.5 - 0.5 * np.cos(2.0 * np.pi *
                        np.arange(frac, 2*frac) / (2*frac-1)))

    if taper_type == 'cos':
        signal[:idx1] *= 1. - np.cos( np.pi * np.range(0, frac) /\
                                    (2*frac-1) )
        signal[idx2:] *= 1. - np.cos( np.pi * np.range(frac, 2*frac) /\
                                    (2*frac-1) )

    if taper_type == 'cos_p10':
        power = 10.
        signal[:idx1] *= 1. - np.cos( np.pi * np.range(0, frac) /\
                                    (2*frac-1) )**power
        signal[idx2:] *= 1. - np.cos( np.pi * np.range(frac, 2*frac) /\
                                    (2*frac-1) )**power
    return signal


def get_example_data():
    """
    Helper function returning example data for Pyadjoint.

    The returned data is fully preprocessed and ready to be used with Pyflex.

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
                                left_window_border, right_window_border,
                                adjoint_source_name):
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
    :param left_window_border: Left border of the window to be tapered in
        seconds since the first sample in the data arrays.
    :type left_window_border: float
    :param right_window_border: Right border of the window to be tapered in
        seconds since the first sample in the data arrays.
    :type right_window_border: float
    :param adjoint_source_name: The name of the adjoint source.
    :type adjoint_source_name: str
    """
    x_range = observed.stats.endtime - observed.stats.starttime
    buf = (right_window_border - left_window_border) * 1.0
    left_window_border -= buf
    right_window_border += buf
    left_window_border = max(0, left_window_border)
    right_window_border = min(x_range, right_window_border)

    plt.subplot(211)
    plt.plot(observed.times(), observed.data, color="0.2", label="Observed",
             lw=2)
    plt.plot(synthetic.times(), synthetic.data, color="#bb474f",
             label="Synthetic", lw=2)
    plt.grid()
    plt.legend(fancybox=True, framealpha=0.5)
    plt.xlim(left_window_border, right_window_border)
    ylim = max(map(abs, plt.ylim()))
    plt.ylim(-ylim, ylim)

    plt.subplot(212)
    plt.plot(observed.times(), adjoint_source, color="#2f8d5b", lw=2,
             label="Adjoint Source")
    plt.grid()
    plt.legend(fancybox=True, framealpha=0.5)
    plt.xlim(x_range - right_window_border, x_range - left_window_border)
    plt.xlabel("Time in seconds since first sample")
    ylim = max(map(abs, plt.ylim()))
    plt.ylim(-ylim, ylim)

    plt.suptitle("%s Adjoint Source with a Misfit of %.3g" % (
        adjoint_source_name, misfit))
