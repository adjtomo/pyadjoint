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
import os

import obspy


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
