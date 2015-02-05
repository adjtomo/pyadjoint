#!/usr/bin/env python
# -*- encoding: utf8 -*-
"""
Utility functions for Pyadjoint.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""


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
