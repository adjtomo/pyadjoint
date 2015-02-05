#!/usr/bin/env python
# -*- encoding: utf8 -*-
"""
Simple waveform misfit and adjoint source.

This file will also serve as an explanation of how to add new adjoint
sources to Pyadjoint.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from scipy.integrate import simps

from ..utils import generic_adjoint_source_plot, taper_window


# This is the verbose and pretty name of the adjoint source defined in this
# function.
VERBOSE_NAME = "Waveform Misfit"

# Long and detailed description of the adjoint source defined in this file.
# Don't spare any details. This will be rendered as restructured text in the
# documentation.
DESCRIPTION = r"""
This is the simplest of all misfits and is defined as the squared difference
of observed and synthetic data. The misfit :math:`\chi(\bm{m})` for a given
Earth model :math:`\bm{m}` and a single receiver and component is given by

.. math::

    \chi (\bm{m}) = \frac{1}{2} \int_0^T \left| \bm{d}(t) - \bm{s}(t, \bm{m})
    \right| ^ 2 dt

:math:`\bm{d}(t)` is the observed data and :math:`\bm{s}(t, \bm{m})` the
synthetic data.

The adjoint source for the same receiver and component is given by

.. math::

    f^{\dag}(t) = - \left[ \bm{d}(T - t) - \bm{s}(T - t, \bm{m}) \right]

For the sake of simplicity we omit the spatial Kronecker delta and define
the source as acting only at the receiver's location.
"""


def calculate_adjoint_source(observed, synthetic, min_period, max_period,
                             left_window_border, right_window_border,
                             adjoint_src, figure, taper_percentage=0.15,
                             taper_type="hann"):  # NOQA
    """
    :param observed: The observed data.
    :type observed: :class:`obspy.core.trace.Trace`
    :param synthetic: The synthetic data.
    :type synthetic: :class:`obspy.core.trace.Trace`
    :param min_period: The minimum period of the spectral content of the data.
    :type min_period: float
    :param max_period: The maximum period of the spectral content of the data.
    :type max_period: float
    :param left_window_border: Left border of the window to be tapered in
        seconds since the first sample in the data arrays.
    :type left_window_border: float
    :param right_window_border: Right border of the window to be tapered in
        seconds since the first sample in the data arrays.
    :type right_window_border: float
    :param adjoint_src: Only calculate the misfit or also derive
        the adjoint source.
    :type adjoint_src: bool
    :param figure: If given, use it to plot a representation of the data and
        adjoint source to it.
    :type figure: :class:`matplotlib.figure.Figure`
    """
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

    ret_val = {}

    d = observed.data
    s = synthetic.data

    diff = d - s

    # Integrate with the composite Simpson's rule.
    ret_val["misfit"] = 0.5 * simps(y=diff ** 2, dx=observed.stats.delta)

    if adjoint_src is True:
        # Reverse in time and reverse the actual values.
        ret_val["adjoint_source"] = (-1.0 * diff)[::-1]

    if figure:
        generic_adjoint_source_plot(
            observed, synthetic, ret_val["adjoint_source"], ret_val["misfit"],
            left_window_border, right_window_border,
            VERBOSE_NAME)

    return ret_val
