#!/usr/bin/env python
# -*- encoding: utf8 -*-
"""
Cross correlation traveltime misfit.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from ..utils import generic_adjoint_source_plot, taper_window


VERBOSE_NAME = "Cross Correlation Traveltime Misfit"

DESCRIPTION = r"""
"""

# Optional: document any additional parameters this particular adjoint sources
# receives in addition to the ones passed to the central adjoint source
# calculation function. Make sure to indicate the default values. This is a
# bit redundant but the only way I could figure out to make it work with the
#  rest of the architecture.
ADDITIONAL_PARAMETERS = r"""
**taper_percentage** (:class:`float`)
    Decimal percentage of taper at one end (ranging from ``0.0`` (0%) to
    ``0.5`` (50%)). Defauls to ``0.15``.

**taper_type** (:class:`float`)
    The taper type, supports anything :meth:`obspy.core.trace.Trace.taper`
    can use. Defaults to ``"hann"``.
"""


def _xcorr_shift(d, s):
    cc = np.correlate(d, s, mode="full")
    time_shift = cc.argmax() - len(d) + 1
    return time_shift


def calculate_adjoint_source(observed, synthetic, min_period, max_period,
                             left_window_border, right_window_border,
                             adjoint_src, figure, taper_percentage=0.15,
                             taper_type="hann"):  # NOQA

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

    time_shift = _xcorr_shift(d, s) * observed.stats.delta
    ret_val["misfit"] = time_shift

    if adjoint_src is True:
        # Reverse in time and reverse the actual values.
        ret_val["adjoint_source"] = \
            (-1.0 * synthetic.copy().differentiate().data)[::-1]

    if figure:
        generic_adjoint_source_plot(
            observed, synthetic, ret_val["adjoint_source"], ret_val["misfit"],
            left_window_border, right_window_border,
            VERBOSE_NAME)

    return ret_val
