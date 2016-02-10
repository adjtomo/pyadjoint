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

from ..utils import generic_adjoint_source_plot
from ..utils import sac_hann_taper

import numpy as np

# This is the verbose and pretty name of the adjoint source defined in this
# function.
VERBOSE_NAME = "Waveform Misfit"

# Long and detailed description of the adjoint source defined in this file.
# Don't spare any details. This will be rendered as restructured text in the
# documentation. Be careful to escape the string with an ``r`` prefix.
# Otherwise most backslashes will have a special meaning which messes with the
# TeX like formulas.
DESCRIPTION = r"""
This is the simplest of all misfits and is defined as the squared difference
between observed and synthetic data. The misfit :math:`\chi(\mathbf{m})` for a
given Earth model :math:`\mathbf{m}` and a single receiver and component is
given by

.. math::

    \chi (\mathbf{m}) = \frac{1}{2} \int_0^T \left| \mathbf{d}(t) -
    \mathbf{s}(t, \mathbf{m}) \right| ^ 2 dt

:math:`\mathbf{d}(t)` is the observed data and
:math:`\mathbf{s}(t, \mathbf{m})` the synthetic data.

The adjoint source for the same receiver and component is given by

.. math::

    f^{\dagger}(t) = - \left[ \mathbf{d}(T - t) -
    \mathbf{s}(T - t, \mathbf{m}) \right]

For the sake of simplicity we omit the spatial Kronecker delta and define
the adjoint source as acting solely at the receiver's location. For more
details, please see [Tromp2005]_ and [Bozdag2011]_.

This particular implementation here uses
`Simpson's rule <http://en.wikipedia.org/wiki/Simpson's_rule>`_
to evaluate the definite integral.
"""

# Optional: document any additional parameters this particular adjoint sources
# receives in addition to the ones passed to the central adjoint source
# calculation function. Make sure to indicate the default values. This is a
# bit redundant but the only way I could figure out to make it work with the
# rest of the architecture of pyadjoint.
ADDITIONAL_PARAMETERS = r"""
**taper_percentage** (:class:`float`)
    Decimal percentage of taper at one end (ranging from ``0.0`` (0%) to
    ``0.5`` (50%)). Defaults to ``0.15``.

**taper_type** (:class:`str`)
    The taper type, supports anything :method:`obspy.core.trace.Trace.taper`
    can use. Defaults to ``"hann"``.
"""


# Each adjoint source file must contain a calculate_adjoint_source()
# function. It must take observed, synthetic, min_period, max_period,
# left_window_border, right_window_border, adjoint_src, and figure as
# parameters. Other optional keyword arguments are possible.
def calculate_adjoint_source(observed, synthetic, config, window,
                             adjoint_src, figure):  # NOQA

    ret_val = {}

    nlen_data = len(synthetic.data)
    deltat = synthetic.stats.delta

    adj = np.zeros(nlen_data)

    misfit_sum = 0.0

    # loop over time windows

    for wins in window:
        left_window_border = wins[0]
        right_window_border = wins[1]

        left_sample = int(np.floor(left_window_border / deltat)) + 1
        nlen = int(np.floor((right_window_border - left_window_border) /
                            deltat)) + 1
        right_sample = left_sample + nlen

        d = np.zeros(nlen)
        s = np.zeros(nlen)

        d[0: nlen] = observed.data[left_sample: right_sample]
        s[0: nlen] = synthetic.data[left_sample: right_sample]

        # All adjoint sources will need some kind of windowing taper
        # to get rid of kinks at two ends
        sac_hann_taper(d, taper_percentage=config.taper_percentage)
        sac_hann_taper(s, taper_percentage=config.taper_percentage)

        diff = s - d

        # Integrate with the composite Simpson's rule.
        misfit_sum += 0.5 * simps(y=diff ** 2, dx=deltat)

        adj[left_sample: right_sample] = diff[0:nlen]

    if adjoint_src is True:
        # Reverse in time
        ret_val["adjoint_source"] = adj[::-1]
        ret_val["misfit"] = misfit_sum

    if figure:
        generic_adjoint_source_plot(
            observed, synthetic, ret_val["adjoint_source"], ret_val["misfit"],
            left_window_border, right_window_border,
            VERBOSE_NAME)

    return ret_val
