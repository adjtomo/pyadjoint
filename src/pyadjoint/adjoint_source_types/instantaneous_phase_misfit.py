#!/usr/bin/env python
# -*- encoding: utf8 -*-
"""
Instantaneous phase method described in
(Yuan and Bozdag,2016) and adjoint source.

:copyright:
    Yanhua O. Yuan (yanhuay@princeton.edu), 2016
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from scipy.integrate import simps
from scipy import signal

from ..utils import generic_adjoint_source_plot
from ..utils import window_taper

import numpy as np

# This is the verbose and pretty name of the adjoint source defined in this
# function.
VERBOSE_NAME = "Instantaneous Phase Misfit"

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
        window_taper(d, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        window_taper(s, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)

        E_s = abs(signal.hilbert(s))
        E_d = abs(signal.hilbert(d))
        Hilbt_s = np.imag(signal.hilbert(s))
        Hilbt_d = np.imag(signal.hilbert(d))
        E_s_wtr = E_s + config.wtr_env * E_s.max()
        E_d_wtr = E_d + config.wtr_env * E_d.max()

        diff_real = s/E_s_wtr - d/E_d_wtr
        diff_imag = Hilbt_s/E_s_wtr - Hilbt_d/E_d_wtr

        # Integrate with the composite Simpson's rule.
        diff_r = diff_real * -1.0
        diff_i = diff_imag * -1.0
        window_taper(diff_r, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        window_taper(diff_i, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        # for some reason the 0.5 (see 2012 measure_adj mannual, P11) is
        # not in misfit definetion in measure_adj
        # misfit_sum += 0.5 * simps(y=diff_w**2, dx=deltat)
        misfit_sum += simps(y=diff_r**2, dx=deltat) \
            + simps(y=diff_i**2, dx=deltat)

        adj_real = diff_real * Hilbt_s**2 / E_s_wtr**3 \
            + np.imag(signal.hilbert(diff_real * s * Hilbt_s / E_s_wtr**3))
        adj_imag = - diff_imag * s * Hilbt_s / E_s_wtr**3 \
            - np.imag(signal.hilbert(diff_imag * s**2 / E_s_wtr**3))
        adj[left_sample: right_sample] = adj_real[0:nlen] + adj_imag[0:nlen]

    ret_val["misfit"] = misfit_sum

    if adjoint_src is True:
        # Reverse in time
        ret_val["adjoint_source"] = adj[::-1]

    if figure:
        # return NotImplemented
        generic_adjoint_source_plot(observed, synthetic,
                                    ret_val["adjoint_source"],
                                    ret_val["misfit"],
                                    window, VERBOSE_NAME)

    return ret_val
