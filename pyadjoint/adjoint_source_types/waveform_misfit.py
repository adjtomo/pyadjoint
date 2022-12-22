#!/usr/bin/env python3
"""
Simple waveform misfit and adjoint source.

.. note::
    This file serves as the template for generation of new adjoint sources.
    Copy-paste file and adjust name, description and underlying calculation
    function to generate new adjoint source.

:authors:
    adjTomo Dev Team (adjtomo@gmail.com), 2022
    Yanhua O. Yuan (yanhuay@princeton.edu), 2017
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
import numpy as np
from scipy.integrate import simps

from pyadjoint.utils import generic_adjoint_source_plot, window_taper


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
                             adjoint_src=True, window_stats=True, plot=False):
    """
    Calculate adjoint source for the waveform misfit measurement

    :type observed: obspy.core.trace.Trace
    :param observed: observed waveform to calculate adjoint source
    :type synthetic:  obspy.core.trace.Trace
    :param synthetic: synthetic waveform to calculate adjoint source
    :type config: pyadjoint.config.ConfigWaveform
    :param config: Config class with parameters to control processing
    :type window: list of tuples
    :param window: [(left, right),...] representing left and right window
        borders to be tapered in units of seconds since first sample in data
        array
    :type adjoint_src: bool
    :param adjoint_src: flag to calculate adjoint source, if False, will only
        calculate misfit
    :type plot: bool
    :param plot: generate a figure after calculating adjoint source
    """
    assert(config.__class__.__name__ == "ConfigWaveform"), \
        "Incorrect configuration class passed to Waveform misfit"

    # Dictionary of values to be used to fill out the adjoint source class
    ret_val = {}

    # List of windows and some measurement values for each
    win_stats = []

    # Initiate constants and empty return values to fill
    nlen_data = len(synthetic.data)
    deltat = synthetic.stats.delta
    adj = np.zeros(nlen_data)
    misfit_sum = 0.0

    # Loop over time windows and calculate misfit for each window range
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

        # Adjoint sources will need some kind of windowing taper to remove kinks
        # at window start and end times
        window_taper(d, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        window_taper(s, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        diff = s - d

        # Integrate with the composite Simpson's rule.
        misfit_win = 0.5 * simps(y=diff**2, dx=deltat)
        misfit_sum += misfit_win

        # Taper again for smooth connection of windows adjoint source
        # with the full adjoint source
        window_taper(diff, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)

        # Include some information about each window's total misfit,
        # since its already calculated
        win_stats.append(
            {"left": left_window_border, "right": right_window_border,
             "misfit": misfit_win, "difference": np.mean(diff)}
        )

        # Overwrite the adjoint source where the window is located
        adj[left_sample: right_sample] = diff[0:nlen]

    # Determine the amount of information to return relative to the misfit calc.
    ret_val["misfit"] = misfit_sum
    if window_stats is True:
        ret_val["window_stats"] = win_stats
    if adjoint_src is True:
        ret_val["adjoint_source"] = adj[::-1]

    # Generate a figure if requested to
    if plot:
        generic_adjoint_source_plot(
            observed, synthetic, ret_val["adjoint_source"], ret_val["misfit"],
            window, VERBOSE_NAME)

    return ret_val
