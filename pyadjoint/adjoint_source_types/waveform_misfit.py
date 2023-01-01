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

from pyadjoint import plot_adjoint_source
from pyadjoint.utils.signal import get_window_info, window_taper


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
def calculate_adjoint_source(observed, synthetic, config, windows,
                             adjoint_src=True, window_stats=True, plot=False,
                             double_difference=False, observed_dd=None,
                             synthetic_dd=None, windows_dd=None):
    """
    Calculate adjoint source for the waveform misfit measurement

    :type observed: obspy.core.trace.Trace
    :param observed: observed waveform to calculate adjoint source
    :type synthetic:  obspy.core.trace.Trace
    :param synthetic: synthetic waveform to calculate adjoint source
    :type config: pyadjoint.config.ConfigWaveform
    :param config: Config class with parameters to control processing
    :type windows: list of tuples
    :param windows: [(left, right),...] representing left and right window
        borders to be tapered in units of seconds since first sample in data
        array
    :type adjoint_src: bool
    :param adjoint_src: flag to calculate adjoint source, if False, will only
        calculate misfit
    :type window_stats: bool
    :param window_stats: flag to return stats for individual misfit windows used
        to generate the adjoint source
    :type plot: bool
    :param plot: generate a figure after calculating adjoint source
    :type double_difference: bool
    :param double_difference: flag to turn on double difference waveform
        misfit measurement. Requires `observed_dd`, `synthetic_dd`, `windows_dd`
    :type observed_dd: obspy.core.trace.Trace
    :param observed_dd: second observed waveform to calculate double difference
        adjoint source
    :type synthetic_dd:  obspy.core.trace.Trace
    :param synthetic_dd: second synthetic waveform to calculate double
        difference adjoint source
    :type windows_dd: list of tuples
    :param windows_dd: [(left, right),...] representing left and right window
        borders to be tapered in units of seconds since first sample in data
        array. Used to window `observed_dd` and `synthetic_dd`
    """
    assert(config.__class__.__name__ == "ConfigWaveform"), \
        "Incorrect configuration class passed to Waveform misfit"

    # Dictionary of values to be used to fill out the adjoint source class
    ret_val = {}

    # List of windows and some measurement values for each
    win_stats = []

    # Initiate constants and empty return values to fill
    nlen_data = len(synthetic.data)
    dt = synthetic.stats.delta
    adj = np.zeros(nlen_data)
    misfit_sum = 0.0

    if double_difference:
        adj_dd = np.zeros(nlen_data)

    # Loop over time windows and calculate misfit for each window range
    for i, window in enumerate(windows):
        left_sample, right_sample, nlen = get_window_info(window, dt)

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

        # Prepare double difference waveforms if requested.
        # Repeat the steps above for second set of waveforms
        if double_difference:
            left_sample_dd, right_sample_dd, nlen_dd = \
                get_window_info(windows_dd[i], dt)

            d_dd = np.zeros(nlen)
            s_dd = np.zeros(nlen)

            d_dd[0: nlen_dd] = observed.data[left_sample_dd: right_sample_dd]
            s_dd[0: nlen_dd] = synthetic.data[left_sample_dd: right_sample_dd]

            # Taper DD measurements
            window_taper(d_dd, taper_percentage=config.taper_percentage,
                         taper_type=config.taper_type)
            window_taper(s_dd, taper_percentage=config.taper_percentage,
                         taper_type=config.taper_type)

            # Diff the two sets of waveforms
            diff = (s - s_dd) - (d - d_dd)
        else:
            diff = s - d

        # Integrate with the composite Simpson's rule.
        misfit_win = 0.5 * simps(y=diff**2, dx=dt)
        misfit_sum += misfit_win

        # Taper again for smooth connection of windows adjoint source
        # with the full adjoint source
        window_taper(diff, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)

        # Include some information about each window's total misfit,
        # since its already calculated
        if double_difference:
            win_stats.append(
                {"type": "waveform_dd", "left": left_sample * dt,
                 "right": right_sample * dt, "left_dd": left_sample_dd * dt,
                 "right_dd": right_sample_dd, "misfit": misfit_win,
                 "difference": np.mean(diff)}
            )
            adj[left_sample:right_sample] = diff[0:nlen]
            adj_dd[left_sample_dd: right_sample_dd] = -1 * diff[0:nlen_dd]
        else:
            win_stats.append(
                {"type": "waveform", "left": left_sample * dt,
                 "right": right_sample * dt, "misfit": misfit_win,
                 "difference": np.mean(diff)}
            )
            adj[left_sample: right_sample] = diff[0:nlen]

    # Determine the amount of information to return relative to the misfit calc.
    ret_val["misfit"] = misfit_sum
    if window_stats is True:
        ret_val["window_stats"] = win_stats
    if adjoint_src is True:
        ret_val["adjoint_source"] = adj[::-1]
        if double_difference:
            ret_val["adjoint_source_dd"] = adj_dd[::-1]

    # Generate a figure if requested to
    if plot:
        plot_adjoint_source(observed, synthetic, ret_val["adjoint_source"],
                            ret_val["misfit"], windows, VERBOSE_NAME)
        if double_difference:
            plot_adjoint_source(observed_dd, synthetic_dd,
                                ret_val["adjoint_source_dd"], ret_val["misfit"],
                                windows_dd, VERBOSE_NAME)

    return ret_val
