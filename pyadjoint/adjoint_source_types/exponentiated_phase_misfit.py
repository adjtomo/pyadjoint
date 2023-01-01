#!/usr/bin/env python3
"""
Exponentiated phase misfit function and adjoint source.

TODO
    - write description
    - add citation for bozdag paper
    - write additional parameters

:authors:
    adjtomo Dev Team (adjtomo@gmail.com), 2022
    Yanhua O. Yuan (yanhuay@princeton.edu), 2016
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
import numpy as np

from scipy import signal
from scipy.integrate import simps

from pyadjoint import plot_adjoint_source
from pyadjoint.utils.signal import window_taper

VERBOSE_NAME = "Exponentiated Phase Misfit"

DESCRIPTION = r"""
"""

ADDITIONAL_PARAMETERS = r"""
"""


def calculate_adjoint_source(observed, synthetic, config, windows,
                             adjoint_src=True, window_stats=True, plot=False):
    """
    Calculate adjoint source for the exponentiated phase misfit.

    :type observed: obspy.core.trace.Trace
    :param observed: observed waveform to calculate adjoint source
    :type synthetic:  obspy.core.trace.Trace
    :param synthetic: synthetic waveform to calculate adjoint source
    :type config: pyadjoint.config.ConfigCCTraveltime
    :param config: Config class with parameters to control processing
    :type windows: list of tuples
    :param windows: [(left, right),...] representing left and right window
        borders to be used to calculate misfit and adjoint sources
    :type adjoint_src: bool
    :param adjoint_src: flag to calculate adjoint source, if False, will only
        calculate misfit
    :type window_stats: bool
    :param window_stats: flag to return stats for individual misfit windows used
        to generate the adjoint source
    :type plot: bool
    :param plot: generate a figure after calculating adjoint source
    """
    assert(config.__class__.__name__ == "ExponentiatedPhase"), \
        "Incorrect configuration class passed to CCTraveltime misfit"

    # Dictionary of return values related to exponentiated phase
    ret_val = {}

    # List of windows and some measurement values for each
    win_stats = []

    # Initiate constants and empty return values to fill
    nlen_w_data = len(synthetic.data)
    dt = synthetic.stats.delta

    f = np.zeros(nlen_w_data)  # adjoint source

    misfit_sum = 0.0

    # loop over time windows
    for window in windows:
        left_sample, right_sample, nlen_w = get_window_info(window, dt)

        # Initiate empty window arrays for memory efficiency
        d = np.zeros(nlen_w)
        s = np.zeros(nlen_w)

        d[0: nlen_w] = observed.data[left_sample: right_sample]
        s[0: nlen_w] = synthetic.data[left_sample: right_sample]

        # Taper window to get rid of kinks at two ends
        window_taper(d, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        window_taper(s, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)

        # Calculate envelope and hilbert transform of data, synthetics
        env_s = abs(signal.hilbert(s))
        env_d = abs(signal.hilbert(d))
        hilbt_s = np.imag(signal.hilbert(s))
        hilbt_d = np.imag(signal.hilbert(d))

        # Determine water level threshold
        thresh_s = config.wtr_env * env_s.max()
        thresh_d = config.wtr_env * env_d.max()
        env_s_wtr = env_s + thresh_s
        env_d_wtr = env_d + thresh_d

        # Calculate differences between data and synthetic acct for waterlevel
        diff_real = d/env_d_wtr - s/env_s_wtr
        diff_imag = hilbt_d/env_d_wtr - hilbt_s/env_s_wtr

        # Integrate with the composite Simpson's rule.
        misfit_real = 0.5 * simps(y=diff_real**2, dx=deltat)
        misfit_imag = 0.5 * simps(y=diff_imag**2, dx=deltat)

        misfit_sum += misfit_real + misfit_imag

        env_s_wtr_cubic = env_s_wtr**3

        adj_real = (
            -1 * (diff_real * hilbt_s ** 2 / env_s_wtr_cubic) -
            np.imag(signal.hilbert(diff_real * s * hilbt_s / env_s_wtr_cubic))
        )
        adj_imag = (
            diff_imag * s * hilbt_s / env_s_wtr_cubic +
            np.imag(signal.hilbert(diff_imag * s**2 / env_s_wtr_cubic))
        )

        # Re-taper newly generated adjoint source
        window_taper(adj_real, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        window_taper(adj_imag, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)

        f[left_sample:right_sample] = adj_real[0:nlen_w] + adj_imag[0:nlen_w]

        win_stats.append(
            {"left": left_sample * dt, "right": right_sample * dt,
             "measurement_type": "exp_phase",
             "diff_real": np.mean(diff_real[0:nlen_w]),
             "diff_imag": np.mean(diff_imag[0:nlen_w]),
             "misfit_real": misfit_real, "misfit_imag": misfit_imag
             }
        )

    # Place return values in output dictionary
    ret_val["misfit"] = misfit_sum

    if window_stats:
        ret_val["measurement"] = measurement

    # Time reverse adjoint sources w.r.t synthetic waveforms
    if adjoint_src is True:
        ret_val["adjoint_source"] = f[::-1]

    if figure:
        plot_adjoint_source(observed, synthetic, ret_val["adjoint_source"],
                            ret_val["misfit"], windows, VERBOSE_NAME)

    return ret_val
