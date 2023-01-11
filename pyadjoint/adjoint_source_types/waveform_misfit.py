#!/usr/bin/env python3
"""
Simple waveform-based misfit and adjoint source. Contains options for
'waveform' and 'convolution' misfit functions, as well as their double
difference counterparts

This one function takes care of implementation for multiple misfit functions:
    - 'waveform': Waveform misfit from [Tromp2005]_
    - 'convolution': Waveform convolution misfit from [Choi2011]_
    - 'waveform_dd': Double difference waveform misfit from [Yuan2016]_
    - 'convolution_dd': Double difference convolution misfit

:authors:
    adjTomo Dev Team (adjtomo@gmail.com), 2023
    Yanhua O. Yuan (yanhuay@princeton.edu), 2017
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
import numpy as np
from scipy.integrate import simps
from pyadjoint import logger
from pyadjoint.utils.signal import get_window_info, window_taper


def calculate_adjoint_source(observed, synthetic, config, windows,
                             observed_2=None, synthetic_2=None, windows_2=None):
    """
    Calculate adjoint source for the waveform-based misfit measurements.

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
    :type observed_2: obspy.core.trace.Trace
    :param observed_2: second observed waveform to calculate adjoint sources
        from station pairs
    :type synthetic_2:  obspy.core.trace.Trace
    :param synthetic_2: second synthetic waveform to calculate adjoint sources
        from station pairs
    :type windows_2: list of tuples
    :param windows_2: [(left, right),...] representing left and right window
        borders to be tapered in units of seconds since first sample in data
        array. Used to window `observed_2` and `synthetic_2`
    """
    assert(config.__class__.__name__ == "ConfigWaveform"), \
        "Incorrect configuration class passed to Waveform misfit"

    adjsrc_types = ["waveform", "convolution", "waveform_dd", "convolution_dd"]
    assert config.adjsrc_type in adjsrc_types, \
        f"`config.adjsrc_type` must be in {adjsrc_types}"
    logger.info(f"performing misfit caluclation with adjoint source type: "
                f"`{config.adjsrc_type}`")

    if config.double_difference:
        for val in [observed_2, synthetic_2, windows_2]:
            assert val is not None, (
                "Double difference measurements require a second set of "
                "waveforms and windows (`observed_2`, `synthetic_2`, "
                "`windows_2`)"
            )

    # Dictionary of values to be used to fill out the adjoint source class
    ret_val = {}

    # List of windows and some measurement values for each
    win_stats = []

    # Initiate constants and empty return values to fill
    nlen_data = len(synthetic.data)
    dt = synthetic.stats.delta
    adj = np.zeros(nlen_data)
    misfit_sum = 0.0

    # Sets up for double difference misfit types
    if config.double_difference:
        adj_2 = np.zeros(nlen_data)

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
        if config.double_difference:
            left_sample_2, right_sample_2, nlen_2 = \
                get_window_info(windows_2[i], dt)

            d_2 = np.zeros(nlen)
            s_2 = np.zeros(nlen)

            d_2[0: nlen_2] = observed_2.data[left_sample_2: right_sample_2]
            s_2[0: nlen_2] = \
                synthetic_2.data[left_sample_2: right_sample_2]

            # Taper DD measurements
            window_taper(d_2, taper_percentage=config.taper_percentage,
                         taper_type=config.taper_type)
            window_taper(s_2, taper_percentage=config.taper_percentage,
                         taper_type=config.taper_type)

            # Diff the two sets of waveforms
            if config.adjsrc_type == "waveform_dd":
                diff = (s - s_2) - (d - d_2)
            # Convolve the two sets of waveforms
            elif config.adjsrc_type == "convolution_dd":
                diff = np.convolve(s, d_2, "same") - np.convolve(d, s_2, "same")
            # Check at the top of function should avoid this
            else:
                raise NotImplementedError
        # Addressing a single set of waveforms
        else:
            # Convolve the two waveforms
            if config.adjsrc_type == "convolution":
                diff = np.convolve(s, d, "same")
            # Difference the two waveforms
            elif config.adjsrc_type == "waveform":
                diff = s - d
            else:
                raise NotImplementedError

        # Integrate with the composite Simpson's rule.
        misfit_win = 0.5 * simps(y=diff**2, dx=dt)
        misfit_sum += misfit_win

        # Taper again for smooth connection of windows adjoint source
        # with the full adjoint source
        window_taper(diff, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)

        # Include some information about each window's total misfit,
        # since its already calculated
        win_stats.append(
            {"type": config.adjsrc_type, "left": left_sample * dt,
             "right": right_sample * dt, "misfit": misfit_win,
             "difference": np.mean(diff)}
        )
        adj[left_sample: right_sample] = diff[0:nlen]

        # If doing differential measurements, add some information about
        # second set of waveforms
        if config.double_difference:
            win_stats[i]["right_2"] = right_sample_2 * dt
            win_stats[i]["left_2"] = left_sample_2 * dt
            adj_2[left_sample_2: right_sample_2] = -1 * diff[0:nlen_2]

    # Finally, set the return dictionary
    ret_val["misfit"] = misfit_sum
    ret_val["window_stats"] = win_stats
    ret_val["adjoint_source"] = adj[::-1]
    if config.double_difference:
        ret_val["adjoint_source_2"] = adj_2[::-1]

    return ret_val
