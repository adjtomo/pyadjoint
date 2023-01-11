#!/usr/bin/env python3
"""
Cross correlation traveltime misfit and associated adjoint source.

:authors:
    adjtomo Dev Team (adjtomo@gmail.com), 2023
    Yanhua O. Yuan (yanhuay@princeton.edu), 2017
    Youyi Ruan (youyir@princeton.edu) 2016
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
import numpy as np

from pyadjoint import logger
from pyadjoint.utils.signal import get_window_info, window_taper
from pyadjoint.utils.cctm import (calculate_cc_shift, calculate_cc_adjsrc,
                                  calculate_dd_cc_shift, calculate_dd_cc_adjsrc)


def calculate_adjoint_source(observed, synthetic, config, windows,
                             observed_2=None, synthetic_2=None, windows_2=None):
    """
    Calculate adjoint source for the cross-correlation traveltime misfit
    measurement

    :type observed: obspy.core.trace.Trace
    :param observed: observed waveform to calculate adjoint source
    :type synthetic:  obspy.core.trace.Trace
    :param synthetic: synthetic waveform to calculate adjoint source
    :type config: pyadjoint.config.ConfigCCTraveltime
    :param config: Config class with parameters to control processing
    :type windows: list of tuples
    :param windows: [(left, right),...] representing left and right window
        borders to be used to calculate misfit and adjoint sources
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
    assert(config.__class__.__name__ == "ConfigCCTraveltime"), \
        "Incorrect configuration class passed to CCTraveltime misfit"

    if config.double_difference:
        for val in [observed_2, synthetic_2, windows_2]:
            assert val is not None, (
                "Double difference measurements require a second set of "
                "waveforms and windows (`observed_2`, `synthetic_2`, "
                "`windows_2`)"
            )

    logger.info(f"calculating misfit with adjoint source type "
                f"`{config.adjsrc_type}`")

    # Allow for measurement types related to `dt` (p) and `dlna` (q)
    ret_val_p = {}
    ret_val_q = {}

    # List of windows and some measurement values for each
    win_stats = []

    # Initiate constants and empty return values to fill
    nlen_data = len(synthetic.data)
    dt = synthetic.stats.delta

    # Initiate empty arrays for memory efficiency
    fp = np.zeros(nlen_data)
    fq = np.zeros(nlen_data)
    if config.double_difference:
        # Initiate empty arrays for memory efficiency
        fp_2 = np.zeros(nlen_data)
        fq_2 = np.zeros(nlen_data)

    misfit_sum_p = 0.0
    misfit_sum_q = 0.0

    # Loop over time windows and calculate misfit for each window range
    for i, window in enumerate(windows):
        # Convenience variables, quick access to information about time series
        dt = synthetic.stats.delta
        left_sample, right_sample, nlen_w = get_window_info(window, dt)

        # Pre-allocate arrays for memory efficiency
        d = np.zeros(nlen_w)
        s = np.zeros(nlen_w)

        # d and s represent the windowed data and synthetic arrays, respectively
        d[0: nlen_w] = observed.data[left_sample: right_sample]
        s[0: nlen_w] = synthetic.data[left_sample: right_sample]

        # Taper windowed signals in place
        window_taper(d, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        window_taper(s, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)

        if config.double_difference:
            # Prepare second set of waveforms the same as the first
            dt_2 = synthetic.stats.delta
            window_2 = windows_2[i]
            left_sample_2, right_sample_2, nlen_w_2 = \
                get_window_info(window_2, dt_2)

            # Pre-allocate arrays for memory efficiency
            d_2 = np.zeros(nlen_w)
            s_2 = np.zeros(nlen_w)

            # d and s represent the windowed data and synthetic arrays
            d_2[0: nlen_w_2] = observed_2.data[left_sample_2: right_sample_2]
            s_2[0: nlen_w_2] = synthetic_2.data[left_sample_2: right_sample_2]

            # Taper windowed signals in place
            window_taper(d_2, taper_percentage=config.taper_percentage,
                         taper_type=config.taper_type)
            window_taper(s_2, taper_percentage=config.taper_percentage,
                         taper_type=config.taper_type)

            # Calculate double difference time shift
            tshift, _, _, dlna, _, sigma_dt, sigma_dlna = \
                calculate_dd_cc_shift(d=d, s=s, d_2=d_2, s_2=s_2, dt=dt,
                                      **vars(config)
                                      )
            # Calculate misfit and adjoint source for the given window
            # TODO: Add in dlna misfit and adjoint source in below function
            misfit_p, misfit_q, fp_win, fp_win_2, fq_win, fq_win_2 = \
                calculate_dd_cc_adjsrc(s=s, s_2=s_2, tshift=tshift, 
                                       dlna=dlna, dt=dt, sigma_dt=sigma_dt,
                                       sigma_dlna=sigma_dlna, **vars(config)
                                       )
        else:
            # Calculate cross correlation time shift, amplitude anomaly and
            # errors. Config passed as **kwargs to control constants required
            # by function
            tshift, dlna, sigma_dt, sigma_dlna = calculate_cc_shift(
                d=d, s=s, dt=dt, **vars(config)
            )

            # Calculate misfit and adjoint source for the given window
            misfit_p, misfit_q, fp_win, fq_win = \
                calculate_cc_adjsrc(s=s, tshift=tshift, dlna=dlna, dt=dt,
                                    sigma_dt=sigma_dt, sigma_dlna=sigma_dlna,
                                    **vars(config)
                                    )

        # Sum misfit into the overall waveform misfit
        misfit_sum_p += misfit_p
        misfit_sum_q += misfit_q

        # Add windowed adjoint source to the full adjoint source waveform
        window_taper(fp_win, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        window_taper(fq_win, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        fp[left_sample:right_sample] = fp_win[:]
        fq[left_sample:right_sample] = fq_win[:]

        if config.double_difference:
            window_taper(fp_win_2, taper_percentage=config.taper_percentage,
                         taper_type=config.taper_type)
            window_taper(fq_win_2, taper_percentage=config.taper_percentage,
                         taper_type=config.taper_type)
            fp_2[left_sample_2:right_sample_2] = fp_win_2[:]
            fq_2[left_sample_2:right_sample_2] = fq_win_2[:]

        # Store some information for each window
        win_stats.append(
            {"type": config.adjsrc_type,
             "left": left_sample * dt, "right": right_sample * dt,
             "measurement_type": config.measure_type, "tshift": tshift,
             "misfit_dt": misfit_p, "sigma_dt": sigma_dt, "dlna": dlna,
             "misfit_dlna": misfit_q, "sigma_dlna": sigma_dlna,
             }
        )

    # Keep track of both misfit values but only returning one of them
    ret_val_p["misfit"] = misfit_sum_p
    ret_val_q["misfit"] = misfit_sum_q

    # Time reverse adjoint sources w.r.t synthetic waveforms
    ret_val_p["adjoint_source"] = fp[::-1]
    ret_val_q["adjoint_source"] = fq[::-1]
    if config.double_difference:
        ret_val_p["adjoint_source_2"] = fp_2[::-1]
        ret_val_q["adjoint_source_2"] = fq_2[::-1]

    if config.measure_type == "dt":
        ret_val = ret_val_p
    elif config.measure_type == "am":
        ret_val = ret_val_q
    
    ret_val["window_stats"] = win_stats

    return ret_val
