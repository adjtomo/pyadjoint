"""
Cross correlation coefficient waveform misfit and adjoint source.

This file will also serve as an explanation of how to add new adjoint
sources to Pyadjoint.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
    Shi Yao (yaoshi229@gmail.com)
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
import numpy as np
from pyadjoint.utils.signal import window_taper

def calculate_adjoint_source(observed, synthetic, config, windows,
                             observed_2=None, synthetic_2=None, windows_2=None, 
                             adjoint_src=True, window_stats=True):
    """
    Calculate adjoint source for the Cross correlation coefficient misfit
    measurement

    :type observed: obspy.core.trace.Trace
    :param observed: observed waveform to calculate adjoint source
    :type synthetic:  obspy.core.trace.Trace
    :param synthetic: synthetic waveform to calculate adjoint source
    :type config: pyadjoint.config.ConfigCCC
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
    assert(config.__class__.__name__ == "ConfigTFPhase"), \
        "Incorrect configuration class passed to Time-Frequency Phase misfit"

    if config.double_difference:
        raise NotImplementedError(
            "Time-Frequency phase misfit does not have double difference "
            "capabilities"
        )
    
    # Dictionary of return values related to exponentiated phase
    ret_val = {}

    # List of windows and some measurement values for each
    win_stats = []

    nlen_data = len(synthetic.data)
    deltat = synthetic.stats.delta

    adj = np.zeros(nlen_data)

    misfit_sum = 0.0

    # loop over time windows
    for window in windows:

        left_window_border = window[0]
        right_window_border = window[1]

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

        #diff = s - d
        CC = np.dot(d,s)
        weight_2 = np.sqrt(np.dot(d, d) * np.dot(s, s))
        misfit = 1 - CC / weight_2
        
        A = np.dot(d, s) / np.dot(s, s)
        # print(A)
        diff_w = (d - A * s) / weight_2
        #diff_w = diff * -1.0
        window_taper(diff_w, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        # for some reason the 0.5 (see 2012 measure_adj mannual, P11) is
        # not in misfit definetion in measure_adj
        # misfit_sum += 0.5 * simps(y=diff_w**2, dx=deltat)
        misfit_sum += misfit

        adj[left_sample: right_sample] = diff_w

        win_stats.append(
            {"type": config.adjsrc_type, "measurement_type": "dt",
             "left": left_sample * deltat, "right": right_sample * deltat,
             "misfit" : misfit
            }
        )
    
    ret_val["misfit"] = misfit_sum

    if window_stats:
        ret_val["window_stats"] = win_stats

    if adjoint_src:
        # Reverse in time
        ret_val["adjoint_source"] = -adj[::-1]

    return ret_val        
