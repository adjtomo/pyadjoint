"""
An implementation of the time frequency phase misfit and adjoint source after
Fichtner et al. (2008). This is different from lasif version in order to account for
structure of the recent version of pyadjoint.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
    Yajian Gao (krischer@geophysik.uni-muenchen.de), 2021
    Shi Yao (yaoshi229@gmail.com), 2024
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import warnings

import numexpr as ne
import numpy as np
from obspy.signal.interpolation import lanczos_interpolation
from pyadjoint.utils.signal import get_window_info, window_taper
from pyadjoint.timefrequency_utils import utils
from pyadjoint.timefrequency_utils import time_frequency


def calculate_adjoint_source(observed, synthetic, config, windows,
                             observed_2=None, synthetic_2=None, windows_2=None, 
                             adjoint_src=True, window_stats=True):
    """
    Calculate adjoint source for the time-frequency phase misfit
    measurement

    :type observed: obspy.core.trace.Trace
    :param observed: observed waveform to calculate adjoint source
    :type synthetic:  obspy.core.trace.Trace
    :param synthetic: synthetic waveform to calculate adjoint source
    :type config: pyadjoint.config.ConfigTFPhase
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

    # Assumes that t starts at 0. Pad your data if that is not the case -
    # Parts with zeros are essentially skipped making it fairly efficient.
    t = observed.times(type="relative")
    max_criterion = 7.0
    taper = True
    taper_ratio = 0.3
    taper_type="cosine"
    assert t[0] == 0

    window_weight = 1.0

    # Initiate constants and empty return values to fill
    nlen_w_data = len(synthetic.data)
    dt = synthetic.stats.delta

    adj = np.zeros(nlen_w_data)  # adjoint source

    misfit_sum = 0.0

    # loop over time windows
    for window in windows:
        left_sample, right_sample, nlen_w = get_window_info(window, dt)

        # Initiate empty window arrays for memory efficiency
        observed = utils.window_trace(
            trace=observed,
            window=window,
            taper=taper,
            taper_ratio=taper_ratio,
            taper_type=taper_type
        )
        synthetic = utils.window_trace(
            trace=synthetic,
            window=window,
            taper=taper,
            taper_ratio=taper_ratio,
            taper_type=taper_type
        )
        
        # Internal sampling interval. Some explanations for this "magic" number.
        # LASIF's preprocessing allows no frequency content with smaller periods
        # than min_period / 2.2 (see function_templates/preprocesssing_function.py
        # for details). Assuming most users don't change this, this is equal to
        # the Nyquist frequency and the largest possible sampling interval to
        # catch everything is min_period / 4.4.
        #
        # The current choice is historic as changing does (very slightly) chance
        # the calculated misfit and we don't want to disturb inversions in
        # progress. The difference is likely minimal in any case. We might have
        # same aliasing into the lower frequencies but the filters coupled with
        # the TF-domain weighting will get rid of them in essentially all
        # realistically occurring cases.
        dt_new = max(float(int(config.min_period / 4.0)), t[1] - t[0])
        dt_old = t[1] - t[0]

        # new time axis
        ti = utils.matlab_range(t[0], t[-1], dt_new)
        # Make sure its odd - that avoid having to deal with some issues
        # regarding frequency bin interpolation. Now positive and negative
        # frequencies will always be all symmetric. Data is assumed to be
        # tapered in any case so no problem are to be expected.
        if not len(ti) % 2:
            ti = ti[:-1]

        # Interpolate both signals to the new time axis - this massively speeds
        # up the whole procedure as most signals are highly oversampled. The
        # adjoint source at the end is re-interpolated to the original sampling
        # points.        
        data = lanczos_interpolation(
            data=observed.data,
            old_start=t[0],
            old_dt=t[1] - t[0],
            new_start=t[0],
            new_dt=dt_new,
            new_npts=len(ti),
            a=8,
            window="blackmann",
            )
        synthetic_data = lanczos_interpolation(
            data=synthetic.data,
            old_start=t[0],
            old_dt=t[1] - t[0],
            new_start=t[0],
            new_dt=dt_new,
            new_npts=len(ti),
            a=8,
            window="blackmann",
            )
        original_time = t
        t = ti        

        # -------------------------------------------------------------------------
        # Compute time-frequency representations

        # Window width is twice the minimal period.
        width = 2.0 * config.min_period

        # Compute time-frequency representation of the cross-correlation
        _, _, tf_cc = time_frequency.time_frequency_cc_difference(
            t, data, synthetic_data, width
            )
        # Compute the time-frequency representation of the synthetic
        tau, nu, tf_synth = time_frequency.time_frequency_transform(
            t, synthetic_data, width
            )
        
        # -------------------------------------------------------------------------
        # compute tf window and weighting function

        # noise taper: down-weight tf amplitudes that are very low
        tf_cc_abs = np.abs(tf_cc)
        m = tf_cc_abs.max() / 10.0  # NOQA
        weight = ne.evaluate("1.0 - exp(-(tf_cc_abs ** 2) / (m ** 2))")
        nu_t = nu.T 

        # highpass filter (periods longer than max_period are suppressed
        # exponentially)
        weight *= 1.0 - np.exp(-((nu_t * config.max_period) ** 2))  

        # lowpass filter (periods shorter than min_period are suppressed
        # exponentially)
        nu_t_large = np.zeros(nu_t.shape)
        nu_t_small = np.zeros(nu_t.shape)
        thres = nu_t <= 1.0 / config.min_period
        nu_t_large[np.invert(thres)] = 1.0
        nu_t_small[thres] = 1.0
        weight *= (
            np.exp(-10.0 * np.abs(nu_t * config.min_period - 1.0)) * nu_t_large
            + nu_t_small
            )

        # normalization
        weight /= weight.max()                     

        # computation of phase difference, make quality checks and misfit ---------

        # Compute the phase difference.
        # DP = np.imag(np.log(m + tf_cc / (2 * m + np.abs(tf_cc))))
        DP = np.angle(tf_cc)

        # Attempt to detect phase jumps by taking the derivatives in time and
        # frequency direction. 0.7 is an emperical value.
        abs_weighted_DP = np.abs(weight * DP)
        _x = abs_weighted_DP.max()  # NOQA
        test_field = ne.evaluate("weight * DP / _x")

        criterion_1 = np.sum([np.abs(np.diff(test_field, axis=0)) > 0.7])
        criterion_2 = np.sum([np.abs(np.diff(test_field, axis=1)) > 0.7])
        criterion = np.sum([criterion_1, criterion_2])

        # Compute the phase misfit
        dnu = nu[1] - nu[0]          

        i = ne.evaluate("sum(weight ** 2 * DP ** 2)")

        phase_misfit = np.sqrt(i * dt_new * dnu) * window_weight

        misfit_sum += phase_misfit

        if np.isnan(phase_misfit):
            print("The phase misfit is NaN.")
            raise ValueError("Phase misfit cannot be NaN.")
        
        # The misfit can still be computed, even if not adjoint source is
        # available.
        if criterion > max_criterion:
            warning = (
            "Possible phase jump detected. Misfit included. No "
            "adjoint source computed. Criterion: %.1f - Max allowed "
            "criterion: %.1f" % (criterion, max_criterion)
            )
            warnings.warn(warning)

        # Make kernel for the inverse tf transform
        idp = ne.evaluate(
            "weight ** 2 * DP * tf_synth / (m + abs(tf_synth) ** 2)"
            )
        
        # Invert tf transform and make adjoint source
        ad_src, it, I = time_frequency.itfa(tau, idp, width)

        # Interpolate both signals to the new time axis
        # Pad with a couple of zeros in case some where lost in all
        # these resampling operations. The first sample should not
        # change the time.
        ad_src = lanczos_interpolation(
                data=np.concatenate([ad_src.imag, np.zeros(100)]),
                old_start=tau[0],
                old_dt=tau[1] - tau[0],
                new_start=original_time[0],
                new_dt=original_time[1] - original_time[0],
                new_npts=len(original_time),
                a=8,
                window="blackmann",
            )        
                     
        # Divide by the misfit and change sign.
        ad_src /= phase_misfit + np.spacing(1)
        ad_src = ad_src / ((t[1] - t[0]) ** 2) * dt_old
        
        # taper the adjoint source
        ad_src_short = ad_src[left_sample: right_sample]
        window_taper(ad_src_short, taper_percentage=config.taper_percentage,
            taper_type=config.taper_type)
        adj[left_sample: right_sample] = ad_src_short * window_weight        

        win_stats.append(
            {"type": config.adjsrc_type, "measurement_type": "dt",
             "left": left_sample * dt, "right": right_sample * dt,
             "misfit" : phase_misfit
            }
        )

    ret_val["misfit"] = misfit_sum

    if window_stats:
        ret_val["window_stats"] = win_stats
    
    if adjoint_src is True:
        # Reverse time and add a leading zero so the adjoint source has the
        # same length as the input time series.
        ret_val["adjoint_source"] = -adj[::-1]
        
    return ret_val
