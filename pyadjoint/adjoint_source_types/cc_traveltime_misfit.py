#!/usr/bin/env python3
"""
Cross correlation traveltime misfit and associated adjoint source.

:authors:
    adjtomo Dev Team (adjtomo@gmail.com), 2022
    Youyi Ruan (youyir@princeton.edu) 2016
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
import numpy as np

from pyadjoint import plot_adjoint_source
from pyadjoint.utils.signal import get_window_info, window_taper
from pyadjoint.utils.cctm import calculate_cc_shift, calculate_cc_adjsrc


VERBOSE_NAME = "Cross Correlation Traveltime Misfit"

DESCRIPTION = r"""
Traveltime misfits simply measure the squared traveltime difference. The
misfit :math:`\chi(\mathbf{m})` for a given Earth model :math:`\mathbf{m}`
and a single receiver and component is given by

.. math::

    \chi (\mathbf{m}) = \frac{1}{2} \left[ T^{obs} - T(\mathbf{m}) \right] ^ 2

:math:`T^{obs}` is the observed traveltime, and :math:`T(\mathbf{m})` the
predicted traveltime in Earth model :math:`\mathbf{m}`.

In practice traveltime are measured by cross correlating observed and
predicted waveforms. This particular implementation here measures cross
correlation time shifts with subsample accuracy with a fitting procedure
explained in [Deichmann1992]_. For more details see the documentation of the
:func:`~obspy.signal.cross_correlation.xcorr_pick_correction` function and the
corresponding
`Tutorial <http://docs.obspy.org/tutorial/code_snippets/xcorr_pick_correction.html>`_.


The adjoint source for the same receiver and component is then given by

.. math::

    f^{\dagger}(t) = - \left[ T^{obs} - T(\mathbf{m}) \right] ~ \frac{1}{N} ~
    \partial_t \mathbf{s}(T - t, \mathbf{m})

For the sake of simplicity we omit the spatial Kronecker delta and define
the adjoint source as acting solely at the receiver's location. For more
details, please see [Tromp2005]_ and [Bozdag2011]_.


:math:`N` is a normalization factor given by


.. math::

    N = \int_0^T ~ \mathbf{s}(t, \mathbf{m}) ~
    \partial^2_t \mathbf{s}(t, \mathbf{m}) dt

This particular implementation here uses
`Simpson's rule <http://en.wikipedia.org/wiki/Simpson's_rule>`_
to evaluate the definite integral.
"""

ADDITIONAL_PARAMETERS = r"""
**taper_percentage** (:class:`float`)
    Decimal percentage of taper at one end (ranging from ``0.0`` (0%) to
    ``0.5`` (50%)). Defauls to ``0.15``.

**taper_type** (:class:`float`)
    The taper type, supports anything :meth:`obspy.core.trace.Trace.taper`
    can use. Defaults to ``"hann"``.
"""


def calculate_adjoint_source(observed, synthetic, config, windows,
                             adjoint_src=True, window_stats=True, plot=False):
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
    :type adjoint_src: bool
    :param adjoint_src: flag to calculate adjoint source, if False, will only
        calculate misfit
    :type window_stats: bool
    :param window_stats: flag to return stats for individual misfit windows used
        to generate the adjoint source
    :type plot: bool
    :param plot: generate a figure after calculating adjoint source
    """
    assert(config.__class__.__name__ == "ConfigCCTraveltime"), \
        "Incorrect configuration class passed to CCTraveltime misfit"

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

    misfit_sum_p = 0.0
    misfit_sum_q = 0.0

    # Loop over time windows and calculate misfit for each window range
    for window in windows:
        left_sample, right_sample, nlen_w = get_window_info(window, dt)

        # Calculate cross correlation time shift, amplitude anomaly and errors
        # Config passed as **kwargs to control constants required by function
        d, s, tshift, dlna, sigma_dt, sigma_dlna = \
            calculate_cc_shift(observed=observed, synthetic=synthetic,
                               window=window, **vars(config)
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

        # Store some information for each window
        win_stats.append(
            {"left": left_sample * dt, "right": right_sample * dt,
             "measurement_type": config.measure_type, "tshift": tshift,
             "misfit_dt": misfit_p, "sigma_dt": sigma_dt, "dlna": dlna,
             "misfit_dlna": misfit_q, "sigma_dlna": sigma_dlna,
             }
        )

    # Keep track of both misfit values but only returning one of them
    ret_val_p["misfit"] = misfit_sum_p
    ret_val_q["misfit"] = misfit_sum_q

    # Time reverse adjoint sources w.r.t synthetic waveforms
    if adjoint_src is True:
        ret_val_p["adjoint_source"] = fp[::-1]
        ret_val_q["adjoint_source"] = fq[::-1]
    if window_stats is True:
        ret_val_p["window_stats"] = win_stats
        ret_val_q["window_stats"] = win_stats

    if config.measure_type == "dt":
        ret_val = ret_val_p
    elif config.measure_type == "am":
        ret_val = ret_val_q

    if plot:
        plot_adjoint_source(observed, synthetic, ret_val["adjoint_source"],
                            ret_val["misfit"], windows, VERBOSE_NAME)

    return ret_val
