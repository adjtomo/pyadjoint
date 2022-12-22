#!/usr/bin/env python3
"""
Cross correlation traveltime misfit and associated adjoint source.

:copyright:
    adjtomo Dev Team (adjtomo@gmail.com), 2022
    Youyi Ruan (youyir@princeton.edu) 2016
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
import numpy as np
from scipy.integrate import simps

from pyadjoint.utils import (window_taper, generic_adjoint_source_plot,
                             xcorr_shift, cc_error)


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


def calculate_adjoint_source(observed, synthetic, config, window,
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
    assert(config.__class__.__name__ == "ConfigCCTraveltime"), \
        "Incorrect configuration class passed to CCTraveltime misfit"

    # Allow for measurement types related to `dt` (p) and `dlna` (q)
    ret_val_p = {}
    ret_val_q = {}

    # List of windows and some measurement values for each
    win_stats_p = []
    win_stats_q = []

    # Initiate constants and empty return values to fill
    nlen_data = len(synthetic.data)
    deltat = synthetic.stats.delta

    fp = np.zeros(nlen_data)
    fq = np.zeros(nlen_data)

    misfit_sum_p = 0.0
    misfit_sum_q = 0.0

    # Loop over time windows and calculate misfit for each window range
    for wins in window:
        left_window_border = wins[0]
        right_window_border = wins[1]

        left_sample = int(np.floor(left_window_border / deltat)) + 1
        nlen = int(np.floor((right_window_border -
                             left_window_border) / deltat)) + 1
        right_sample = left_sample + nlen

        d = np.zeros(nlen)
        s = np.zeros(nlen)

        d[0:nlen] = observed.data[left_sample:right_sample]
        s[0:nlen] = synthetic.data[left_sample:right_sample]

        # All adjoint sources will need some kind of windowing taper
        window_taper(d, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        window_taper(s, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)

        i_shift = xcorr_shift(d, s)
        t_shift = i_shift * deltat

        cc_dlna = 0.5 * np.log(sum(d[0:nlen] * d[0:nlen]) /
                               sum(s[0:nlen] * s[0:nlen]))

        # Determine if a calculated error value should be used for normalization
        if config.use_cc_error:
            sigma_dt, sigma_dlna = cc_error(
                d1=d, d2=s, deltat=deltat, cc_shift=i_shift, cc_dlna=cc_dlna,
                dt_sigma_min=config.dt_sigma_min,
                dlna_sigma_min=config.dlna_sigma_min
            )
        else:
            sigma_dt, sigma_dlna = 1., 1.

        # Calculate the misfit for both time shift and amplitude anomaly
        misfit_p = 0.5 * (t_shift / sigma_dt) ** 2
        misfit_q = 0.5 * (cc_dlna / sigma_dlna) ** 2
        misfit_sum_p += misfit_p
        misfit_sum_q += misfit_q

        # Calculate adjoint sources for both time shift and amplitude anomaly
        dsdt = np.gradient(s, deltat)
        nnorm = simps(y=dsdt * dsdt, dx=deltat)
        fp[left_sample:right_sample] = dsdt[:] * t_shift / nnorm / sigma_dt ** 2

        mnorm = simps(y=s * s, dx=deltat)
        fq[left_sample:right_sample] = \
            -1.0 * s[:] * cc_dlna / mnorm / sigma_dlna ** 2

        # Store some information for each window
        win_stats_q.append(
            {"left": left_window_border, "right": right_window_border,
             "measurement_type": config.measure_type,
             "dlna": cc_dlna,  "misfit_dlna": misfit_q,
             "sigma_dlna": sigma_dlna,
             }
        )
        win_stats_q.append(
            {"left": left_window_border, "right": right_window_border,
             "measurement_type": config.measure_type,
             "tshift": t_shift,  "misfit_dt": misfit_p, "sigma_dt": sigma_dt,
             }
        )

    # Keep track of both misfit values
    ret_val_p["misfit"] = misfit_sum_p
    ret_val_q["misfit"] = misfit_sum_q

    if adjoint_src is True:
        ret_val_p["adjoint_source"] = fp[::-1]
        ret_val_q["adjoint_source"] = fq[::-1]
    if window_stats is True:
        ret_val_p["window_stats"] = win_stats_p
        ret_val_q["window_stats"] = win_stats_q

    if config.measure_type == "dt":
        ret_val = ret_val_p
    elif config.measure_type == "am":
        ret_val = ret_val_q

    if plot:
        generic_adjoint_source_plot(
            observed, synthetic, ret_val["adjoint_source"], ret_val["misfit"],
            window, VERBOSE_NAME
        )

    return ret_val
