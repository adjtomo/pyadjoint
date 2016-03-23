#!/usr/bin/env python
# -*- encoding: utf8 -*-
"""
Double-Difference Cross correlation traveltime misfit.

:copyright:
    Yanhua O. Yuan(yanhuay@princeton.edu) 2016
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from ..utils import window_taper,  generic_adjoint_source_plot
from ..cc import _xcorr_shift, cc_adj_DD

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
:func:`~obspy.signal.cross_correlation.xcorrPickCorrection` function and the
corresponding
`Tutorial <http://docs.obspy.org/tutorial/code_snippets/xcorr_pick_correction.html>`_.


The adjoint source for the same receiver and component is then given by

.. math::

    f^{\dagger}(t) = - \left[ T^{obs} - T(\mathbf{m}) \right] ~ \frac{1}{N} ~
    \partial_t \mathbf{s}(T - t, \mathbf{m})

For the sake of simplicity we omit the spatial Kronecker delta and define
the adjoint source as acting solely at the receiver's location. For more
details, please see [Yuan2016].


:math:`N` is a normalization factor given by


.. math::

    N = \int_0^T ~ \mathbf{s}(t, \mathbf{m}) ~
    \partial^2_t \mathbf{s}(t, \mathbf{m}) dt

This particular implementation here uses
`Simpson's rule <http://en.wikipedia.org/wiki/Simpson's_rule>`_
to evaluate the definite integral.
"""  # NOQA

# Optional: document any additional parameters this particular adjoint sources
# receives in addition to the ones passed to the central adjoint source
# calculation function. Make sure to indicate the default values. This is a
# bit redundant but the only way I could figure out to make it work with the
#  rest of the architecture.
ADDITIONAL_PARAMETERS = r"""
**taper_percentage** (:class:`float`)
    Decimal percentage of taper at one end (ranging from ``0.0`` (0%) to
    ``0.5`` (50%)). Defauls to ``0.15``.

**taper_type** (:class:`float`)
    The taper type, supports anything :meth:`obspy.core.trace.Trace.taper`
    can use. Defaults to ``"hann"``.
"""


def calculate_adjoint_source_DD(observed1, synthetic1, observed2, synthetic2,
                             config, window1, window2,
                             adjoint_src, figure):  # NOQA

    ret_val_p1 = {}
    ret_val_p2 = {}

    nlen_data = len(synthetic1.data)
    deltat = synthetic1.stats.delta

    fp1 = np.zeros(nlen_data)
    fp2 = np.zeros(nlen_data)

    misfit_sum_p = 0.0

    # ===
    # loop over time windows
    # ===
    for wins1, wins2 in zip(window1, window2):
        left_window_border_1 = wins1[0]
        right_window_border_1 = wins1[1]
        left_window_border_2 = wins2[0]
        right_window_border_2 = wins2[1]

        left_sample_1 = int(np.floor(left_window_border_1 / deltat)) + 1
        left_sample_2 = int(np.floor(left_window_border_2 / deltat)) + 1
        nlen1 = int(np.floor((right_window_border_1 -
                             left_window_border_1) / deltat)) + 1
        nlen2 = int(np.floor((right_window_border_2 -
                             left_window_border_2) / deltat)) + 1

        if(nlen1 != nlen2):
            continue
        nlen = nlen1

        right_sample_1 = left_sample_1 + nlen
        right_sample_2 = left_sample_2 + nlen

        d1 = np.zeros(nlen)
        s1 = np.zeros(nlen)
        d2 = np.zeros(nlen)
        s2 = np.zeros(nlen)

        d1[0:nlen] = observed1.data[left_sample_1:right_sample_1]
        s1[0:nlen] = synthetic1.data[left_sample_1:right_sample_1]
        d2[0:nlen] = observed2.data[left_sample_2:right_sample_2]
        s2[0:nlen] = synthetic2.data[left_sample_2:right_sample_2]

        # All adjoint sources will need some kind of windowing taper
        window_taper(d1, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        window_taper(s1, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        window_taper(d2, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        window_taper(s2, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)

        shift_obs = _xcorr_shift(d1, d2)
        shift_syn = _xcorr_shift(s1, s2)
        dd_shift = shift_syn - shift_obs

        # misfit and adjoint source
        dt_adj1, dt_adj2, misfit_dt = cc_adj_DD(
            s1, s2, shift_syn, dd_shift, deltat)

        # taper adjoint source again
        window_taper(dt_adj1, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        window_taper(dt_adj2, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)

        misfit_sum_p += misfit_dt
        fp1[left_sample_1:right_sample_1] = dt_adj1[0:nlen]
        fp2[left_sample_2:right_sample_2] = dt_adj2[0:nlen]

    ret_val_p1["misfit"] = misfit_sum_p
    ret_val_p2["misfit"] = misfit_sum_p

    if adjoint_src is True:
        ret_val_p1["adjoint_source"] = fp1[::-1]
        ret_val_p2["adjoint_source"] = fp2[::-1]

    if config.measure_type == "dt":
        if figure:
            generic_adjoint_source_plot(observed1, synthetic1,
                                        ret_val_p1["adjoint_source"],
                                        ret_val_p1["misfit"],
                                        window1, VERBOSE_NAME)
            generic_adjoint_source_plot(observed2, synthetic2,
                                        ret_val_p2["adjoint_source"],
                                        ret_val_p2["misfit"],
                                        window2, VERBOSE_NAME)

        # ===
        # YY: only return adjoint source of
        # master trace to match adjoint_source wrapper
        # ===
        return ret_val_p1
