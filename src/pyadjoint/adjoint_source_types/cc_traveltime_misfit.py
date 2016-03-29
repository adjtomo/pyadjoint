#!/usr/bin/env python
# -*- encoding: utf8 -*-
"""
Cross correlation traveltime misfit.

:copyright:
    Youyi Ruan (youyir@princeton.edu) 2016
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from ..utils import window_taper,  generic_adjoint_source_plot
from ..cc import _xcorr_shift, cc_error, cc_adj

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
details, please see [Tromp2005]_ and [Bozdag2011]_.


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


def calculate_adjoint_source(observed, synthetic, config, window,
                             adjoint_src, figure):  # NOQA

    ret_val_p = {}
    ret_val_q = {}

    nlen_data = len(synthetic.data)
    deltat = synthetic.stats.delta

    fp = np.zeros(nlen_data)
    fq = np.zeros(nlen_data)

    misfit_sum_p = 0.0
    misfit_sum_q = 0.0

    # ===
    # loop over time windows
    # ===
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

        cc_shift = _xcorr_shift(d, s)

        cc_dlna = 0.5 * np.log(sum(d[0:nlen]*d[0:nlen]) /
                               sum(s[0:nlen]*s[0:nlen]))

        sigma_dt, sigma_dlna = cc_error(d, s, deltat, cc_shift, cc_dlna,
                                        config.dt_sigma_min,
                                        config.dlna_sigma_min)
        # calculate c.c. adjoint source
        fp_t, fq_t, misfit_p, misfit_q =\
            cc_adj(s, cc_shift, cc_dlna, deltat,
                   sigma_dt, sigma_dlna)

        # YY: All adjoint sources will need windowing taper again
        window_taper(fp_t, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        window_taper(fq_t, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)

        misfit_sum_p += misfit_p
        misfit_sum_q += misfit_q

        fp[left_sample:right_sample] = fp_t[:]
        fq[left_sample:right_sample] = fq_t[:]

    ret_val_p["misfit"] = misfit_sum_p
    ret_val_q["misfit"] = misfit_sum_q

    if adjoint_src is True:
        ret_val_p["adjoint_source"] = fp[::-1]
        ret_val_q["adjoint_source"] = fq[::-1]

    if config.measure_type == "dt":
        if figure:
            generic_adjoint_source_plot(observed, synthetic,
                                        ret_val_p["adjoint_source"],
                                        ret_val_p["misfit"],
                                        window, VERBOSE_NAME)

        return ret_val_p

    if config.measure_type == "am":
        if figure:
            generic_adjoint_source_plot(observed, synthetic,
                                        ret_val_q["adjoint_source"],
                                        ret_val_q["misfit"],
                                        window, VERBOSE_NAME)

        return ret_val_q
