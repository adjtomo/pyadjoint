#!/usr/bin/env pythonG
# -*- coding: utf-8 -*-
"""
Automated tests for all defined double-difference
adjoint sources.
using 2 shifted Gaussian signals

:copyright:
    Yanhua O. Yuan (yanhuay@princeton.edu),2016
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import pyadjoint
from obspy.core.trace import Trace


def gaussian_obs_syn():
    """
        gaussian signals for bench mark tests
    """

    syn = Trace()
    obs = Trace()
    syn.stats.channel = 'BHZ'
    obs.stats.channel = 'BHZ'
    syn.stats.delta = 1.0
    obs.stats.delta = 1.0
    syn.stats.npts = 100
    obs.stats.npts = 100

    t = np.arange(0, syn.stats.npts, syn.stats.delta)
    mu = 50
    sigma = 3.5
    syn.data = np.exp(-np.power(t - mu, 2.) / (2 * np.power(sigma, 2.)))
    mu = 52
    obs.data = np.exp(-np.power(t - mu, 2.) / (2 * np.power(sigma, 2.)))

    config = pyadjoint.Config(min_period=0.1, max_period=10.0,
                              wtr_env=0.2)

    window = [[45, 52]]

    return obs, syn, config, window

if __name__ == "__main__":

    obs, syn, config, window = gaussian_obs_syn()

    adj_src = 'multitaper_misfit'
    adj_src = 'cc_traveltime_misfit'

    a_src = pyadjoint.calculate_adjoint_source(
        adj_src_type=adj_src, observed=obs,
        synthetic=syn, config=config, window=window,
        adjoint_src=True, plot=True)
