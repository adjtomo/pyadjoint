#!/usr/bin/env pythonG
# -*- coding: utf-8 -*-
"""
Automated tests for all defined double-difference
adjoint sources.
using four shifted Gaussian signals

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

    syn1 = Trace()
    obs1 = Trace()
    syn2 = Trace()
    obs2 = Trace()
    syn1.stats.channel = 'BHZ'
    obs1.stats.channel = 'BHZ'
    syn2.stats.channel = 'BHZ'
    obs2.stats.channel = 'BHZ'
    syn1.stats.delta = 1.0
    obs1.stats.delta = 1.0
    syn2.stats.delta = 1.0
    obs2.stats.delta = 1.0
    syn1.stats.npts = 100
    obs1.stats.npts = 100
    syn2.stats.npts = 100
    obs2.stats.npts = 100

    t = np.arange(0, syn1.stats.npts, syn1.stats.delta)
    mu = 50
    sigma = 3.5
    syn1.data = np.exp(-np.power(t - mu, 2.) / (2 * np.power(sigma, 2.)))
    mu = 52
    obs1.data = np.exp(-np.power(t - mu, 2.) / (2 * np.power(sigma, 2.)))
    mu = 51
    syn2.data = np.exp(-np.power(t - mu, 2.) / (2 * np.power(sigma, 2.)))
    mu = 52
    obs2.data = np.exp(-np.power(t - mu, 2.) / (2 * np.power(sigma, 2.)))

    config = pyadjoint.Config(min_period=0.1, max_period=10.0,
                              wtr_env=0.2)

    window1 = [[30, 70]]
    window2 = [[30, 70]]

    return obs1, syn1, obs2, syn2, config, window1, window2

if __name__ == "__main__":

    obs1, syn1, obs2, syn2, config, window1, window2 = gaussian_obs_syn()

    adj_src_DD = 'cc_traveltime_misfit_DD'

    a_src = pyadjoint.calculate_adjoint_source_DD(
        adj_src_type=adj_src_DD, observed1=obs1,
        synthetic1=syn1, observed2=obs2, synthetic2=syn2,
        config=config, window1=window1, window2=window2,
        adjoint_src=True, plot=True)
