#!/usr/bin/env pythonG
# -*- coding: utf-8 -*-
"""
Automated tests for all defined adjoint sources.
using two shifted Gaussian signals

:copyright:
    Yanhua O. Yuan (yanhuay@princeton.edu),2016
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import pytest
import pyadjoint
from obspy.core.trace import Trace


@pytest.fixture(params=list(pyadjoint.AdjointSource._ad_srcs.keys()))
def adj_src(request):
    """
    Fixture returning the name of all adjoint sources.
    """
    return request.param


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

    window = [[30, 70]]

    return obs, syn, config, window


def test_normal_adjoint_source_calculation(adj_src):
    """
    Make sure everything at least runs. Executed for every adjoint source type.
    """
    obs, syn, config, window = gaussian_obs_syn()

    a_src = pyadjoint.calculate_adjoint_source(
        adj_src_type=adj_src, observed=obs,
        synthetic=syn, config=config, window=window,
        adjoint_src=True, plot=False)

    assert a_src.adjoint_source.any()
    assert a_src.misfit >= 0.0

    assert isinstance(a_src.adjoint_source, np.ndarray)


def test_no_adjoint_src_calculation_is_honored(adj_src):
    """
    If no adjoint source is requested, it should not be returned/calculated.
    """
    obs, syn, config, window = gaussian_obs_syn()

    a_src = pyadjoint.calculate_adjoint_source(
        adj_src_type=adj_src, observed=obs,
        synthetic=syn, config=config, window=window,
        adjoint_src=False, plot=False)

    assert a_src.adjoint_source is None
    assert a_src.misfit >= 0.0
