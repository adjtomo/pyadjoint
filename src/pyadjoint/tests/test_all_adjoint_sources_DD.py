#!/usr/bin/env pythonG
# -*- coding: utf-8 -*-
"""
Automated tests for all defined double-difference
adjoint sources. Essentially just checks
that they all work and do something.

:copyright:
    Yanhua O. Yuan (yanhuay@princeton.edu), 2016
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import pytest
import pyadjoint


@pytest.fixture(params=list(pyadjoint.AdjointSource._ad_srcs_DD.keys()))
def adj_src(request):
    """
    Fixture returning the name of all adjoint sources.
    """
    return request.param


def test_normal_adjoint_source_calculation(adj_src):
    """
    Make sure everything at least runs. Executed for every adjoint source type.
    """
    config = pyadjoint.Config(min_period=30.0, max_period=75.0,
                              lnpt=15,
                              transfunc_waterlevel=1.0E-10,
                              water_threshold=0.02,
                              ipower_costaper=10,
                              min_cycle_in_window=3,
                              taper_percentage=0.3,
                              taper_type='hann',
                              mt_nw=4,
                              phase_step=1.5,
                              use_cc_error=False,
                              use_mt_error=False)

    obs, syn = pyadjoint.utils.get_example_data()
    obs = obs.select(component="Z")[0]
    syn = syn.select(component="Z")[0]
    # start, end = pyadjoint.utils.EXAMPLE_DATA_PDIFF

    window1 = [[1993., 2063.0]]
    window2 = [[2193., 2263.0]]

    a_src = pyadjoint.calculate_adjoint_source_DD(
            adj_src_type=adj_src, observed1=obs,
            synthetic1=syn, observed2=obs, synthetic2=syn,
            config=config, window1=window1, window2=window2,
            adjoint_src=True, plot=False)

    assert a_src.adjoint_source.any()
    assert a_src.misfit >= 0.0

    assert isinstance(a_src.adjoint_source, np.ndarray)


def test_no_adjoint_src_calculation_is_honored(adj_src):
    """
    If no adjoint source is requested, it should not be returned/calculated.
    """
    config = pyadjoint.Config(min_period=30.0, max_period=75.0,
                              lnpt=15,
                              transfunc_waterlevel=1.0E-10,
                              water_threshold=0.02,
                              ipower_costaper=10,
                              min_cycle_in_window=3,
                              taper_percentage=0.3,
                              taper_type='hann',
                              mt_nw=4,
                              phase_step=1.5,
                              use_cc_error=False,
                              use_mt_error=False)

    obs, syn = pyadjoint.utils.get_example_data()
    obs = obs.select(component="Z")[0]
    syn = syn.select(component="Z")[0]
    window1 = [[1993., 2063.0]]
    window2 = [[2193., 2263.0]]

    a_src = pyadjoint.calculate_adjoint_source_DD(
            adj_src_type=adj_src, observed1=obs,
            synthetic1=syn, observed2=obs, synthetic2=syn,
            config=config, window1=window1, window2=window2,
            adjoint_src=False, plot=False)

    # start, end = pyadjoint.utils.EXAMPLE_DATA_PDIFF
    # a_src = pyadjoint.calculate_adjoint_source(
    #    adj_src, obs, syn, 20, 100, start, end, adjoint_src=False)

    assert a_src.adjoint_source is None
    # assert a_src.misfit >= 0.0

    # But the misfit should nonetheless be identical as if the adjoint
    # source would have been calculated.

    # assert a_src.misfit == pyadjoint.calculate_adjoint_source(
    #    adj_src_type=adj_src, observed=obs,
    #    synthetic=syn, config=config, window=window,
    #    adjoint_src=True, plot=False).misfit

    # pyadjoint.calculate_adjoint_source(
    # adj_src, obs, syn, 20, 100, start, end, adjoint_src=True).misfit
