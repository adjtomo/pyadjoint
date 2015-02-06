#!/usr/bin/env pythonG
# -*- coding: utf-8 -*-
"""
Automated tests for all defined adjoint sources. Essentially just checks
that they all work and do something.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import pytest
import pyadjoint


@pytest.fixture(params=list(pyadjoint.AdjointSource._ad_srcs.keys()))
def adj_src(request):
    """
    Fixture returning the name of all adjoint sources.
    """
    return request.param


def test_normal_adjoint_source_calculation(adj_src):
    """
    Make sure everything at least runs. Executed for every adjoint source type.
    """
    obs, syn = pyadjoint.utils.get_example_data()
    obs = obs.select(component="Z")[0]
    syn = syn.select(component="Z")[0]
    start, end = pyadjoint.utils.EXAMPLE_DATA_PDIFF
    a_src = pyadjoint.calculate_adjoint_source(
        adj_src, obs, syn, 20, 100, start, end)

    assert a_src.adjoint_source.any()
    assert a_src.misfit >= 0.0

    assert isinstance(a_src.adjoint_source, np.ndarray)


def test_no_adjoint_src_calculation_is_honored(adj_src):
    """
    If no adjoint source is requested, it should not be returned/calculated.
    """
    obs, syn = pyadjoint.utils.get_example_data()
    obs = obs.select(component="Z")[0]
    syn = syn.select(component="Z")[0]
    start, end = pyadjoint.utils.EXAMPLE_DATA_PDIFF
    a_src = pyadjoint.calculate_adjoint_source(
        adj_src, obs, syn, 20, 100, start, end, adjoint_src=False)

    assert a_src.adjoint_source is None
    assert a_src.misfit >= 0.0

    # But the misfit should nonetheless be identical as if the adjoint
    # source would have been calculated.
    assert a_src.misfit == pyadjoint.calculate_adjoint_source(
        adj_src, obs, syn, 20, 100, start, end, adjoint_src=True).misfit
